"""Compare matched LightGBM and XGBoost predictions without refitting either model."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

from .experiment import sha256_file
from .configuration import command_config
from .reporting import classification_metrics, format_number, prediction_outcome
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent


def read_predictions(report_path, split, metadata, threshold):
    path = report_path.parent / f'{split}_predictions.csv'
    if sha256_file(path) != metadata.get('prediction_csv_sha256'):
        raise ValueError(f'Prediction CSV hash mismatch for {path}; rerun evaluation')
    records = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['source_file'], int(row['source_line']))
            label, probability = int(row['label']), float(row['malicious_probability'])
            if key in records or key[1] < 1:
                raise ValueError('Duplicate or invalid sample source identity')
            if label not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid prediction label/probability')
            outcome = prediction_outcome(label, probability, threshold)
            if row.get('outcome') and row['outcome'] != outcome:
                raise ValueError('Prediction outcome is inconsistent with the recorded threshold')
            records[key] = dict(row, label=label, probability=probability, outcome=outcome)
    if len(records) != metadata['rows'] or not records:
        raise ValueError('Prediction sample count differs from evaluation report')
    if sha256_file(path) != metadata['prediction_csv_sha256']:
        raise ValueError('Prediction CSV hash changed while reading')
    return records


def plot_comparison(labels, scores, metrics, split, directory):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for index, (backend, probabilities) in enumerate(scores.items()):
        item = metrics[backend]
        if item['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(labels, probabilities)
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            axes[0].plot(fpr, tpr, label=f'{backend}: AUC={item["roc_auc"]:.4f}')
            axes[1].step(recall, precision, where='post', label=f'{backend}: AP={item["average_precision"]:.4f}')
        values = [item['confusion'][key] for key in ('tp', 'tn', 'fp', 'fn')]
        axes[2].bar([x + (index - 0.5) * 0.36 for x in range(4)], values, width=0.36, label=backend)
    if len(set(labels)) == 2:
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].axhline(sum(labels) / len(labels), ls='--', color='gray')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    else:
        for axis in axes[:2]:
            axis.text(0.5, 0.5, 'Undefined for a single-class split', ha='center', va='center', fontsize=9)
    axes[0].set(title='ROC', xlabel='False-positive rate', ylabel='True-positive rate')
    axes[1].set(title='Precision–recall', xlabel='Recall', ylabel='Precision')
    for axis in axes[:2]:
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.grid(alpha=0.2)
    axes[2].set(title='Correct decisions and errors', ylabel='Sample count',
                xticks=range(4), xticklabels=['TP', 'TN', 'FP', 'FN'])
    axes[2].legend(fontsize=8)
    figure.suptitle(f'LightGBM vs XGBoost — {split} (same {len(labels):,} labeled rows)')
    for suffix in ('png', 'svg'):
        figure.savefig(directory / f'{split}_comparison.{suffix}', dpi=180)
    plt.close(figure)


def comparison_markdown(report):
    lines = ['# LightGBM vs XGBoost: matched-sample comparison', '',
             f'Generated (UTC): {report["generated_at"]}', '',
             f'Fixed threshold for both models: {report["threshold"]}', '',
             f'Training corpus equivalence: {report["training_dataset_match"]}', '',
             'TP = malicious file detected correctly; TN = benign file accepted correctly; '
             'FP = benign file incorrectly flagged; FN = malicious file missed.', '',
             '## Model provenance', '', '| Model | Model SHA-256 | Evaluation device | CPU threads |',
             '| --- | --- | --- | ---: |']
    for backend, model in report['models'].items():
        lines.append(f'| {backend} | `{model["model_sha256"]}` | {model["device"]} | {model["threads"]} |')
    lines += ['', '## Algorithm structure and parameters', '']
    for backend, model in report['models'].items():
        algorithm = model.get('algorithm')
        if algorithm:
            depth = algorithm['tree_depth']
            lines += [f'### {backend}', '',
                      f'Objective: {algorithm["objective"]}; trees: {algorithm["number_of_trees"]}; '
                      f'actual rounds: {algorithm["boosting_rounds"]}; observed depth '
                      f'min/mean/max: {depth["min"]}/{format_number(depth["mean"])}/{depth["max"]}.', '']
        if model.get('training_parameters'):
            lines += ['Verified training parameters:', '', '```json',
                      json.dumps(model['training_parameters'], indent=2), '```', '']
    for split, result in report['splits'].items():
        lines += ['', f'## {split}', '', f'Labeled rows: {result["rows"]}; input SHA-256: `{result["dataset_sha256"]}`', '',
                  '| Model | Correct | TP | TN | FP | FN | Recall | FPR | Precision | F1 | ROC-AUC | AP |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for backend, metrics in result['models'].items():
            c = metrics['confusion']
            values = [backend, str(c['tp'] + c['tn'])] + [str(c[key]) for key in ('tp', 'tn', 'fp', 'fn')]
            values += [format_number(metrics[key]) for key in ('recall', 'false_positive_rate', 'precision', 'f1', 'roc_auc', 'average_precision')]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines += ['', '### Which model got the same file right?', '', '| Result | Count |', '| --- | ---: |']
        lines += [f'| {key.replace("_", " ")} | {value} |' for key, value in result['paired_correctness'].items()]
        lines += ['', '### Malicious-file detection overlap', '', '| Result | Count |', '| --- | ---: |']
        lines += [f'| {key.replace("_", " ")} | {value} |' for key, value in result['detection_overlap'].items()]
        lines += ['', '### Benign-file false-positive overlap', '', '| Result | Count |', '| --- | ---: |']
        lines += [f'| {key.replace("_", " ")} | {value} |' for key, value in result['false_positive_overlap'].items()]
        lines += ['', f'![Matched-sample curves and confusion counts]({split}_comparison.png)', '',
                  f'[Every sample]({split}_samples.csv) · [Model disagreements]({split}_disagreements.csv) · '
                  f'[Errors by either model]({split}_errors.csv) · [Detected malicious samples]({split}_detections.csv)', '',
                  'Each row identifies the original sample SHA-256 (when available), source JSONL '
                  'file and 1-based line number, both probability scores, TP/TN/FP/FN outcomes, '
                  'and which model was correct. These are feature records, not original PDF paths.', '']
    lines += ['## Methods and limits', ''] + ['- ' + note for note in report['notes']]
    return '\n'.join(lines) + '\n'


def compare_reports(lightgbm_path, xgboost_path, output_dir):
    paths = {'lightgbm': Path(lightgbm_path), 'xgboost': Path(xgboost_path)}
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    inputs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, item in inputs.items():
        if item.get('backend') != name or item.get('schema_version', 0) < 3:
            raise ValueError(f'Rerun {name} evaluation with this version to add sample identities and prediction hashes')
    left, right = inputs['lightgbm'], inputs['xgboost']
    if left['threshold'] != right['threshold']:
        raise ValueError('Reports use different thresholds; evaluate both with the same fixed threshold')
    if not left.get('feature_extractor_sha256') or left['feature_extractor_sha256'] != right.get('feature_extractor_sha256'):
        raise ValueError('Reports use different or missing feature-extractor hashes')
    if not left['splits'] or set(left['splits']) != set(right['splits']):
        raise ValueError('Reports cover different dataset splits')
    training_hashes = [(item.get('training') or {}).get('dataset_sha256') for item in inputs.values()]
    training_match = None if not all(training_hashes) else training_hashes[0] == training_hashes[1]
    if training_match is False:
        raise ValueError('Models were trained on different datasets; retrain on the same training corpus')
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(), 'threshold': left['threshold'],
        'training_dataset_match': training_match,
        'models': {name: {'model_sha256': item['model_sha256'],
                          'evaluation_report_sha256': source_hashes[name],
                          'device': item.get('execution_device', {}).get('resolved', 'unrecorded'),
                          'threads': item.get('protocol', {}).get('threads', 'unrecorded'),
                          'algorithm': item.get('algorithm'),
                          'training_parameters': (item.get('training') or {}).get('parameters')}
                   for name, item in inputs.items()},
        'notes': [
            'Input dataset/feature hashes, thresholds, sample source identities, SHA-256 identifiers, and ground-truth labels must agree before pairing. Prediction CSV integrity is checked.',
            'Correctness and errors are measured against supplied binary labels; the report does not independently verify malware ground truth or inspect original files.',
            'Rows are matched by source JSONL path and line, preserving duplicate SHA-256 rows rather than silently collapsing repeated samples.',
            'No threshold tuning, significance test, or declaration of a universal winner is performed. Consider false positives and missed detections together; class prevalence affects precision.',
            'The shared challenge set is not PDF-only. Without benign challenge samples, FPR and ROC-AUC are undefined.',
            'Full individual reports contain Wilson rate intervals, parameters, timing, calibration, and provenance. Hardware, feature handling, model capacity, and single-seed effects limit causal comparisons.',
        ],
        'splits': {},
    }
    if training_match is None:
        report['notes'].append('Training corpus equivalence is UNVERIFIED because one or both models lack training manifests.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='comparison-', dir=output_dir) as temporary:
        staging = Path(temporary)
        for split in left['splits']:
            left_meta, right_meta = left['splits'][split], right['splits'][split]
            if not left_meta.get('dataset_sha256') or left_meta['dataset_sha256'] != right_meta.get('dataset_sha256'):
                raise ValueError(f'{split} dataset hashes differ; evaluate both on identical input files')
            records = {name: read_predictions(paths[name], split, item['splits'][split], left['threshold'])
                       for name, item in inputs.items()}
            if set(records['lightgbm']) != set(records['xgboost']):
                raise ValueError(f'{split} sample source identities differ')
            labels, rows = [], []
            scores = {'lightgbm': [], 'xgboost': []}
            paired = Counter(dict.fromkeys(['both_correct', 'only_lightgbm_correct', 'only_xgboost_correct', 'both_wrong'], 0))
            detections = Counter(dict.fromkeys(['both_detected', 'only_lightgbm_detected', 'only_xgboost_detected', 'both_missed'], 0))
            false_positives = Counter(dict.fromkeys(['both_false_positive', 'only_lightgbm_false_positive', 'only_xgboost_false_positive', 'both_correct_benign'], 0))
            for key in sorted(records['lightgbm']):
                a, b = records['lightgbm'][key], records['xgboost'][key]
                if a['label'] != b['label'] or a.get('sha256', '') != b.get('sha256', ''):
                    raise ValueError(f'{split} sample labels or SHA-256 identifiers differ at {key}')
                label = a['label']
                labels.append(label)
                correct_a, correct_b = a['outcome'] in ('TP', 'TN'), b['outcome'] in ('TP', 'TN')
                group = ('both_correct' if correct_b else 'only_lightgbm_correct') if correct_a else ('only_xgboost_correct' if correct_b else 'both_wrong')
                paired[group] += 1
                if label:
                    detection = ('both_detected' if correct_b else 'only_lightgbm_detected') if correct_a else ('only_xgboost_detected' if correct_b else 'both_missed')
                    detections[detection] += 1
                else:
                    fp = ('both_correct_benign' if correct_b else 'only_xgboost_false_positive') if correct_a else ('only_lightgbm_false_positive' if correct_b else 'both_false_positive')
                    false_positives[fp] += 1
                row = {'source_file': key[0], 'source_line': key[1], 'sha256': a.get('sha256', ''), 'label': label,
                       'correctness_group': group}
                for name, record in [('lightgbm', a), ('xgboost', b)]:
                    scores[name].append(record['probability'])
                    row[f'{name}_probability'] = record['probability']
                    row[f'{name}_outcome'] = record['outcome']
                rows.append(row)
            metrics = {name: classification_metrics(labels, probabilities, left['threshold']) for name, probabilities in scores.items()}
            result = {'rows': len(labels), 'dataset_sha256': left_meta['dataset_sha256'], 'models': metrics,
                      'paired_correctness': dict(paired), 'detection_overlap': dict(detections),
                      'false_positive_overlap': dict(false_positives)}
            for suffix in ('samples', 'errors', 'disagreements', 'detections'):
                with (staging / f'{split}_{suffix}.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    for row in rows:
                        selected = (suffix == 'samples' or
                                    (suffix == 'errors' and row['correctness_group'] != 'both_correct') or
                                    (suffix == 'disagreements' and row['lightgbm_outcome'] != row['xgboost_outcome']) or
                                    (suffix == 'detections' and row['label'] == 1 and
                                     (row['lightgbm_outcome'] == 'TP' or row['xgboost_outcome'] == 'TP')))
                        if selected:
                            writer.writerow(row)
            plot_comparison(labels, scores, metrics, split, staging)
            report['splits'][split] = result
        if any(sha256_file(path) != source_hashes[name] for name, path in paths.items()):
            raise ValueError('Evaluation report changed during comparison; rerun with stable inputs')
        (staging / 'comparison.md').write_text(comparison_markdown(report))
        (staging / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for path in staging.iterdir():
            if path.name != 'comparison.json':
                path.replace(output_dir / path.name)
        (staging / 'comparison.json').replace(output_dir / 'comparison.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = command_config(parser)
    report_root = ROOT / config['paths']['report_root']
    parser.add_argument('--lightgbm-report', type=Path, default=report_root / 'evaluation.json')
    parser.add_argument('--xgboost-report', type=Path, default=report_root / 'xgboost/evaluation.json')
    parser.add_argument('--output-dir', type=Path, default=report_root / 'comparison')
    args = parser.parse_args()
    compare_reports(args.lightgbm_report, args.xgboost_report, args.output_dir)
    print(f'Comparison report and per-file CSVs: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
