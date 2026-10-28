"""Evaluate a trained binary model on held-out test and shared challenge features."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from thrember import PEFeatureExtractor

from train import training_records

ROOT = Path(__file__).resolve().parent


def classification_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    tn = int(np.sum((labels == 0) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tp = int(np.sum((labels == 1) & predicted))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    both_classes = bool(tn + fp) and bool(tp + fn)
    return {
        'rows': len(labels), 'labels': {'0': tn + fp, '1': tp + fn},
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'accuracy': ratio(tp + tn, len(labels)),
        'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
        'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'false_positive_rate': ratio(fp, fp + tn),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if both_classes else None,
        'average_precision': float(average_precision_score(labels, probabilities)) if both_classes else None,
        'ranking_metrics_note': None if both_classes else 'Requires both benign and malicious labels.',
    }


def evaluate_split(model, extractor, paths, batch_size, threads, threshold):
    labels, scores, batch = [], [], []

    def predict_batch():
        features = np.asarray(batch, dtype=np.float32)
        predictions = np.asarray(model.predict(features, num_threads=threads))
        if predictions.ndim != 1 or not np.all(np.isfinite(predictions)) or np.any((predictions < 0) | (predictions > 1)):
            raise ValueError('Expected finite binary model probabilities between 0 and 1')
        scores.extend(predictions.tolist())
        batch.clear()

    for record in training_records(paths):
        labels.append(record['label'])
        batch.append(extractor.process_raw_features(record))
        if len(batch) == batch_size:
            predict_batch()
    if batch:
        predict_batch()
    if not labels:
        raise ValueError(f'No labeled evaluation records in {paths}')
    return classification_metrics(labels, scores, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/datasets')
    parser.add_argument('--model', type=Path, default=ROOT / 'data/models/ember2024/model.txt')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'data/reports')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1 or args.threads < 1:
        parser.error('threshold must be between 0 and 1; batch-size and threads must be positive')
    data = args.data_dir.resolve()
    split_paths = {
        'test': sorted(path for path in data.rglob('*_test.jsonl') if path.is_file() and 'challenge' not in path.name.lower()),
        'challenge': sorted(path for path in data.rglob('*.jsonl') if path.is_file() and 'challenge' in path.name.lower() and not path.name.endswith('_train.jsonl')),
    }
    for split, paths in split_paths.items():
        if not paths:
            parser.error(f'No {split} JSONL files under {data}; run the download command first')
    model = lgb.Booster(model_file=str(args.model))
    extractor = PEFeatureExtractor()
    if model.num_feature() != extractor.dim:
        parser.error('Model feature count does not match the bundled feature extractor')
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': str(args.model.resolve()),
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'threshold': args.threshold,
        'notes': [
            'Model is evaluated without fitting or selecting a threshold on held-out data.',
            'Challenge is the shared upstream set, not a PDF-only benchmark.',
            'Rows labeled null or -1 are excluded. Other invalid labels fail evaluation.',
            'Null metrics are undefined for the available classes or predictions.',
        ],
        'splits': {},
    }
    for split, paths in split_paths.items():
        print(f'Evaluating {split}: {len(paths)} files...', flush=True)
        metrics = evaluate_split(model, extractor, paths, args.batch_size, args.threads, args.threshold)
        metrics['files'] = [str(path.relative_to(data)) for path in paths]
        report['splits'][split] = metrics
    lines = ['# EMBER2024 evaluation', '', f'Generated: {report["generated_at"]}', '',
             f'Model SHA-256: `{report["model_sha256"]}`', '',
             f'Probability threshold: {args.threshold}', '',
             '| Split | Rows | Accuracy | Precision | Recall / detection rate | F1 | FPR | ROC-AUC | Average precision |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, metrics in report['splits'].items():
        values = [str(metrics['rows'])]
        for key in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate', 'roc_auc', 'average_precision'):
            values.append('N/A' if metrics[key] is None else f'{metrics[key]:.6f}')
        lines.append('| ' + ' | '.join([split] + values) + ' |')
        print(f'{split}: {json.dumps(metrics)}', flush=True)
    lines += ['', '| Split | TN | FP | FN | TP |', '| --- | ---: | ---: | ---: | ---: |']
    for split, metrics in report['splits'].items():
        lines.append('| ' + ' | '.join([split] + [str(metrics['confusion'][key]) for key in ('tn', 'fp', 'fn', 'tp')]) + ' |')
    lines += ['', *report['notes'], '']
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [('evaluation.json', json.dumps(report, indent=2, allow_nan=False) + '\n'),
                          ('evaluation.md', '\n'.join(lines))]:
        target = args.report_dir / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(target)
    print(f'Reports saved in {args.report_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
