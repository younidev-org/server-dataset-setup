"""Statistical summaries and publication-friendly figures for binary classifiers."""

import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)


def wilson_interval(successes, trials):
    """Two-sided 95% Wilson score interval; undefined without any trials."""
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


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
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        'rows': len(labels), 'labels': {'0': tn + fp, '1': tp + fn},
        'prevalence': ratio(tp + fn, len(labels)),
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'accuracy': ratio(tp + tn, len(labels)),
        'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
        'specificity': ratio(tn, tn + fp), 'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if both_classes else None,
        'matthews_correlation': ratio(tp * tn - fp * fn, mcc_denominator),
        'false_positive_rate': ratio(fp, fp + tn),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if both_classes else None,
        'average_precision': float(average_precision_score(labels, probabilities)) if both_classes else None,
        'brier_score': float(np.mean((probabilities - labels) ** 2)),
        'log_loss': float(log_loss(labels, probabilities, labels=[0, 1])),
        'confidence_intervals_95': {
            'accuracy': wilson_interval(tp + tn, len(labels)),
            'precision': wilson_interval(tp, tp + fp),
            'recall': wilson_interval(tp, tp + fn),
            'false_positive_rate': wilson_interval(fp, fp + tn),
            'specificity': wilson_interval(tn, tn + fp),
        },
        'ranking_metrics_note': None if both_classes else 'Requires both benign and malicious labels.',
    }


def save_figures(labels, probabilities, metrics, split, directory):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    if metrics['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
        axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
        axes[1].step(recall, precision, where='post', label=f"AP = {metrics['average_precision']:.4f}")
        axes[1].axhline(metrics['prevalence'], ls='--', color='gray', label='Class prevalence')
        axes[0].plot(metrics['false_positive_rate'], metrics['recall'], 'o', color='firebrick', label='Fixed threshold')
        if metrics['precision'] is not None:
            axes[1].plot(metrics['recall'], metrics['precision'], 'o', color='firebrick', label='Fixed threshold')
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    else:
        for axis in axes[:2]:
            axis.text(0.5, 0.5, 'Undefined for a single-class split', ha='center', va='center', fontsize=9)
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    # Equal-width reliability bins, with counts exported to JSON for interpretation.
    calibration = []
    assignments = np.minimum((probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = assignments == index
        if not np.any(selected):
            continue
        calibration.append({'count': int(selected.sum()),
                            'mean_probability': float(probabilities[selected].mean()),
                            'observed_positive_rate': float(labels[selected].mean())})
    axes[2].plot([item['mean_probability'] for item in calibration],
                 [item['observed_positive_rate'] for item in calibration], 'o-', label='10 equal-width bins')
    axes[2].plot([0, 1], [0, 1], '--', color='gray', label='Ideal calibration')
    axes[2].set(xlabel='Mean predicted probability', ylabel='Observed positive fraction', title='Reliability')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.grid(alpha=0.2)
    figure.suptitle(f'{split}: n={metrics["rows"]:,}; malicious prevalence={metrics["prevalence"]:.3f}')
    for suffix in ('png', 'svg'):
        figure.savefig(directory / f'{split}_curves.{suffix}', dpi=180)
    plt.close(figure)
    confusion = metrics['confusion']
    matrix = np.array([[confusion['tn'], confusion['fp']], [confusion['fn'], confusion['tp']]])
    figure, axis = plt.subplots(figsize=(4.5, 4), layout='constrained')
    axis.imshow(matrix, cmap='Blues')
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha='center', va='center')
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Benign', 'Malicious'],
             yticklabels=['Benign', 'Malicious'], xlabel='Predicted label', ylabel='True label', title=f'{split}: confusion counts')
    for suffix in ('png', 'svg'):
        figure.savefig(directory / f'{split}_confusion.{suffix}', dpi=180)
    plt.close(figure)
    return calibration


def prediction_outcome(label, probability, threshold):
    predicted = int(probability >= threshold)
    return ('TP' if label else 'FP') if predicted else ('FN' if label else 'TN')


def save_predictions(labels, probabilities, identities, threshold, path):
    fields = ['labeled_row_index', 'source_file', 'source_line', 'sha256', 'file_type',
              'label', 'malicious_probability', 'predicted_label', 'outcome']
    rows = [dict(identity, labeled_row_index=index, label=label, malicious_probability=probability,
                 predicted_label=int(probability >= threshold), outcome=prediction_outcome(label, probability, threshold))
            for index, (identity, label, probability) in enumerate(zip(identities, labels, probabilities))]
    for outcome in (None, 'FP', 'FN'):
        suffix = 'false_positives' if outcome == 'FP' else 'false_negatives'
        target = path if outcome is None else path.with_name(path.name.replace('predictions', suffix))
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if outcome is None or row['outcome'] == outcome)


def format_number(value):
    return 'N/A' if value is None else f'{value:.6f}'


def markdown_report(report):
    lines = ['# EMBER2024 classifier evaluation', '', '## Protocol and reproducibility', '',
             f'- Backend: **{report["backend"]}**',
             f'- Generated (UTC): {report["generated_at"]}',
             f'- Model: `{report["model"]}`',
             f'- Model SHA-256: `{report["model_sha256"]}`',
             f'- Fixed probability threshold: {report["threshold"]}',
             f'- Training metadata: {report["training_metadata_status"]}',
             f'- CPU threads: {report["protocol"]["threads"]}; device: {report["execution_device"]["resolved"]}',
             f'- Device selection: {report["execution_device"]["reason"]}',
             f'- Evaluation wall time: {report["timing_seconds"]:.3f} s', '']
    training = report.get('training')
    if training:
        lines += [f'Training rows: {training["rows"]:,}; seed: {training["seed"]}; '
                  f'boosting rounds: {training["actual_rounds"]} actual / {training["rounds"]} requested.', '',
                  f'Feature handling: {training["feature_handling"]}', '',
                  'Hyperparameters:', '', '```json', json.dumps(training['parameters'], indent=2), '```', '']
    lines += ['## Results at the fixed threshold', '',
              '| Split | Labeled n | Benign | Malicious | Skipped unlabeled | Accuracy | Precision | Recall | F1 | FPR |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, metrics in report['splits'].items():
        values = [split, str(metrics['rows']), str(metrics['labels']['0']), str(metrics['labels']['1']),
                  str(metrics['skipped_unlabeled'])]
        values += [format_number(metrics[key]) for key in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Ranking, balance, and probability quality', '',
              '| Split | ROC-AUC | Average precision | AP baseline (prevalence) | Balanced accuracy | MCC | Brier ↓ | Log loss ↓ |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, metrics in report['splits'].items():
        lines.append('| ' + ' | '.join([split] + [format_number(metrics[key]) for key in (
            'roc_auc', 'average_precision', 'prevalence', 'balanced_accuracy', 'matthews_correlation', 'brier_score', 'log_loss')]) + ' |')
    lines += ['', '## 95% Wilson confidence intervals', '',
              '| Split | Accuracy | Precision | Recall / detection rate | FPR | Specificity |',
              '| --- | --- | --- | --- | --- | --- |']
    for split, metrics in report['splits'].items():
        values = []
        for key in ('accuracy', 'precision', 'recall', 'false_positive_rate', 'specificity'):
            interval = metrics['confidence_intervals_95'][key]
            values.append('N/A' if interval is None else f'[{interval[0]:.6f}, {interval[1]:.6f}]')
        lines.append('| ' + ' | '.join([split] + values) + ' |')
    lines += ['', '## Confusion counts', '', '| Split | TN | FP | FN | TP |', '| --- | ---: | ---: | ---: | ---: |']
    for split, metrics in report['splits'].items():
        lines.append('| ' + ' | '.join([split] + [str(metrics['confusion'][key]) for key in ('tn', 'fp', 'fn', 'tp')]) + ' |')
    for split, metrics in report['splits'].items():
        lines += ['', f'## {split}: figures and data', '',
                  f'Dataset SHA-256: `{metrics["dataset_sha256"]}`', '',
                  f'![ROC, precision–recall and reliability]({split}_curves.png)', '',
                  f'![Confusion matrix]({split}_confusion.png)', '',
                  f'[Vector curves]({split}_curves.svg) · [Predictions]({split}_predictions.csv)', '']
        lines += [f'[False positives]({split}_false_positives.csv) · [False negatives]({split}_false_negatives.csv)', '']
    lines += ['## Interpretation and limitations', '', *['- ' + item for item in report['notes']], '',
              '## Software versions', '', '| Component | Version |', '| --- | --- |']
    lines += [f'| {name} | {value} |' for name, value in report['software'].items()]
    return '\n'.join(lines) + '\n'
