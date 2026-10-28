"""Evaluate LightGBM or XGBoost on held-out data with reproducible scientific reports."""

import argparse
from datetime import datetime, timezone
import json
import inspect
from pathlib import Path
import tempfile
import time

import lightgbm as lgb
import numpy as np
from thrember import PEFeatureExtractor

from experiment import input_manifest, manifest_sha256, sha256_file, software_versions, resolve_xgboost_device
from reporting import classification_metrics, markdown_report, save_figures, save_predictions
from train import training_records
from configuration import command_config

ROOT = Path(__file__).resolve().parent


def evaluate_split(model, extractor, paths, batch_size, threads, threshold, backend='lightgbm', data_root=None):
    labels, scores, batch, identities = [], [], [], []
    statistics = {'total_records': 0, 'skipped_unlabeled': 0}

    def predict_batch():
        features = np.asarray(batch, dtype=np.float32)
        if backend == 'lightgbm':
            predictions = np.asarray(model.predict(features, num_threads=threads))
        else:
            import xgboost as xgb
            # DMatrix prediction supports transfer from host arrays to the chosen GPU.
            predictions = np.asarray(model.predict(xgb.DMatrix(features, nthread=threads)))
        if predictions.ndim != 1 or len(predictions) != len(batch) or not np.all(np.isfinite(predictions)) or np.any((predictions < 0) | (predictions > 1)):
            raise ValueError('Expected one finite binary probability between 0 and 1 per record')
        scores.extend(predictions.tolist())
        batch.clear()

    for record, path, line_number in training_records(paths, statistics, with_source=True):
        labels.append(record['label'])
        identities.append({'source_file': str(path.relative_to(data_root)) if data_root else str(path),
                           'source_line': line_number, 'sha256': record.get('sha256') or '',
                           'file_type': record.get('file_type') or ''})
        batch.append(extractor.process_raw_features(record))
        if len(batch) == batch_size:
            predict_batch()
    if batch:
        predict_batch()
    if not labels:
        raise ValueError(f'No labeled evaluation records in {paths}')
    metrics = classification_metrics(labels, scores, threshold)
    metrics.update(statistics)
    return metrics, labels, scores, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = command_config(parser)
    parser.add_argument('--backend', choices=['lightgbm', 'xgboost'], default='lightgbm')
    parser.add_argument('--data-dir', type=Path, default=ROOT / config['paths']['data_dir'])
    parser.add_argument('--model', type=Path)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--threshold', type=float, default=config['evaluation']['threshold'])
    parser.add_argument('--batch-size', type=int, default=config['evaluation']['batch_size'])
    parser.add_argument('--threads', type=int, default=config['runtime']['threads'])
    parser.add_argument('--device', default=None, help='XGBoost: auto, cpu, cuda, or cuda:N')
    args = parser.parse_args()
    if args.device is None:
        args.device = config['runtime']['xgboost_device'] if args.backend == 'xgboost' else 'cpu'
    started = time.perf_counter()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1 or args.threads < 1:
        parser.error('threshold must be between 0 and 1; batch-size and threads must be positive')
    if args.backend == 'lightgbm' and args.device not in ('auto', 'cpu'):
        parser.error('--device CUDA applies to XGBoost only; LightGBM uses CPU')
    device_info = (resolve_xgboost_device(args.device, args.threads) if args.backend == 'xgboost'
                   else {'requested': args.device, 'resolved': 'cpu', 'reason': 'LightGBM CPU backend'})
    print(f'Using {args.threads} CPU threads; device: {device_info["resolved"]} ({device_info["reason"]})', flush=True)
    if args.model is None:
        args.model = ROOT / config['paths']['model_root'] / ('ember2024/model.txt' if args.backend == 'lightgbm' else 'ember2024-xgboost/model.ubj')
    if args.report_dir is None:
        # Preserve existing LightGBM report paths; keep XGBoost results separate.
        args.report_dir = ROOT / config['paths']['report_root'] / ('' if args.backend == 'lightgbm' else 'xgboost')
    data = args.data_dir.resolve()
    split_paths = {
        'test': sorted(path for path in data.rglob('*_test.jsonl') if path.is_file() and 'challenge' not in path.name.lower()),
        'challenge': sorted(path for path in data.rglob('*.jsonl') if path.is_file() and 'challenge' in path.name.lower() and not path.name.endswith('_train.jsonl')),
    }
    for split, paths in split_paths.items():
        if not paths:
            parser.error(f'No {split} JSONL files under {data}; run the download command first')
    if args.backend == 'lightgbm':
        model = lgb.Booster(model_file=str(args.model))
        num_features = model.num_feature()
    else:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(args.model)
        model.set_param({'nthread': args.threads, 'device': device_info['resolved']})
        num_features = model.num_features()
    extractor = PEFeatureExtractor()
    if num_features != extractor.dim:
        parser.error('Model feature count does not match the bundled feature extractor')
    model_hash = sha256_file(args.model)
    extractor_hash = sha256_file(Path(inspect.getfile(PEFeatureExtractor)))
    training, training_status = None, 'legacy_or_missing_unverified'
    metadata_path = args.model.parent / 'training.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('model_sha256'):
            if metadata['model_sha256'] != model_hash or metadata.get('backend') != args.backend:
                parser.error('training.json does not match this model hash/backend; use the matching model directory')
            if metadata.get('feature_extractor_sha256') != extractor_hash:
                parser.error('Installed feature extractor differs from the one used for training')
            training, training_status = metadata, 'verified_model_hash'
    report = {
        'schema_version': 3, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend, 'model': str(args.model.resolve()), 'model_sha256': model_hash,
        'threshold': args.threshold, 'feature_count': extractor.dim,
        'feature_extractor_sha256': extractor_hash,
        'evaluator_sha256': sha256_file(Path(__file__)),
        'execution_device': device_info,
        'software': software_versions(), 'training': training, 'training_metadata_status': training_status,
        'protocol': {'threshold_selection': 'Fixed before evaluation; no held-out tuning',
                     'confidence_intervals': 'Two-sided 95% Wilson score intervals for binomial rates',
                     'prediction_batch_size': args.batch_size, 'threads': args.threads},
        'notes': [
            'Only training files are used for fitting. Test and challenge are evaluated separately; no held-out threshold tuning or early stopping is performed.',
            'Test discovery includes all *_test.jsonl files under the selected data directory; it is PDF-only only when those files contain PDF records. Input manifests identify the actual corpus.',
            'Challenge is the shared upstream set, not a PDF-only benchmark. It is an out-of-distribution stress test for a PDF-trained model.',
            'Challenge precision/accuracy reflect that selected challenge population and must not be interpreted as deployment precision or population-wide accuracy.',
            'Rows labeled null or -1 are excluded and counted. Other invalid labels fail evaluation.',
            'Null/N/A denotes an undefined metric, including ROC-AUC and FPR on an all-malicious challenge set.',
            'Wilson intervals assume independent Bernoulli trials. Duplicate samples, shared malware families, and temporal correlations can make these intervals too narrow; they do not establish future deployment performance.',
            'ROC-AUC, average precision, F1, MCC, Brier score and log loss are point estimates; no uncertainty interval or significance test is computed for them.',
            'Average precision is a step-weighted summary, not trapezoidal PR area; its random-ranking baseline is malicious prevalence.',
            'Class weighting can distort probability calibration. Brier score, log loss and reliability plots assess raw, uncalibrated probabilities; no calibration is fitted on held-out data.',
            'These are fixed-hyperparameter, single-seed baselines, not an optimized or multi-seed benchmark. LightGBM and XGBoost have different capacity and feature treatment; identical boosting-round counts do not imply equal models or compute budgets.',
        ],
        'splits': {},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # Prepare the entire report before replacing outputs; publish evaluation.json last.
    with tempfile.TemporaryDirectory(prefix='report-', dir=args.report_dir) as temporary:
        staging = Path(temporary)
        for split, paths in split_paths.items():
            print(f'Evaluating {split}: {len(paths)} files...', flush=True)
            manifest = input_manifest(paths, data)
            split_started = time.perf_counter()
            metrics, labels, scores, identities = evaluate_split(model, extractor, paths, args.batch_size, args.threads, args.threshold, args.backend, data)
            metrics['evaluation_seconds'] = time.perf_counter() - split_started
            if input_manifest(paths, data) != manifest:
                raise ValueError('Evaluation inputs changed during this run; report not published')
            metrics['input_manifest'] = manifest
            metrics['dataset_sha256'] = manifest_sha256(manifest)
            metrics['files'] = [item['path'] for item in manifest]
            metrics['calibration_bins'] = save_figures(labels, scores, metrics, split, staging)
            prediction_path = staging / f'{split}_predictions.csv'
            save_predictions(labels, scores, identities, args.threshold, prediction_path)
            metrics['prediction_csv_sha256'] = sha256_file(prediction_path)
            report['splits'][split] = metrics
            print(f'{split}: n={metrics["rows"]}; ROC-AUC={metrics["roc_auc"]}; recall={metrics["recall"]}', flush=True)
        if sha256_file(args.model) != model_hash:
            raise ValueError('Model file changed during evaluation; report not published')
        report['timing_seconds'] = time.perf_counter() - started
        (staging / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (staging / 'evaluation.md').write_text(markdown_report(report))
        for path in staging.iterdir():
            if path.name != 'evaluation.json':
                path.replace(args.report_dir / path.name)
        (staging / 'evaluation.json').replace(args.report_dir / 'evaluation.json')
    print(f'Reports saved in {args.report_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
