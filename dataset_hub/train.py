"""Train a binary LightGBM or XGBoost model on nested *_train.jsonl features."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import inspect
from pathlib import Path
import tempfile
import time
import sys

from .experiment import input_manifest, manifest_sha256, sha256_file, software_versions, resolve_xgboost_device
from .configuration import command_config

import lightgbm as lgb
import numpy as np
from thrember import PEFeatureExtractor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def training_records(paths, statistics=None, with_source=False):
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    label = record['label']
                    if statistics is not None:
                        statistics['total_records'] = statistics.get('total_records', 0) + 1
                    if label is None or label == -1:
                        if statistics is not None:
                            statistics['skipped_unlabeled'] = statistics.get('skipped_unlabeled', 0) + 1
                        continue
                    if type(label) is not int or label not in (0, 1):
                        raise ValueError('Expected binary label 0 or 1')
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                yield (record, path, line_number) if with_source else record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = command_config(parser)
    parser.add_argument('--backend', choices=['lightgbm', 'xgboost'], default='lightgbm')
    parser.add_argument('--data-dir', type=Path, default=ROOT / config['paths']['data_dir'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=config['training']['seed'])
    parser.add_argument('--rounds', type=int, default=config['training']['rounds'])
    parser.add_argument('--threads', type=int, default=config['runtime']['threads'], help='CPU threads (TOML default: 10; 0 uses all cores)')
    parser.add_argument('--device', default=None, help='XGBoost: auto, cpu, cuda, or cuda:N')
    args = parser.parse_args()
    if args.device is None:
        args.device = config['runtime']['xgboost_device'] if args.backend == 'xgboost' else 'cpu'
    started = time.perf_counter()
    if args.rounds < 1 or args.threads < 0:
        parser.error('--rounds must be positive; --threads must be nonnegative')
    if args.backend == 'lightgbm' and args.device not in ('auto', 'cpu'):
        parser.error('--device CUDA applies to XGBoost only; LightGBM uses CPU')
    device_info = (resolve_xgboost_device(args.device, args.threads) if args.backend == 'xgboost'
                   else {'requested': args.device, 'resolved': 'cpu', 'reason': 'LightGBM CPU backend'})
    print(f'Using {args.threads} CPU threads; device: {device_info["resolved"]} ({device_info["reason"]})', flush=True)
    data = args.data_dir.resolve()
    paths = sorted(path for path in data.rglob('*_train.jsonl') if path.is_file())
    if not paths:
        parser.error(f'No training JSONL files (*_train.jsonl) under {data}')
    print(f'Found {len(paths)} training files; counting labeled rows...', flush=True)
    manifest = input_manifest(paths, data)
    statistics = {}
    labels = Counter(record['label'] for record in training_records(paths, statistics))
    if set(labels) != {0, 1}:
        parser.error(f'Training requires both benign (0) and malicious (1) rows: {dict(labels)}')
    rows = sum(labels.values())
    extractor = PEFeatureExtractor()
    output = (args.output_dir or ROOT / config['paths']['model_root'] / (
        'ember2024' if args.backend == 'lightgbm' else 'ember2024-xgboost')).resolve()
    model_name = 'model.txt' if args.backend == 'lightgbm' else 'model.ubj'
    output.mkdir(parents=True, exist_ok=True)
    print(f'Vectorizing {rows:,} rows into {extractor.dim} features.', flush=True)
    # Disk-backed vectors avoid keeping a Python list of every feature vector.
    with tempfile.TemporaryDirectory(prefix='vectors-', dir=output) as temporary:
        temporary = Path(temporary)
        features = np.memmap(temporary / 'X.dat', dtype='float32', mode='w+', shape=(rows, extractor.dim))
        targets = np.empty(rows, dtype=np.uint8)
        count = 0
        for index, record in enumerate(tqdm(training_records(paths), total=rows, desc='Vectorizing')):
            if index >= rows:
                raise ValueError('Data changed during vectorization; rerun after downloads finish')
            features[index] = extractor.process_raw_features(record)
            targets[index] = record['label']
            count += 1
        if count != rows:
            raise ValueError('Data changed during vectorization; rerun after downloads finish')
        features.flush()
        print(f'Training {args.backend} on all selected rows...', flush=True)
        fit_started = time.perf_counter()
        if args.backend == 'lightgbm':
            categorical = [2, 3, 4, 5, 6, 701, 702]
            parameters = {
                'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 64,
                'min_data_in_leaf': 100, 'lambda_l2': 1.0, 'is_unbalance': True,
                'seed': args.seed, 'num_threads': args.threads, 'verbosity': 1,
            }
            train_set = lgb.Dataset(features, label=targets, categorical_feature=categorical)
            model = lgb.train(parameters, train_set, num_boost_round=args.rounds)
            actual_rounds = model.current_iteration()
        else:
            import xgboost as xgb
            categorical = []
            parameters = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss',
                'tree_method': 'hist', 'device': device_info['resolved'], 'eta': 0.1, 'max_depth': 6,
                'min_child_weight': 1, 'lambda': 1.0,
                'scale_pos_weight': labels[0] / labels[1],
                'seed': args.seed, 'nthread': args.threads,
            }
            train_set = xgb.DMatrix(features, label=targets, nthread=args.threads)
            model = xgb.train(parameters, train_set, num_boost_round=args.rounds)
            actual_rounds = model.num_boosted_rounds()
        fit_seconds = time.perf_counter() - fit_started
        if input_manifest(paths, data) != manifest:
            raise ValueError('Training input files changed during this run; model not published')
        model.save_model(str(temporary / model_name))
        (temporary / model_name).replace(output / model_name)
        del features
    report = {
        'rows': rows, 'labels': dict(labels), 'features': extractor.dim,
        'files': [str(path.relative_to(data)) for path in paths],
        'rounds': args.rounds,
        'actual_rounds': actual_rounds, 'backend': args.backend, 'seed': args.seed,
        'execution_device': device_info,
        'parameters': parameters, 'categorical_feature_indices': categorical,
        'feature_handling': 'Bundled 2568-dimensional float32 EMBER vectors; XGBoost treats all columns as numeric, LightGBM declares selected categorical indices.',
        'input_manifest': manifest, 'dataset_sha256': manifest_sha256(manifest),
        'model_sha256': sha256_file(output / model_name),
        'feature_extractor_sha256': sha256_file(Path(inspect.getfile(PEFeatureExtractor))),
        'trainer_sha256': sha256_file(Path(__file__)),
        'software': software_versions(), 'command': sys.argv,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'timing_seconds': {'fit': fit_seconds, 'total': time.perf_counter() - started},
        **statistics,
    }
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output / model_name} and {output / "training.json"}', flush=True)


if __name__ == '__main__':
    main()
