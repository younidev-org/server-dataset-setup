"""Train one benign/malicious LightGBM model on all nested *_train.jsonl files."""

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np
from thrember import PEFeatureExtractor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def training_records(paths):
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    label = record['label']
                    if label is None or label == -1:
                        continue
                    if type(label) is not int or label not in (0, 1):
                        raise ValueError('Expected binary label 0 or 1')
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'models' / 'ember2024')
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--threads', type=int, default=0, help='0 uses all CPU cores')
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 0:
        parser.error('--rounds must be positive; --threads must be nonnegative')
    data = args.data_dir.resolve()
    paths = sorted(path for path in data.rglob('*_train.jsonl') if path.is_file())
    if not paths:
        parser.error(f'No training JSONL files (*_train.jsonl) under {data}')
    print(f'Found {len(paths)} training files; counting labeled rows...', flush=True)
    labels = Counter(record['label'] for record in training_records(paths))
    if set(labels) != {0, 1}:
        parser.error(f'Training requires both benign (0) and malicious (1) rows: {dict(labels)}')
    rows = sum(labels.values())
    extractor = PEFeatureExtractor()
    output = args.output_dir.resolve()
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
        train_set = lgb.Dataset(features, label=targets, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        print('Training LightGBM on all selected rows...', flush=True)
        model = lgb.train({
            'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 64,
            'min_data_in_leaf': 100, 'lambda_l2': 1.0, 'is_unbalance': True,
            'seed': 0, 'num_threads': args.threads, 'verbosity': 1,
        }, train_set, num_boost_round=args.rounds)
        model.save_model(str(temporary / 'model.txt'))
        (temporary / 'model.txt').replace(output / 'model.txt')
        del features
    report = {
        'rows': rows, 'labels': dict(labels), 'features': extractor.dim,
        'files': [str(path.relative_to(data)) for path in paths],
        'rounds': args.rounds,
    }
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output / "model.txt"} and {output / "training.json"}', flush=True)


if __name__ == '__main__':
    main()
