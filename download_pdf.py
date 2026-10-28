"""Download and extract only the EMBER2024 PDF training set."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from thrember import download_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'datasets')
    args = parser.parse_args()
    destination = args.data_dir.resolve() / 'PDF'
    destination.mkdir(parents=True, exist_ok=True)
    print(f'Downloading PDF training data into {destination}', flush=True)
    download_dataset(str(destination), split='train', file_type='PDF')
    print('Download and extraction complete.', flush=True)


if __name__ == '__main__':
    main()
