# Server dataset and training hub

This repository distributes server setup and training scripts. Develop and publish
changes from a development machine; the server only needs read access and pulls
updates. Server data and generated artifacts stay local under ignored `data/`.

## Layout

```text
scripts/server.sh           setup, tests, download, and training commands
train.py                   EMBER2024 binary LightGBM training
download_pdf.py             public PDF training feature downloader
requirements.txt           Python dependencies
vendor/EMBER2024/           bundled upstream source and Apache 2.0 license
test_scripts.py            functional tests
data/                      ignored; never commit this directory
  datasets/PDF/            downloaded PDF *_train.jsonl files
  datasets/ELF/            optional additional ELF training features
  models/ember2024/        model.txt, training.json, temporary training vectors
  cache/                   Hugging Face, pip, uv, Python, and matplotlib caches
  venv/                    local Python environment
  tmp/                     temporary installation/runtime files
  logs/                    optional run logs
  archives/                optional large archives and backups
```

The original portable archive has been unpacked into tracked source files. Its
local backup is in `data/archives/`; fresh clones do not need that archive. No
pretrained model or dataset is included. Keep future large files inside `data/`.
Training writes models there automatically.

## Prepare or update the server

Requirements: Linux, Git, Python 3.10+, Python venv/pip support, and the OpenMP
runtime (`libgomp1` on Debian/Ubuntu). Setup uses uv if available, otherwise pip.
Public package and dataset downloads need network access; the public Hugging Face
dataset does not require a token.

For the first checkout:

```bash
git clone https://github.com/younidev-org/server-dataset-setup.git
cd server-dataset-setup
bash scripts/server.sh setup
bash scripts/server.sh test
```

For subsequent updates, from the checkout:

```bash
git pull --ff-only
bash scripts/server.sh setup
bash scripts/server.sh test
```

These commands do not push or delete local data. Avoid editing tracked files on
the server so pulls can fast-forward. Do not run `git clean -fdx`: it deletes
ignored data. Dependency installation refreshes the environment; it does not
automatically download datasets or start training. Stop an active training run
before updating scripts or dependencies. No background service is required.

## Download and train

```bash
bash scripts/server.sh download
bash scripts/server.sh train --threads 4 --rounds 500
```

Download retrieves only `PDF_train.zip` from `joyce8/EMBER2024` and extracts it
into `data/datasets/PDF/`. It contains precomputed features, not original PDFs.
Hugging Face cache is in `data/cache/huggingface/`; Xet is disabled. Rerunning
download may download and extract the archive again.

Training recursively reads `*_train.jsonl` under `data/datasets/`, skips null/-1
labels, requires both binary classes, and excludes held-out filenames. To add
ELF training features, put official `*_train.jsonl` files under
`data/datasets/ELF/`. Do not duplicate dataset files. Finish downloads before
training. Each run retrains from scratch and replaces
`data/models/ember2024/model.txt` and `training.json` after training succeeds.
It does not compute holdout accuracy.

Full training needs about 10 KB of temporary disk per labeled row plus RAM for
LightGBM; check disk and memory before large runs. Defaults are 500 boosting
rounds and all CPU cores; use `--threads` to limit CPU use. Optional overrides
are `download --data-dir PATH` and `train --data-dir PATH --output-dir PATH`;
keep overrides under `data/` to preserve this layout. Paths default relative
to the repository, regardless of the caller's working directory.

Tests use generated feature records and a local ZIP, including a real LightGBM
fit and a check that default model outputs stay under `data/`. They need no
dataset download. A passing test suite does not validate full-dataset training
or external dataset connectivity.

Vendored upstream commit: `0ef753e81d98bf209f71b03cd331dfc190b5b54d`
from https://github.com/FutureComputing4AI/EMBER2024. Signify is pinned to 0.8.1
because this source uses `SignedPEFile`, removed in Signify 0.9.
