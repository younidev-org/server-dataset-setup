#!/usr/bin/env bash
# Run from any working directory. This script never pushes to Git.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p data/{datasets,models/ember2024,reports,cache,tmp,logs,archives}
export HF_HOME="$repo_root/data/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="$repo_root/data/cache/pip"
export UV_CACHE_DIR="$repo_root/data/cache/uv"
export MPLCONFIGDIR="$repo_root/data/cache/matplotlib"
export XDG_CACHE_HOME="$repo_root/data/cache"
export PYTHONPYCACHEPREFIX="$repo_root/data/cache/pycache"
export TMPDIR="$repo_root/data/tmp"
python_bin="$repo_root/data/venv/bin/python"
action="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$action" in
  setup)
    if [ ! -x "$python_bin" ]; then
      python3 -m venv "$repo_root/data/venv"
    fi
    setup_fingerprint="$("$python_bin" setup_state.py fingerprint)"
    if [ -f data/setup.sha256 ] && [ "$(cat data/setup.sha256)" = "$setup_fingerprint" ] && "$python_bin" setup_state.py check; then
      echo 'Dependencies already prepared and healthy; skipping installation.'
      exit 0
    fi
    if "$python_bin" -c "from importlib.metadata import distribution; distribution('xgboost-cpu')" >/dev/null 2>&1; then
      echo 'Existing xgboost-cpu preserved. Full CUDA XGBoost cannot be installed alongside it; use a separate environment for CUDA.' >&2
      exit 1
    fi
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$python_bin" -r requirements.txt
      uv pip check --python "$python_bin"
    else
      "$python_bin" -m pip install --disable-pip-version-check -r requirements.txt
      "$python_bin" -m pip check
    fi
    "$python_bin" setup_state.py check
    printf '%s\n' "$setup_fingerprint" > data/setup.sha256
    ;;
  train|download|evaluate|compare|benchmark|gpu|test)
    if [ ! -x "$python_bin" ]; then
      echo 'Run bash scripts/server.sh setup first.' >&2
      exit 1
    fi
    case "$action" in
      train) exec "$python_bin" train.py "$@" ;;
      download) exec "$python_bin" download_pdf.py "$@" ;;
      evaluate) exec "$python_bin" evaluate.py "$@" ;;
      benchmark) exec "$python_bin" benchmark.py "$@" ;;
      gpu) exec "$python_bin" gpu_info.py "$@" ;;
      compare) exec "$python_bin" compare.py "$@" ;;
      test) exec "$python_bin" -m unittest discover -v "$@" ;;
    esac
    ;;
  help|-h|--help)
    echo 'Usage: bash scripts/server.sh {setup|test|download|train|evaluate|compare|benchmark|gpu} [options]'
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
