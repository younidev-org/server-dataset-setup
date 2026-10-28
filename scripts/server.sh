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
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$python_bin" -r requirements.txt
      uv pip check --python "$python_bin"
    else
      "$python_bin" -m pip install --disable-pip-version-check -r requirements.txt
      "$python_bin" -m pip check
    fi
    ;;
  train|download|evaluate|test)
    if [ ! -x "$python_bin" ]; then
      echo 'Run bash scripts/server.sh setup first.' >&2
      exit 1
    fi
    case "$action" in
      train) exec "$python_bin" train.py "$@" ;;
      download) exec "$python_bin" download_pdf.py "$@" ;;
      evaluate) exec "$python_bin" evaluate.py "$@" ;;
      test) exec "$python_bin" -m unittest discover -v "$@" ;;
    esac
    ;;
  help|-h|--help)
    echo 'Usage: bash scripts/server.sh {setup|test|download|train|evaluate} [options]'
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
