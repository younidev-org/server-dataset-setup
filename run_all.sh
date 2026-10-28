#!/usr/bin/env bash
# Commands (including setup) come from TOML. The launcher needs only a TOML parser.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
executor_python=python3
if [ -x "$repo_root/data/venv/bin/python" ]; then
  executor_python="$repo_root/data/venv/bin/python"
fi
if ! "$executor_python" -c 'import tomllib' >/dev/null 2>&1 && ! "$executor_python" -c 'import tomli' >/dev/null 2>&1; then
  # Python 3.10 compatibility: bootstrap only the TOML parser, outside tracked files.
  mkdir -p data/cache/pip data/tmp
  if [ ! -x data/bootstrap/bin/python ]; then python3 -m venv data/bootstrap; fi
  executor_python="$repo_root/data/bootstrap/bin/python"
  PIP_CACHE_DIR="$repo_root/data/cache/pip" TMPDIR="$repo_root/data/tmp" \
    "$executor_python" -m pip install --disable-pip-version-check 'tomli>=2,<3'
fi
exec "$executor_python" "$repo_root/pipeline.py" "$@"
