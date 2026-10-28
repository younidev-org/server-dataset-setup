"""Reproducibility metadata shared by training and evaluation."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(paths, root):
    return [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
             'sha256': sha256_file(path)} for path in paths]


def manifest_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def software_versions():
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost-cpu', 'matplotlib', 'thrember'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
