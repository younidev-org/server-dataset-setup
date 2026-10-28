"""Reproducibility metadata shared by training and evaluation."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import re
import warnings


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
    for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost', 'matplotlib', 'thrember'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def resolve_xgboost_device(requested, threads):
    """Probe CUDA through XGBoost itself, detecting its automatic CPU fallback."""
    import numpy as np
    import xgboost as xgb
    if requested not in ('auto', 'cpu', 'cuda') and not re.fullmatch(r'cuda:\d+', requested):
        raise ValueError('device must be auto, cpu, cuda, or cuda:N')
    cuda_build = bool(xgb.build_info().get('USE_CUDA'))
    info = {'requested': requested, 'resolved': 'cpu', 'cuda_build': cuda_build}
    if requested == 'cpu':
        return dict(info, reason='CPU explicitly selected')
    if not cuda_build:
        if requested != 'auto':
            raise RuntimeError('CUDA requested, but this XGBoost build has no CUDA support; rerun setup')
        return dict(info, reason='XGBoost build has no CUDA support')
    target = 'cuda' if requested == 'auto' else requested
    try:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            probe = xgb.train({'tree_method': 'hist', 'device': target, 'nthread': threads,
                               'objective': 'binary:logistic'},
                              xgb.DMatrix(np.array([[0.0], [1.0]], dtype=np.float32),
                                          label=[0, 1], nthread=threads), num_boost_round=1)
        resolved = json.loads(probe.save_config())['learner']['generic_param']['device']
        if resolved.startswith('cuda'):
            if requested.startswith('cuda:') and resolved != f'cuda:{int(requested.split(":")[1])}':
                raise RuntimeError(f'CUDA device {requested} unavailable; XGBoost selected {resolved}')
            return dict(info, resolved=resolved, reason='CUDA training probe succeeded')
        reason = '; '.join(str(item.message) for item in messages) or 'XGBoost selected CPU during CUDA probe'
    except xgb.core.XGBoostError as exc:
        reason = str(exc).splitlines()[0]
    if requested != 'auto':
        raise RuntimeError(f'CUDA requested but unavailable: {reason}')
    return dict(info, reason=reason)
