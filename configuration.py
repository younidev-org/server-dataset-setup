"""Shared TOML settings. Explicit command-line flags override these defaults."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    'runtime': {'threads': 10, 'xgboost_device': 'auto'},
    'training': {'rounds': 500, 'seed': 0},
    'evaluation': {'threshold': 0.5, 'batch_size': 1024},
    'paths': {'data_dir': 'data/datasets', 'model_root': 'data/models', 'report_root': 'data/reports', 'run_root': 'data/runs'},
    'steps': [],
}


def available_threads():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def load_config(path=None, root=ROOT):
    root = Path(root)
    settings = deepcopy(DEFAULTS)
    base = root / 'config.toml'
    files = [base] if base.is_file() else []
    override = Path(path) if path is not None else root / 'data/config.toml'
    if path is not None and not override.is_file():
        raise ValueError(f'Configuration file does not exist: {override}')
    if override.is_file() and override.resolve() != base.resolve():
        files.append(override)
    if files:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        for file in files:
            with file.open('rb') as stream:
                document = tomllib.load(stream)
            for section, values in document.items():
                if section not in DEFAULTS:
                    raise ValueError(f'Unknown configuration section: {section}')
                if section == 'steps':
                    settings['steps'] = values
                else:
                    if not isinstance(values, dict) or set(values) - set(DEFAULTS[section]):
                        raise ValueError(f'Unknown or invalid settings in [{section}]')
                    settings[section].update(values)
    threads = settings['runtime']['threads']
    if threads == 'auto':
        threads = available_threads()
    if type(threads) is not int or threads < 1:
        raise ValueError('runtime.threads must be a positive integer or "auto"')
    settings['runtime']['threads'] = threads
    device = settings['runtime']['xgboost_device']
    if not isinstance(device, str) or not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', device):
        raise ValueError('runtime.xgboost_device must be auto, cpu, cuda, or cuda:N')
    for key, minimum in [('rounds', 1), ('seed', 0)]:
        value = settings['training'][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f'training.{key} must be an integer >= {minimum}')
    evaluation = settings['evaluation']
    if type(evaluation['batch_size']) is not int or evaluation['batch_size'] < 1:
        raise ValueError('evaluation.batch_size must be a positive integer')
    if type(evaluation['threshold']) not in (int, float) or not 0 <= evaluation['threshold'] <= 1:
        raise ValueError('evaluation.threshold must be between 0 and 1')
    if any(not isinstance(value, str) or not value.strip() for value in settings['paths'].values()):
        raise ValueError('paths settings must be nonempty strings')
    if not isinstance(settings['steps'], list):
        raise ValueError('steps must use TOML [[steps]] tables')
    for step in settings['steps']:
        if not isinstance(step, dict) or set(step) - {'name', 'command', 'enabled'}:
            raise ValueError('Each step accepts only name, command, and enabled')
        if not isinstance(step.get('name'), str) or not step['name'].strip():
            raise ValueError('Each step needs a name')
        if type(step.get('enabled', True)) is not bool:
            raise ValueError('step.enabled must be true or false')
        command = step.get('command')
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command) or not command[0]:
            raise ValueError('step.command must be a nonempty array of string arguments')
    return settings


def command_config(parser):
    """Read --config before constructing defaults; argparse still handles CLI overrides."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selected, _ = selector.parse_known_args()
    parser.add_argument('--config', type=Path, help='TOML overrides (default local override: data/config.toml)')
    return load_config(selected.config)
