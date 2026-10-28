"""Execute the enabled command arrays in TOML, sequentially and fail-fast."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from configuration import load_config

ROOT = Path(__file__).resolve().parent


def template_values(settings, root):
    root = Path(root).resolve()
    paths = {name: str((root / value).resolve()) for name, value in settings['paths'].items()}
    values = dict(paths, **settings['runtime'], **settings['training'], **settings['evaluation'])
    values.update(root=str(root), server=str(root / 'scripts/server.sh'),
                  python=str(root / 'data/venv/bin/python'))
    values.update(lightgbm_model_dir=str(Path(paths['model_root']) / 'ember2024'),
                  xgboost_model_dir=str(Path(paths['model_root']) / 'ember2024-xgboost'),
                  xgboost_report_dir=str(Path(paths['report_root']) / 'xgboost'),
                  comparison_dir=str(Path(paths['report_root']) / 'comparison'))
    values.update(lightgbm_model=str(Path(values['lightgbm_model_dir']) / 'model.txt'),
                  xgboost_model=str(Path(values['xgboost_model_dir']) / 'model.ubj'),
                  lightgbm_report=str(Path(paths['report_root']) / 'evaluation.json'),
                  xgboost_report=str(Path(values['xgboost_report_dir']) / 'evaluation.json'))
    return {name: str(value) for name, value in values.items()}


def build_commands(settings, root=ROOT):
    values = template_values(settings, root)
    commands = []
    for step in settings['steps']:
        if step.get('enabled', True):
            try:
                commands.append([arg.format_map(values) for arg in step['command']])
            except (KeyError, ValueError) as exc:
                raise ValueError(f'Invalid command placeholder in step {step["name"]}: {exc}') from exc
    return commands


def run_commands(commands, root=ROOT, names=None):
    for index, command in enumerate(commands):
        name = names[index] if names else command[0]
        print(f'[{index + 1}/{len(commands)}] {name}\n  {shlex.join(command)}', flush=True)
        subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='TOML overrides; otherwise use data/config.toml when present')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing any step')
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        commands = build_commands(settings)
        names = [step['name'] for step in settings['steps'] if step.get('enabled', True)]
        if args.dry_run:
            for name, command in zip(names, commands):
                print(f'{name}\n  {shlex.join(command)}')
            return 0
        if not commands:
            print('No enabled steps. Nothing to run.', flush=True)
            return 0
        run_commands(commands, names=names)
        print('All configured steps completed.', flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f'Step failed with exit code {exc.returncode}; remaining steps were not run.', file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
