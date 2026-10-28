"""Execute the enabled command arrays in TOML, sequentially and fail-fast."""

import argparse
from datetime import datetime, timezone
import json
import uuid
from pathlib import Path
import shlex
import subprocess
import sys

from configuration import load_config

ROOT = Path(__file__).resolve().parent


def template_values(settings, root, run_dir=None):
    root = Path(root).resolve()
    paths = {name: str((root / value).resolve()) for name, value in settings['paths'].items()}
    if run_dir is not None:
        paths['report_root'] = str(Path(run_dir) / 'reports')
    values = dict(paths, **settings['runtime'], **settings['training'], **settings['evaluation'])
    values.update(run_dir=str(run_dir or ''), run_id=Path(run_dir).name if run_dir else '', root=str(root), server=str(root / 'scripts/server.sh'),
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


def build_commands(settings, root=ROOT, run_dir=None):
    values = template_values(settings, root, run_dir)
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
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        run_dir = (ROOT / settings['paths']['run_root'] / run_id).resolve()
        commands = build_commands(settings, run_dir=run_dir)
        names = [step['name'] for step in settings['steps'] if step.get('enabled', True)]
        if args.dry_run:
            for name, command in zip(names, commands):
                print(f'{name}\n  {shlex.join(command)}')
            return 0
        if not commands:
            print('No enabled steps. Nothing to run.', flush=True)
            return 0
        run_dir.mkdir(parents=True, exist_ok=False)
        record = {'run_id': run_id, 'status': 'running', 'settings': settings,
                  'resolved_paths': template_values(settings, ROOT, run_dir), 'commands': commands}
        def save_record():
            pending = run_dir / 'run.json.tmp'
            pending.write_text(json.dumps(record, indent=2) + '\n')
            pending.replace(run_dir / 'run.json')
        save_record()
        print(f'Run directory: {run_dir}', flush=True)
        try:
            run_commands(commands, names=names)
        except BaseException as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            save_record()
            raise
        record['status'] = 'completed'
        save_record()
        print(f'Reports: {run_dir / "reports"}', flush=True)
        print('All configured steps completed.', flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f'Step failed with exit code {exc.returncode}; remaining steps were not run.', file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
