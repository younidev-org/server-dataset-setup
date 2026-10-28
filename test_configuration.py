import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class ConfigurationTest(unittest.TestCase):
    def test_local_override_preserves_shared_defaults(self):
        from configuration import load_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            (root / 'config.toml').write_text('[runtime]\nthreads = 10\n[training]\nrounds = 500\n')
            (root / 'data/config.toml').write_text('[runtime]\nthreads = 3\n')
            settings = load_config(root=root)
            self.assertEqual(settings['runtime']['threads'], 3)
            self.assertEqual(settings['training']['rounds'], 500)

    def test_explicit_config_overrides_local_and_rejects_invalid_values(self):
        from configuration import load_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            (root / 'data/config.toml').write_text('[runtime]\nthreads = 3\n')
            explicit = root / 'experiment.toml'
            explicit.write_text('[runtime]\nthreads = 7\n')
            self.assertEqual(load_config(explicit, root=root)['runtime']['threads'], 7)
            for text in ('[runtime]\nthreads = -1\n', '[runtime]\nthreads = true\n',
                         '[runtime]\nthreadz = 10\n', '[evaluation]\nthreshold = 2.0\n'):
                explicit.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(explicit, root=root)

    def test_auto_threads_resolves_to_a_positive_worker_count(self):
        from configuration import load_config
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'auto.toml'
            path.write_text('[runtime]\nthreads = "auto"\n')
            self.assertGreaterEqual(load_config(path, root=Path(folder))['runtime']['threads'], 1)

    def test_pipeline_runs_required_actions_in_order_with_shared_settings(self):
        from configuration import load_config
        from pipeline import build_commands
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(ROOT / 'config.toml', root / 'config.toml')
            config = root / 'run.toml'
            config.write_text('[runtime]\nthreads = 10\nxgboost_device = "cpu"\n')
            commands = build_commands(load_config(config, root=root), root)
            self.assertEqual([command[2] for command in commands],
                             ['setup', 'test', 'download', 'train', 'evaluate', 'train', 'evaluate', 'compare'])
            for command in commands:
                if command[2] in ('train', 'evaluate'):
                    self.assertEqual(command[command.index('--threads') + 1], '10')
                if '--backend' in command and command[command.index('--backend') + 1] == 'xgboost':
                    self.assertEqual(command[command.index('--device') + 1], 'cpu')

    def test_pipeline_stops_at_failure_and_preserves_exit_status(self):
        from pipeline import run_commands
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / 'steps.txt'
            script = root / 'step.py'
            script.write_text('import pathlib, sys\np = pathlib.Path(sys.argv[1])\n'
                              'with p.open("a") as f: f.write(sys.argv[2] + "\\n")\n'
                              'sys.exit(int(sys.argv[3]))\n')
            commands = [[sys.executable, str(script), str(marker), name, str(code)]
                        for name, code in [('first', 0), ('failed', 7), ('must-not-run', 0)]]
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                run_commands(commands, root)
            self.assertEqual(failure.exception.returncode, 7)
            self.assertEqual(marker.read_text().splitlines(), ['first', 'failed'])

    def test_bash_executor_obeys_custom_toml_steps_and_dry_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / 'result with spaces.txt'
            script = root / 'record.py'
            script.write_text('import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.argv[2])\n')
            config = root / 'custom.toml'
            arguments = [sys.executable, str(script), str(marker), 'literal ; $(not-a-shell)']
            config.write_text('[[steps]]\nname = "Custom command"\ncommand = ' + json.dumps(arguments) + '\n'
                              '[[steps]]\nname = "Disabled"\nenabled = false\ncommand = ["must-not-execute"]\n')
            preview = subprocess.run(['bash', str(ROOT / 'run_all.sh'), '--config', str(config), '--dry-run'],
                                     capture_output=True, text=True, cwd='/tmp')
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse(marker.exists())
            executed = subprocess.run(['bash', str(ROOT / 'run_all.sh'), '--config', str(config)],
                                      capture_output=True, text=True, cwd='/tmp')
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertEqual(marker.read_text(), 'literal ; $(not-a-shell)')


if __name__ == '__main__':
    unittest.main()
