import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import lightgbm as lgb
from thrember import PEFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent


class TrainingTest(unittest.TestCase):
    def test_default_paths_keep_training_artifacts_under_data(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            shutil.copytree(ROOT / 'dataset_hub', folder / 'dataset_hub', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT / 'train.py', folder / 'train.py')
            shutil.copy2(ROOT / 'experiment.py', folder / 'experiment.py')
            shutil.copy2(ROOT / 'configuration.py', folder / 'configuration.py')
            (folder / 'config.toml').write_text('[runtime]\nthreads = 7\n[training]\nrounds = 9\n')
            dataset = folder / 'data/datasets/PDF/sample_train.jsonl'
            dataset.parent.mkdir(parents=True)
            extractor = PEFeatureExtractor()
            with dataset.open('w') as stream:
                for i in range(40):
                    record = extractor.raw_features(b'PDF' + bytes([i]) * 512)
                    record['label'] = i % 2
                    stream.write(json.dumps(record) + '\n')
            result = subprocess.run([
                sys.executable, str(folder / 'train.py'), '--rounds', '2', '--threads', '1',
            ], capture_output=True, text=True, cwd='/tmp')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model_path = folder / 'data/models/ember2024/model.txt'
            self.assertTrue(model_path.is_file(), result.stdout)
            model = lgb.Booster(model_file=str(model_path))
            self.assertEqual(model.num_feature(), 2568)
            report = json.loads((model_path.parent / 'training.json').read_text())
            self.assertEqual(report['rows'], 40)
            self.assertEqual(report['parameters']['num_threads'], 1)
            self.assertEqual(report['rounds'], 2)
            self.assertFalse((folder / 'output').exists())

    def test_downloader_extracts_all_splits_and_reports_sources(self):
        import download_pdf
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archives = {}
            for name, member in [('PDF_train.zip', 'sample_PDF_train.jsonl'),
                                 ('PDF_test.zip', 'sample_PDF_test.jsonl'),
                                 ('ELF_train.zip', 'sample_ELF_train.jsonl'),
                                 ('ELF_test.zip', 'sample_ELF_test.jsonl'),
                                 ('challenge.zip', 'sample_challenge_malicious.jsonl')]:
                archive = folder / name
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr(member, '{"label": 0}\n')
                archives[name] = str(archive)
            previous = Path.cwd()
            try:
                with patch('sys.argv', ['download_pdf.py', '--data-dir', str(folder / 'data'),
                                       '--report-dir', str(folder / 'reports'), '--file-types', 'PDF', 'ELF']), patch(
                    'download_pdf.hf_hub_download', side_effect=lambda **kw: archives[kw['filename']], create=True
                ):
                    download_pdf.main()
            finally:
                os.chdir(previous)
            self.assertEqual(
                (folder / 'data/PDF/sample_PDF_train.jsonl').read_text(), '{"label": 0}\n'
            )
            self.assertTrue((folder / 'data/PDF/sample_PDF_test.jsonl').is_file())
            self.assertTrue((folder / 'data/challenge/sample_challenge_malicious.jsonl').is_file())
            report = json.loads((folder / 'reports/download.json').read_text())
            self.assertEqual([item['archive'] for item in report['downloads']],
                             ['PDF_train.zip', 'PDF_test.zip', 'ELF_train.zip', 'ELF_test.zip', 'challenge.zip'])
            self.assertTrue(all(Path(path).is_file() for path in archives.values()))
            # Repeat without deleting cached archives or creating duplicate training files.
            with patch('sys.argv', ['download_pdf.py', '--data-dir', str(folder / 'data'),
                                   '--report-dir', str(folder / 'reports'), '--file-types', 'PDF', 'ELF']), patch(
                'download_pdf.hf_hub_download', side_effect=AssertionError('Completed data must not contact the network')
            ):
                download_pdf.main()
            self.assertEqual(len(list((folder / 'data').rglob('*_train.jsonl'))), 2)

    def test_recursively_trains_on_pdf_and_added_elf_without_test_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = folder / 'data'
            for kind in ('PDF', 'ELF'):
                target = data / kind / f'sample_{kind}_train.jsonl'
                target.parent.mkdir(parents=True)
                extractor = PEFeatureExtractor()
                with target.open('w') as stream:
                    for i in range(40):
                        record = extractor.raw_features(kind.encode() + bytes([i]) * 512)
                        record['label'] = i % 2
                        stream.write(json.dumps(record) + '\n')
                    stream.write('{"label": null}\n{"label": -1}\n')
            # Invalid content must never be parsed from held-out files.
            (data / 'sample_PDF_test.jsonl').write_text('not training data')
            (data / 'sample_challenge_malicious.jsonl').write_text('not training data')
            output = folder / 'output'
            result = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--data-dir', str(data),
                '--output-dir', str(output), '--rounds', '2', '--threads', '1',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'training.json').read_text())
            self.assertEqual(report['rows'], 80)
            self.assertEqual(len(report['files']), 2)
            self.assertEqual(report['labels'], {'0': 40, '1': 40})
            model = lgb.Booster(model_file=str(output / 'model.txt'))
            self.assertEqual(model.num_feature(), 2568)

    def test_empty_data_fails_without_writing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--data-dir', folder,
                '--output-dir', str(Path(folder) / 'output'),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No training JSONL files', result.stderr)
            self.assertFalse((Path(folder) / 'output/model.txt').exists())

    def test_invalid_label_fails_with_source_location(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad_PDF_train.jsonl'
            path.write_text('{"label": 2}\n')
            result = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--data-dir', folder,
                '--output-dir', str(Path(folder) / 'output'),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f'{path}:1:', result.stderr)
            self.assertIn('Expected binary label', result.stderr)
            self.assertFalse((Path(folder) / 'output/model.txt').exists())


if __name__ == '__main__':
    unittest.main()
