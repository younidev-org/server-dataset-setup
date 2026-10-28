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

ROOT = Path(__file__).resolve().parent


class TrainingTest(unittest.TestCase):
    def test_default_paths_keep_training_artifacts_under_data(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            shutil.copy2(ROOT / 'train.py', folder / 'train.py')
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
            self.assertFalse((folder / 'output').exists())

    def test_downloader_requests_only_pdf_training_and_extracts_into_pdf_folder(self):
        import download_pdf
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive = folder / 'PDF_train.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('sample_PDF_train.jsonl', '{"label": 0}\n')
            previous = Path.cwd()
            try:
                with patch('sys.argv', ['download_pdf.py', '--data-dir', str(folder / 'data')]), patch(
                    'thrember.download.hf_hub_download', return_value=str(archive)
                ) as download:
                    download_pdf.main()
                    download.assert_called_once_with(
                        repo_id='joyce8/EMBER2024', filename='PDF_train.zip', repo_type='dataset'
                    )
            finally:
                os.chdir(previous)
            self.assertEqual(
                (folder / 'data/PDF/sample_PDF_train.jsonl').read_text(), '{"label": 0}\n'
            )

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
