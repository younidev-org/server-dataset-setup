import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from thrember import PEFeatureExtractor

ROOT = Path(__file__).resolve().parent


class XGBoostTest(unittest.TestCase):
    def test_trains_and_evaluates_same_features_with_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = folder / 'data/datasets/PDF'
            data.mkdir(parents=True)
            extractor = PEFeatureExtractor()
            for split, count in [('train', 80), ('test', 12), ('challenge', 6)]:
                with (data / f'sample_PDF_{split}.jsonl').open('w') as stream:
                    for i in range(count):
                        record = extractor.raw_features(b'PDF' + bytes([i % 2 + 65]) * 512)
                        record['label'] = i % 2 if split != 'challenge' else 1
                        stream.write(json.dumps(record) + '\n')
            model_dir = folder / 'data/models/xgboost'
            trained = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--backend', 'xgboost',
                '--data-dir', str(data), '--output-dir', str(model_dir),
                '--rounds', '3', '--threads', '1', '--seed', '42',
            ], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            self.assertTrue((model_dir / 'model.ubj').is_file())
            metadata = json.loads((model_dir / 'training.json').read_text())
            self.assertEqual(metadata['backend'], 'xgboost')
            self.assertEqual(metadata['rows'], 80)
            self.assertEqual(metadata['seed'], 42)
            self.assertEqual(metadata['actual_rounds'], 3)
            self.assertEqual(metadata['parameters']['tree_method'], 'hist')
            self.assertTrue(metadata['model_sha256'])
            self.assertEqual(len(metadata['input_manifest']), 1)
            # Held-out data must not enter fitting, and training data must not enter evaluation.
            (data / 'sample_PDF_train.jsonl').write_text('not evaluation data')
            reports = folder / 'reports'
            evaluated = subprocess.run([
                sys.executable, str(ROOT / 'evaluate.py'), '--backend', 'xgboost',
                '--model', str(model_dir / 'model.ubj'), '--data-dir', str(data),
                '--report-dir', str(reports), '--threads', '1', '--batch-size', '5',
            ], capture_output=True, text=True)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            report = json.loads((reports / 'evaluation.json').read_text())
            self.assertEqual(report['backend'], 'xgboost')
            self.assertEqual(report['splits']['test']['rows'], 12)
            self.assertEqual(report['splits']['test']['roc_auc'], 1.0)
            self.assertEqual(report['training_metadata_status'], 'verified_model_hash')
            self.assertTrue((reports / 'test_curves.png').is_file())


if __name__ == '__main__':
    unittest.main()
