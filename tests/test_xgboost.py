import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from thrember import PEFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent


class XGBoostTest(unittest.TestCase):
    def test_real_cuda_training_when_available(self):
        from experiment import resolve_xgboost_device
        import numpy as np
        import xgboost as xgb
        device = resolve_xgboost_device('auto', 1)
        if not device['resolved'].startswith('cuda'):
            self.skipTest('CUDA training unavailable: ' + device['reason'])
        matrix = xgb.DMatrix(np.array([[0], [0], [1], [1]], dtype=np.float32), label=[0, 0, 1, 1])
        model = xgb.train({'device': device['resolved'], 'tree_method': 'hist', 'nthread': 1,
                           'objective': 'binary:logistic'}, matrix, num_boost_round=2)
        self.assertTrue(json.loads(model.save_config())['learner']['generic_param']['device'].startswith('cuda'))
        self.assertEqual(len(model.predict(matrix)), 4)

    def test_explicit_cuda_does_not_silently_fall_back_to_cpu_build(self):
        from experiment import resolve_xgboost_device
        from unittest.mock import patch
        with patch('xgboost.build_info', return_value={'USE_CUDA': False}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                resolve_xgboost_device('cuda', 1)
            self.assertEqual(resolve_xgboost_device('auto', 1)['resolved'], 'cpu')

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
                '--rounds', '3', '--threads', '1', '--seed', '42', '--device', 'cpu',
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
            lightgbm_dir = folder / 'data/models/lightgbm'
            baseline = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--backend', 'lightgbm',
                '--data-dir', str(data), '--output-dir', str(lightgbm_dir),
                '--rounds', '3', '--threads', '1', '--seed', '42',
            ], capture_output=True, text=True)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            # Held-out data must not enter fitting, and training data must not enter evaluation.
            (data / 'sample_PDF_train.jsonl').write_text('not evaluation data')
            reports = folder / 'reports'
            evaluated = subprocess.run([
                sys.executable, str(ROOT / 'evaluate.py'), '--backend', 'xgboost',
                '--model', str(model_dir / 'model.ubj'), '--data-dir', str(data),
                '--report-dir', str(reports), '--threads', '1', '--batch-size', '5', '--device', 'cpu',
            ], capture_output=True, text=True)
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            report = json.loads((reports / 'evaluation.json').read_text())
            self.assertEqual(report['backend'], 'xgboost')
            self.assertEqual(report['algorithm']['number_of_trees'], 3)
            self.assertLessEqual(report['algorithm']['tree_depth']['max'], 6)
            self.assertEqual(report['splits']['test']['rows'], 12)
            self.assertEqual(report['splits']['test']['roc_auc'], 1.0)
            self.assertEqual(report['training_metadata_status'], 'verified_model_hash')
            self.assertTrue((reports / 'test_curves.png').is_file())
            import csv
            with (reports / 'test_predictions.csv').open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(predictions[0]['source_file'], 'sample_PDF_test.jsonl')
            self.assertEqual(predictions[0]['source_line'], '1')
            self.assertTrue(predictions[0]['sha256'])
            self.assertIn(predictions[0]['outcome'], ['TN', 'TP', 'FP', 'FN'])
            self.assertTrue(report['splits']['test']['prediction_csv_sha256'])
            baseline_reports = folder / 'lightgbm-reports'
            baseline_eval = subprocess.run([
                sys.executable, str(ROOT / 'evaluate.py'), '--data-dir', str(data),
                '--model', str(lightgbm_dir / 'model.txt'), '--report-dir', str(baseline_reports),
                '--threads', '1',
            ], capture_output=True, text=True)
            self.assertEqual(baseline_eval.returncode, 0, baseline_eval.stdout + baseline_eval.stderr)
            compared = subprocess.run([
                sys.executable, str(ROOT / 'compare.py'),
                '--lightgbm-report', str(baseline_reports / 'evaluation.json'),
                '--xgboost-report', str(reports / 'evaluation.json'),
                '--output-dir', str(folder / 'comparison'),
            ], capture_output=True, text=True)
            self.assertEqual(compared.returncode, 0, compared.stdout + compared.stderr)
            comparison = json.loads((folder / 'comparison/comparison.json').read_text())
            self.assertTrue(comparison['training_dataset_match'])
            self.assertEqual(comparison['splits']['test']['rows'], 12)


if __name__ == '__main__':
    unittest.main()
