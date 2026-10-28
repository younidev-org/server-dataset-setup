import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from thrember import PEFeatureExtractor

ROOT = Path(__file__).resolve().parent


class EvaluationTest(unittest.TestCase):
    def test_metrics_match_hand_calculated_confusion_counts(self):
        from evaluate import classification_metrics
        result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], 0.5)
        self.assertEqual(result['confusion'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
        for name in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate'):
            self.assertEqual(result[name], 0.5)
        self.assertEqual(result['roc_auc'], 0.75)

    def test_single_class_challenge_does_not_claim_auc_or_false_positive_rate(self):
        from evaluate import classification_metrics
        result = classification_metrics([1, 1], [0.8, 0.2], 0.5)
        self.assertEqual(result['recall'], 0.5)
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['false_positive_rate'])

    def test_real_model_evaluates_held_out_splits_without_reading_training_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = folder / 'datasets'
            data.mkdir()
            extractor = PEFeatureExtractor()
            for suffix, count in [('PDF_train', 40), ('PDF_test', 8), ('challenge_malicious', 6)]:
                with (data / f'sample_{suffix}.jsonl').open('w') as stream:
                    for i in range(count):
                        record = extractor.raw_features(b'PDF' + bytes([i]) * 512)
                        record['label'] = 1 if suffix.startswith('challenge') else i % 2
                        stream.write(json.dumps(record) + '\n')
                    stream.write('{"label": null}\n{"label": -1}\n')
            model_dir = folder / 'models'
            trained = subprocess.run([
                sys.executable, str(ROOT / 'train.py'), '--data-dir', str(data),
                '--output-dir', str(model_dir), '--rounds', '2', '--threads', '1',
            ], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            (data / 'sample_PDF_train.jsonl').write_text('must not be read during evaluation')
            report_dir = folder / 'reports'
            result = subprocess.run([
                sys.executable, str(ROOT / 'evaluate.py'), '--data-dir', str(data),
                '--model', str(model_dir / 'model.txt'), '--report-dir', str(report_dir),
                '--batch-size', '3', '--threads', '1',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((report_dir / 'evaluation.json').read_text())
            self.assertEqual(report['splits']['test']['rows'], 8)
            self.assertEqual(report['splits']['challenge']['rows'], 6)
            self.assertIsNone(report['splits']['challenge']['roc_auc'])
            self.assertIn('challenge', (report_dir / 'evaluation.md').read_text())


if __name__ == '__main__':
    unittest.main()
