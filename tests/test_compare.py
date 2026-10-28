import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiment import sha256_file


class ComparisonTest(unittest.TestCase):
    def make_report(self, root, backend, probabilities):
        directory = root / backend
        directory.mkdir()
        csv_path = directory / 'test_predictions.csv'
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_file', 'source_line', 'sha256', 'label', 'malicious_probability'])
            writer.writeheader()
            for index, (label, score) in enumerate(zip([0, 0, 1, 1], probabilities)):
                writer.writerow({'source_file': 'PDF/test.jsonl', 'source_line': index + 1,
                                 'sha256': str(index) * 64, 'label': label, 'malicious_probability': score})
        report = {
            'schema_version': 3, 'backend': backend, 'threshold': 0.5,
            'feature_extractor_sha256': 'same-extractor', 'model_sha256': backend + '-model',
            'training': {'dataset_sha256': 'same-training'},
            'splits': {'test': {'rows': 4, 'dataset_sha256': 'same-test',
                                'prediction_csv_sha256': sha256_file(csv_path)}}
        }
        path = directory / 'evaluation.json'
        path.write_text(json.dumps(report))
        return path

    def test_pairs_samples_and_reports_which_model_is_correct(self):
        from compare import compare_reports
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            lgb = self.make_report(folder, 'lightgbm', [0.1, 0.9, 0.8, 0.1])
            xgb = self.make_report(folder, 'xgboost', [0.1, 0.1, 0.2, 0.9])
            report = compare_reports(lgb, xgb, folder / 'comparison')
            split = report['splits']['test']
            self.assertEqual(split['paired_correctness'], {
                'both_correct': 1, 'only_lightgbm_correct': 1, 'only_xgboost_correct': 2, 'both_wrong': 0})
            self.assertEqual(split['models']['lightgbm']['confusion'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
            self.assertEqual(split['models']['xgboost']['confusion'], {'tn': 2, 'fp': 0, 'fn': 1, 'tp': 1})
            with (folder / 'comparison/test_samples.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]['sha256'], '1' * 64)
            self.assertEqual(rows[1]['lightgbm_outcome'], 'FP')
            self.assertEqual(rows[1]['xgboost_outcome'], 'TN')
            self.assertEqual(rows[3]['correctness_group'], 'only_xgboost_correct')
            self.assertTrue((folder / 'comparison/comparison.md').is_file())

    def test_refuses_different_datasets_or_thresholds_and_tampered_predictions(self):
        from compare import compare_reports
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            lgb = self.make_report(folder, 'lightgbm', [0.1, 0.9, 0.8, 0.1])
            xgb = self.make_report(folder, 'xgboost', [0.1, 0.1, 0.2, 0.9])
            original = json.loads(xgb.read_text())
            changed = json.loads(xgb.read_text())
            changed['splits']['test']['dataset_sha256'] = 'different'
            xgb.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'dataset'):
                compare_reports(lgb, xgb, folder / 'comparison')
            changed = dict(original, threshold=0.7)
            xgb.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'threshold'):
                compare_reports(lgb, xgb, folder / 'comparison')
            xgb.write_text(json.dumps(original))
            with (xgb.parent / 'test_predictions.csv').open('a') as stream:
                stream.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                compare_reports(lgb, xgb, folder / 'comparison')

    def test_refuses_different_source_rows_even_with_matching_dataset_hash(self):
        from compare import compare_reports
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            lgb = self.make_report(folder, 'lightgbm', [0.1, 0.9, 0.8, 0.1])
            xgb = self.make_report(folder, 'xgboost', [0.1, 0.1, 0.2, 0.9])
            path = xgb.parent / 'test_predictions.csv'
            path.write_text(path.read_text().replace('PDF/test.jsonl,4,', 'PDF/test.jsonl,5,'))
            report = json.loads(xgb.read_text())
            report['splits']['test']['prediction_csv_sha256'] = sha256_file(path)
            xgb.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'sample'):
                compare_reports(lgb, xgb, folder / 'comparison')


if __name__ == '__main__':
    unittest.main()
