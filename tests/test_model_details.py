import unittest
import numpy as np
import lightgbm as lgb
import xgboost as xgb

class ModelDetailsTest(unittest.TestCase):
    def test_actual_tree_structure_for_both_backends(self):
        from dataset_hub.model_details import model_details
        features = np.array([[i % 2] for i in range(32)], dtype=np.float32)
        labels = features[:, 0]
        models = {
            'lightgbm': lgb.train({'objective': 'binary', 'min_data_in_leaf': 1, 'min_data_in_bin': 1,
                                   'num_threads': 1, 'max_depth': 2, 'verbosity': -1},
                                  lgb.Dataset(features, label=labels), num_boost_round=2),
            'xgboost': xgb.train({'objective': 'binary:logistic', 'max_depth': 2, 'nthread': 1},
                                 xgb.DMatrix(features, label=labels), num_boost_round=2),
        }
        for backend, model in models.items():
            with self.subTest(backend=backend):
                details = model_details(model, backend)
                self.assertEqual(details['number_of_trees'], 2)
                self.assertEqual(details['tree_depth']['per_tree'], [1, 1])
                self.assertEqual(details['leaves_per_tree'], [2, 2])
                self.assertTrue(details['objective'])
