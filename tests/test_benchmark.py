import unittest
import numpy as np

class BenchmarkTest(unittest.TestCase):
    def test_file_type_metrics_do_not_hide_elf_errors_in_pdf_results(self):
        from dataset_hub.benchmark import metrics_by_file_type
        result=metrics_by_file_type([{'file_type':kind} for kind in ['PDF','PDF','ELF','ELF']],
                                   np.array([0,1,0,1]),np.array([0.1,0.9,0.8,0.2]),0.5,['PDF','ELF','APK'])
        self.assertEqual(result['PDF']['accuracy'],1.0)
        self.assertEqual(result['ELF']['confusion'],{'tp':0,'tn':0,'fp':1,'fn':1})
        self.assertEqual(result['APK']['status'],'not_evaluated')

    def test_threshold_is_selected_under_validation_false_positive_budget(self):
        from dataset_hub.benchmark import choose_threshold
        threshold = choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.5, 0.9]), 0.0)
        self.assertGreater(threshold, 0.6)
        self.assertLessEqual(threshold, 0.9)

    def test_chronological_partition_keeps_later_files_for_validation(self):
        from dataset_hub.benchmark import split_training
        rows = [{'id': str(i), 'date': f'2024-0{i+1}-01', 'label': i % 2} for i in range(4)]
        train, validation = split_training(rows, 'chronological', 0.5, 0)
        self.assertEqual([r['id'] for r in train], ['0', '1'])
        self.assertEqual([r['id'] for r in validation], ['2', '3'])

    def test_invalid_dataset_and_parameter_overrides_fail_before_training(self):
        from dataset_hub.benchmark import validate_settings
        from dataset_hub.configuration import load_config
        settings = load_config()
        settings['benchmark']['dataset'] = 'unknown'
        with self.assertRaises(ValueError): validate_settings(settings)

    def test_cross_split_leakage_is_rejected(self):
        import tempfile, json
        from pathlib import Path
        from dataset_hub.benchmark import load_subset
        from dataset_hub.configuration import load_config
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            record={'label':1,'sha256':'a'*64,'file_type':'PDF'}
            for split in ('train','test'):
                (root/f'sample_PDF_{split}.jsonl').write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError,'Cross-split leakage'):
                load_subset(root,load_config()['benchmark'])

    def test_real_subset_experiment_preserves_models_and_selects_on_validation(self):
        import tempfile, json
        from pathlib import Path
        from thrember import PEFeatureExtractor
        from dataset_hub.configuration import load_config
        from dataset_hub.benchmark import run_benchmark
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'PDF';data.mkdir();extractor=PEFeatureExtractor()
            for number,(split,count) in enumerate([('train',120),('test',20),('challenge',12)]):
                with (data/f'sample_PDF_{split}.jsonl').open('w') as stream:
                    for i in range(count):
                        record=extractor.raw_features(b'PDF'+bytes([65+i%2])*512+number.to_bytes(2,'little')+i.to_bytes(2,'little'))
                        record.update(label=i%2 if split!='challenge' else 1,file_type='PDF' if i%4<2 else 'ELF')
                        stream.write(json.dumps(record)+'\n')
            settings=load_config();settings['paths']['data_dir']=str(data)
            settings['runtime'].update(threads=1,xgboost_device='cpu');settings['training']['rounds']=3
            settings['benchmark'].update(file_types=['PDF','ELF'], split_strategy='hash',max_train_rows=100,max_test_rows=16,max_challenge_rows=8,
                                         training_fractions=[1.0],early_stopping_rounds=2)
            settings['candidates']=[{'name':'lgb','backend':'lightgbm','parameters':{'min_data_in_leaf':2}},
                                    {'name':'xgb','backend':'xgboost','parameters':{'max_depth':2}}]
            output=root/'experiment';report=run_benchmark(settings,output)
            self.assertEqual(set(report['selected_groups']),{'lightgbm','xgboost'})
            self.assertEqual(len(report['trials']),2)
            self.assertTrue((output/'audit.json').is_file())
            self.assertTrue((output/'benchmark.md').is_file())
            for trial in report['trials']:
                self.assertTrue((output/trial['model_path']).is_file())
                self.assertGreater(trial['validation']['rows'],0)
            self.assertEqual(len(report['held_out']),2)
            for result in report['held_out'].values():
                self.assertEqual(set(result['test']['by_file_type']),{'PDF','ELF'})
                self.assertEqual(sum(m['rows'] for m in result['test']['by_file_type'].values()),result['test']['rows'])
