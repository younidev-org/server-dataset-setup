"""Bounded EMBER experiments: audit, validation-only selection and final evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import inspect
import json
from pathlib import Path
import re
import time
import uuid

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from thrember import PEFeatureExtractor
from sklearn.metrics import roc_curve

from .configuration import command_config
from .experiment import input_manifest, sha256_file, software_versions, resolve_xgboost_device
from .model_details import model_details
from .reporting import classification_metrics, save_figures, save_predictions
from .train import training_records

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {
    'lightgbm': {'learning_rate', 'num_leaves', 'max_depth', 'min_data_in_leaf', 'min_data_in_bin', 'lambda_l1', 'lambda_l2', 'feature_fraction', 'bagging_fraction', 'bagging_freq'},
    'xgboost': {'eta', 'max_depth', 'min_child_weight', 'lambda', 'alpha', 'subsample', 'colsample_bytree', 'max_bin', 'gamma'},
}


def validate_settings(settings):
    b = settings['benchmark']
    if b['dataset'] != 'ember2024': raise ValueError('Only the ember2024 dataset adapter is supported')
    if not isinstance(b['variant'], str) or not b['variant'].strip(): raise ValueError('variant must be a nonempty dataset label')
    if not isinstance(b['file_types'], list): raise ValueError('file_types must be an array')
    if not b['file_types'] or set(b['file_types']) - {'PDF', 'ELF', 'APK', 'Win32', 'Win64', 'Dot_Net'}:
        raise ValueError('Select supported EMBER2024 file_types')
    for key in ('max_train_rows', 'max_test_rows', 'max_challenge_rows', 'early_stopping_rounds'):
        if type(b[key]) is not int or b[key] < 1: raise ValueError(f'{key} must be positive; this runner is a bounded subset benchmark')
    if not 0 < b['validation_fraction'] < 1 or not 0 <= b['target_fpr'] < 1: raise ValueError('Invalid validation fraction/target FPR')
    if b['split_strategy'] not in ('hash', 'chronological'): raise ValueError('split_strategy must be hash or chronological')
    if not b['training_fractions'] or any(type(f) not in (int, float) or not 0 < f <= 1 for f in b['training_fractions']): raise ValueError('training_fractions must be in (0, 1]')
    if not b['seeds'] or any(type(seed) is not int or seed < 0 for seed in b['seeds']): raise ValueError('Use nonnegative integer seeds')
    if len(set(b['seeds'])) != len(b['seeds']): raise ValueError('Seeds must be unique')
    if len(set(b['training_fractions'])) != len(b['training_fractions']): raise ValueError('Training fractions must be unique')
    if not isinstance(settings['candidates'], list): raise ValueError('candidates must use [[candidates]] tables')
    names = set()
    if not settings['candidates']: raise ValueError('Define at least one [[candidates]] entry')
    for c in settings['candidates']:
        if not isinstance(c, dict) or not isinstance(c.get('parameters', {}), dict): raise ValueError('Each candidate and its parameters must be tables')
        if set(c) - {'name', 'backend', 'class_weight', 'parameters'}: raise ValueError('Unknown candidate field')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', c.get('name', '')) or c['name'] in names: raise ValueError('Candidate names must be unique safe identifiers')
        names.add(c['name'])
        if c.get('backend') not in ALLOWED: raise ValueError('Supported algorithms: lightgbm and xgboost')
        if c.get('class_weight', 'balanced') not in ('balanced', 'none'): raise ValueError('class_weight must be balanced or none')
        if set(c.get('parameters', {})) - ALLOWED[c['backend']]: raise ValueError('Unsupported candidate parameter; objective/device/seed are controlled by the protocol')


def priority(identity, seed=0):
    return int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(), 16)


def split_training(rows, strategy, fraction, seed):
    if strategy == 'chronological':
        if any(not r['date'] for r in rows): raise ValueError('Chronological split needs dated training filenames; use hash for undated fixtures')
        dates = sorted({r['date'] for r in rows})
        if len(dates) < 2: raise ValueError('Chronological split requires at least two training date groups')
        boundary = dates[max(1, min(len(dates)-1, int(len(dates)*(1-fraction))))]
        return [r for r in rows if r['date'] < boundary], [r for r in rows if r['date'] >= boundary]
    ordered = sorted(rows, key=lambda r: priority(r['id'], seed))
    count = max(1, min(len(rows)-1, int(len(rows)*fraction)))
    return ordered[:-count], ordered[-count:]


def choose_threshold(labels, scores, target_fpr):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    options = [(float(tpr[i]), float(thresholds[i])) for i in range(len(fpr)) if fpr[i] <= target_fpr]
    # Tie-break toward the higher threshold (fewer alarms). A finite reject-all threshold is allowed.
    recall, threshold = max(options)
    return float(np.nextafter(1.0, 2.0)) if not np.isfinite(threshold) else threshold


def load_subset(data, b):
    seen, counts, heaps = {}, Counter(), {split: [] for split in ('train', 'test', 'challenge')}
    selected = set(b['file_types'])
    paths = sorted(data.rglob('*.jsonl'))
    used_paths = []
    for path in paths:
        name = path.name.lower()
        split = 'challenge' if 'challenge' in name else 'train' if name.endswith('_train.jsonl') else 'test' if name.endswith('_test.jsonl') else None
        if split is None: continue
        used_paths.append(path)
        for record, source, line in training_records([path], counts, with_source=True):
            kind = record.get('file_type')
            if not kind:
                kind = next((k for k in selected if re.search(r'(^|[/_])'+re.escape(k)+r'([/_\.]|$)', str(path.relative_to(data)), re.I)), None)
            if str(kind).upper() not in {k.upper() for k in selected}:
                counts['filtered_file_type'] += 1; continue
            identity = record.get('sha256', '').lower()
            if not re.fullmatch('[0-9a-f]{64}', identity): raise ValueError(f'Missing/invalid SHA-256 at {source}:{line}; identity is required for leakage checks')
            if identity in seen:
                old_split, old_label = seen[identity]
                if old_label != record['label']: raise ValueError(f'Conflicting labels for SHA-256 {identity}')
                if old_split != split: raise ValueError(f'Cross-split leakage: {identity} appears in {old_split} and {split}')
                counts[f'{split}_duplicates_removed'] += 1; continue
            seen[identity] = split, record['label']
            counts[f'{split}_unique_available'] += 1
            match = re.search(r'\d{4}-\d{2}-\d{2}', path.name)
            row = {'id': identity, 'date': match.group() if match else None, 'label': record['label'],
                   'record': record, 'source_file': str(path.relative_to(data)), 'source_line': line, 'sha256': identity, 'file_type': kind or ''}
            heap = heaps[split]; cap = b[f'max_{split}_rows']
            item = (-priority(identity), identity, row)
            if len(heap) < cap: heapq.heappush(heap, item)
            elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
    return {split: sorted([item[2] for item in heap], key=lambda r: r['id']) for split, heap in heaps.items()}, dict(counts), used_paths


def require_classes(rows, name):
    if {r['label'] for r in rows} != {0, 1}: raise ValueError(f'{name} needs both classes; increase row limits or change the validation split')


def vectors(rows, extractor):
    return np.asarray([extractor.process_raw_features(r['record']) for r in rows], dtype=np.float32), np.array([r['label'] for r in rows])


def fit(candidate, features, labels, val_x, val_y, settings, seed, device):
    threads = settings['runtime']['threads']; backend = candidate['backend']
    imbalance = float(np.sum(labels == 0) / np.sum(labels == 1))
    if backend == 'lightgbm':
        params = {'objective': 'binary', 'metric': 'average_precision', 'learning_rate': 0.1, 'num_leaves': 64,
                  'min_data_in_leaf': 100, 'seed': seed, 'num_threads': threads, 'verbosity': -1,
                  'is_unbalance': candidate.get('class_weight', 'balanced') == 'balanced'}
        params.update(candidate.get('parameters', {}))
        train = lgb.Dataset(features, label=labels, categorical_feature=[2,3,4,5,6,701,702])
        valid = lgb.Dataset(val_x, label=val_y, reference=train)
        model = lgb.train(params, train, num_boost_round=settings['training']['rounds'], valid_sets=[valid],
                          callbacks=[lgb.early_stopping(settings['benchmark']['early_stopping_rounds'], verbose=False)])
        return model, params
    params = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'device': device['resolved'],
              'eta': 0.1, 'max_depth': 6, 'seed': seed, 'nthread': threads,
              'scale_pos_weight': imbalance if candidate.get('class_weight', 'balanced') == 'balanced' else 1.0}
    params.update(candidate.get('parameters', {}))
    model = xgb.train(params, xgb.DMatrix(features, label=labels, nthread=threads),
                      num_boost_round=settings['training']['rounds'], evals=[(xgb.DMatrix(val_x, label=val_y, nthread=threads), 'validation')],
                      early_stopping_rounds=settings['benchmark']['early_stopping_rounds'], verbose_eval=False)
    return model[:model.best_iteration+1], params


def predict(model, backend, features, threads):
    return np.asarray(model.predict(features, num_threads=threads) if backend == 'lightgbm'
                      else model.predict(xgb.DMatrix(features, nthread=threads)))


def run_benchmark(settings, output):
    validate_settings(settings)
    if output.exists(): raise ValueError('Benchmark output directory already exists; use a new directory to preserve experiments')
    output.mkdir(parents=True)
    b = settings['benchmark']; data = (ROOT / settings['paths']['data_dir']).resolve()
    rows, audit, paths = load_subset(data, b)
    manifest = input_manifest(paths, data)
    train, validation = split_training(rows['train'], b['split_strategy'], b['validation_fraction'], settings['training']['seed'])
    require_classes(train, 'Training pool'); require_classes(validation, 'Validation'); require_classes(rows['test'], 'Test')
    selected_ids = {split: [r['id'] for r in selected] for split, selected in dict(rows, train_pool=train, validation=validation).items()}
    (output/'audit.json').write_text(json.dumps({'counts': audit, 'selected_sha256': selected_ids, 'input_manifest': manifest}, indent=2))
    (output/'settings.json').write_text(json.dumps(settings, indent=2))
    extractor = PEFeatureExtractor(); val_x, val_y = vectors(validation, extractor)
    device = resolve_xgboost_device(settings['runtime']['xgboost_device'], settings['runtime']['threads']) if any(c['backend']=='xgboost' for c in settings['candidates']) else {'resolved':'cpu'}
    trials, models, groups = [], {}, defaultdict(list)
    # All candidates use exactly the same validation rows. Test/challenge features are not vectorized here.
    for candidate in settings['candidates']:
        for fraction in b['training_fractions']:
            chosen = sorted(train, key=lambda r: priority(r['id']))[:max(1, int(len(train)*fraction))]
            require_classes(chosen, f'{candidate["name"]} training fraction {fraction}')
            features, labels = vectors(chosen, extractor)
            group = f'{candidate["name"]}-fraction-{fraction}'
            for seed in b['seeds']:
                trial_id = f'{group}-seed-{seed}'; print(f'Training {trial_id}', flush=True)
                started = time.perf_counter()
                model, parameters = fit(candidate, features, labels, val_x, val_y, settings, seed, device)
                fit_seconds = time.perf_counter()-started
                backend = candidate['backend']; model_dir = output/'models'/trial_id; model_dir.mkdir(parents=True)
                model_path = model_dir / ('model.txt' if backend=='lightgbm' else 'model.ubj')
                model.save_model(str(model_path))
                model = lgb.Booster(model_file=str(model_path)) if backend=='lightgbm' else model
                scores = predict(model, backend, val_x, settings['runtime']['threads'])
                threshold = choose_threshold(val_y, scores, b['target_fpr'])
                metrics = classification_metrics(val_y, scores, threshold)
                trial = {'id':trial_id, 'group':group, 'backend':backend, 'seed':seed, 'training_fraction':fraction,
                         'training_rows':len(chosen), 'validation':metrics, 'threshold':threshold,
                         'rows':len(chosen), 'rounds':settings['training']['rounds'],
                         'actual_rounds':model_details(model,backend)['boosting_rounds'],
                         'feature_extractor_sha256':sha256_file(Path(inspect.getfile(PEFeatureExtractor))),
                         'feature_handling':'Bundled EMBER float32 features; LightGBM uses selected categorical indices, XGBoost uses numeric columns.',
                         'dataset_sha256':hashlib.sha256(json.dumps({'files':[entry for entry in manifest if entry['path'].endswith('_train.jsonl') and 'challenge' not in entry['path'].lower()],'ids':sorted(r['id'] for r in chosen)},sort_keys=True).encode()).hexdigest(),
                         'candidate':candidate, 'training_sha256_ids':[r['id'] for r in chosen],
                         'parameters':parameters, 'algorithm':model_details(model,backend), 'fit_seconds':fit_seconds,
                         'model_bytes':model_path.stat().st_size, 'model_sha256':sha256_file(model_path), 'model_path':str(model_path.relative_to(output))}
                (model_dir/'training.json').write_text(json.dumps(trial, indent=2))
                trials.append(trial); groups[group].append(trial); del model
    rankings = sorted([{'group':key, 'backend':items[0]['backend'], 'mean_validation_ap':float(np.mean([i['validation']['average_precision'] for i in items])),
                       'mean_validation_log_loss':float(np.mean([i['validation']['log_loss'] for i in items])),
                       'validation_ap_std':float(np.std([i['validation']['average_precision'] for i in items],ddof=1)) if len(items)>1 else None,
                       'replicates':len(items), 'mean_fit_seconds':float(np.mean([i['fit_seconds'] for i in items]))} for key,items in groups.items()],
                       key=lambda g:(-g['mean_validation_ap'],g['mean_validation_log_loss'],g['group']))
    winners = {}
    for ranking in rankings: winners.setdefault(ranking['backend'], ranking['group'])
    results = {}; paired_scores = defaultdict(dict)
    # Evaluate only the validation-selected variant per backend, retaining every configured seed.
    for backend, group in winners.items():
        for trial in groups[group]:
            directory = output/'reports'/trial['id']; directory.mkdir(parents=True)
            model_path = output/trial['model_path']
            model = lgb.Booster(model_file=str(model_path)) if backend == 'lightgbm' else xgb.Booster(model_file=str(model_path))
            if backend == 'xgboost': model.set_param({'device':device['resolved'], 'nthread':settings['runtime']['threads']})
            result = {}
            for split in ('test','challenge'):
                if not rows[split]:
                    result[split] = {'status':'not_evaluated', 'reason':'No matching file-type records in the selected corpus'}; continue
                features, labels = vectors(rows[split], extractor)
                started=time.perf_counter(); scores=predict(model,backend,features,settings['runtime']['threads'])
                paired_scores[split][trial['id']] = (scores, trial['threshold'])
                metrics=classification_metrics(labels,scores,trial['threshold']); metrics['prediction_seconds']=time.perf_counter()-started
                identities=[{k:r[k] for k in ('source_file','source_line','sha256','file_type')} for r in rows[split]]
                save_predictions(labels,scores,identities,trial['threshold'],directory/f'{split}_predictions.csv')
                metrics['calibration_bins']=save_figures(labels,scores,metrics,split,directory)
                result[split]=metrics
            (directory/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)); results[trial['id']]=result
    for split, predictions in paired_scores.items():
        with (output/f'{split}_comparison.csv').open('w',newline='') as stream:
            fields=['sha256','source_file','source_line','label']+[f'{trial}:{value}' for trial in predictions for value in ('probability','outcome')]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            from .reporting import prediction_outcome
            for index,row in enumerate(rows[split]):
                entry={key:row[key] for key in ('sha256','source_file','source_line','label')}
                for trial,(scores,threshold) in predictions.items():
                    entry[f'{trial}:probability']=float(scores[index]);entry[f'{trial}:outcome']=prediction_outcome(row['label'],scores[index],threshold)
                writer.writerow(entry)
    if input_manifest(paths,data)!=manifest: raise ValueError('Input files changed during benchmark; do not use the results')
    report={'protocol':'Validation-only AP selection; validation-selected empirical FPR threshold; no final refit or held-out tuning',
            'dataset':b['dataset'], 'variant':b['variant'], 'file_types':b['file_types'], 'software':software_versions(),
            'feature_extractor_sha256':sha256_file(Path(inspect.getfile(PEFeatureExtractor))),
            'benchmark_source_sha256':sha256_file(Path(__file__)), 'overall_validation_selection':rankings[0]['group'],
            'device':device, 'rankings':rankings, 'selected_groups':winners, 'trials':trials, 'held_out':results}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (output/'leaderboard.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rankings[0]));writer.writeheader();writer.writerows(rankings)
    lines=['# EMBER2024 subset experiment','',report['protocol'],'',
           '| Variant | Backend | Mean validation AP ↑ | Mean validation log loss ↓ | Selected |',
           '| --- | --- | ---: | ---: | --- |']
    for r in rankings: lines.append(f'| {r["group"]} | {r["backend"]} | {r["mean_validation_ap"]:.6f} | {r["mean_validation_log_loss"]:.6f} | {winners[r["backend"]]==r["group"]} |')
    lines += ['', '## Selected models: held-out results', '', '| Trial / split | n | TP | TN | FP | FN | Recall | FPR | AP | Brier |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for trial,result in results.items():
        for split,m in result.items():
            if 'rows' not in m: lines.append(f'| {trial}/{split}: not evaluated | — | — | — | — | — | — | — | — | — |');continue
            c=m['confusion']; values=[trial+'/'+split,str(m['rows'])]+[str(c[k]) for k in ('tp','tn','fp','fn')]+['N/A' if m[k] is None else f'{m[k]:.6f}' for k in ('recall','false_positive_rate','average_precision','brier_score')]
            lines.append('| '+' | '.join(values)+' |')
    for trial_id, result in results.items():
        trial = next(item for item in trials if item['id'] == trial_id)
        depth = trial['algorithm']['tree_depth']
        lines += ['', f'## Selected model: {trial_id}', '',
                  f'Validation-selected threshold: {trial["threshold"]}; training rows: {trial["training_rows"]}; '
                  f'trees: {trial["algorithm"]["number_of_trees"]}; observed depth min/mean/max: '
                  f'{depth["min"]}/{depth["mean"]}/{depth["max"]}; fit seconds: {trial["fit_seconds"]:.3f}; model bytes: {trial["model_bytes"]}.', '',
                  '```json', json.dumps(trial['parameters'], indent=2), '```', '']
        for split, metrics in result.items():
            if 'rows' not in metrics: continue
            lines += [f'### {split}', '',
                      f'[All metrics and Wilson intervals](reports/{trial_id}/metrics.json) · '
                      f'[Predictions](reports/{trial_id}/{split}_predictions.csv) · '
                      f'[False positives](reports/{trial_id}/{split}_false_positives.csv) · '
                      f'[Missed malware](reports/{trial_id}/{split}_false_negatives.csv)', '',
                      f'![ROC, precision–recall, reliability](reports/{trial_id}/{split}_curves.png)', '',
                      f'![Confusion counts](reports/{trial_id}/{split}_confusion.png)', '']
    lines += ['', '## Artifacts and interpretation', '',
              '- [Validation leaderboard](leaderboard.csv), [full results](benchmark.json), [leakage/subset audit](audit.json), [resolved settings](settings.json).',
              '- Per-sample test_comparison.csv and challenge_comparison.csv identify which selected model/seed detects or misclassifies each file, using its own validation-selected threshold.',
              '- Every trial model and its parameters, tree structure, timing and hash are preserved in models/. Selected trial prediction/error CSVs and plots are in reports/.',
              '- Training fractions are nested deterministic subsets of the same training pool. All variants share validation rows. Seeds are replicates; selection uses mean validation AP, with mean log loss as a tie-breaker.',
              '- Native early stopping uses validation ranking metrics (LightGBM average precision; XGBoost aucpr has a different interpolation). Final candidate ranking uses the same scikit-learn AP for both.',
              '- Thresholds meet an empirical validation FPR target, not a guaranteed deployment FPR. Small benign validation counts cannot establish a low false-alarm rate. Full JSON includes Wilson rate intervals and all classification/calibration metrics.',
              '- Chronological splitting separates training filename date groups; hash splitting separates sample identities but does not emulate time drift.',
              '- Single-class challenge ranking/FPR metrics are undefined. Challenge is restricted to selected file types where identifiable, not treated as a PDF-only set by assumption.',
              '- No final refit is performed: selected models retain the validation-excluded fit. No universally best AI or significance claim is made; seeds and hardware influence results.',
              '- Repeated development comparisons on the same test set make it an exploratory benchmark. Keep a new untouched final holdout for the eventual modified dataset.', '']
    (output/'benchmark.md').write_text('\n'.join(lines))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);settings=command_config(parser)
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    output=args.output_dir or ROOT/settings['paths']['run_root']/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])/'benchmark'
    run_benchmark(settings,output.resolve());print(f'Experiment report: {output.resolve()/"benchmark.md"}',flush=True)
