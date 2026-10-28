# Early-development experiments on an EMBER2024 subset

Start with this configurable workflow:

```bash
git pull --ff-only
bash run_all.sh --config configs/ember2024-subset.toml
```

To keep server-specific choices out of tracked files:

```bash
mkdir -p data
cp configs/ember2024-subset.toml data/my-experiment.toml
bash run_all.sh --config data/my-experiment.toml --dry-run
bash run_all.sh --config data/my-experiment.toml
```

The `{config}` placeholder forwards that exact configuration to download and
benchmark steps. Commands remain fully controlled by `[[steps]]`. The old simple
train/evaluate/compare workflow is still available in the root config.toml.

## What you can choose

- `benchmark.dataset`: currently only `"ember2024"` is supported. `variant` names
  the corpus version, initially `"original-pdf-subset"`.
- `file_types`: any subset of PDF, ELF, APK, Win32, Win64, Dot_Net. Downloads fetch
  those train/test archives and the shared challenge archive. Benchmark challenge
  rows are filtered by file-type metadata; unidentified rows are excluded and
  counted, rather than assumed to be PDFs.
- `max_train_rows`, `max_test_rows`, `max_challenge_rows`: positive limits on
  distinct selected samples. The training limit includes validation. The runner
  is deliberately bounded for early development; it does not offer an unchecked
  unlimited-memory mode. Identity auditing still scans all matching source rows.
- `split_strategy`: `chronological` reserves the latest training filename date
  groups for validation; it requires multiple dated training files. `hash` uses
  deterministic identity ordering for undated fixtures and non-temporal studies.
  Hash splitting does not assess generalization to newer malware.
- `validation_fraction`: fraction of dates (chronological) or identities (hash)
  reserved from the sampled training corpus, never from official test data.
- `training_fractions`: nested fractions of the remaining training pool. Use this
  to study learning curves/data efficiency on a common validation set.
- `seeds`: independent training seeds. Start with `[0]` to verify execution, then
  add multiple seeds to measure sensitivity. Sampling/split identities stay fixed.
- `target_fpr`: empirical validation false-positive budget for alarm threshold
  selection. Small validation sets cannot establish a reliable 0.1%/1% deployment
  false-alarm rate. Reject-all thresholds slightly above 1 are possible if no
  nontrivial operating point meets the target.
- `early_stopping_rounds`: stop when the backend's validation ranking metric stops
  improving. `training.rounds` is the maximum budget, not a guaranteed tree count.
- `[[candidates]]`: remove entries to choose algorithms. Each entry has a unique
  name, `backend` (`lightgbm`/`xgboost`), `class_weight` (`balanced`/`none`) and a
  `parameters` table. Each explicit entry is a parameter setting, not a Cartesian
  grid. You can add more entries without changing Python.

The shipped configuration tries two LightGBM and two XGBoost configurations,
each at 50% and 100% training-pool sizes. It uses 5,000 sampled train/validation,
2,000 test and up to 1,000 matching challenge rows, one seed, 200 maximum rounds,
10 threads and automatic CUDA detection. No real model-quality conclusion is
claimed by these defaults. Increase limits, seeds or search size gradually.

Allowed LightGBM parameters: learning_rate, num_leaves, max_depth,
min_data_in_leaf, min_data_in_bin, lambda_l1, lambda_l2, feature_fraction,
bagging_fraction, bagging_freq. Allowed XGBoost parameters: eta, max_depth,
min_child_weight, lambda, alpha, subsample, colsample_bytree, max_bin, gamma.
The protocol controls objectives, threads, device, seeds and class weighting so
candidate overrides cannot silently change the meaning of the experiment.

## Scientific protocol

1. Scan matching records and audit SHA-256 identities. Fail on conflicting labels
   or train/test/challenge overlap. Deduplicate repeated same-label records within
   each source split. This is identity leakage prevention, not a guarantee against
   related-family leakage. Missing identities fail rather than disabling checks.
2. Select deterministic hash-priority subsets without retaining all feature
   records in memory; retain their identities and input file hashes in audit.json.
3. Create a training pool and one common validation set. Both classes must be
   present in training, validation and test; otherwise increase limits/change the
   split. Chronological ordering is by dates in filenames, not guessed timestamps.
4. Train every candidate/fraction/seed using validation-only early stopping. Save
   each model and its exact fit identities, parameters, structure, size and timing.
5. Rank variants using **mean validation AP** across seeds, breaking ties by lower
   mean validation log loss. Seed AP standard deviation is recorded when there are
   multiple seeds; one seed does not support a variability estimate. Select one
   variant per backend, retaining every seed of that variant.
6. Choose each model's alarm threshold using only validation labels/scores to
   maximize recall under the empirical FPR budget. This selection can overfit a
   small validation set; it is not a deployment FPR guarantee.
7. Only after selection, vectorize/evaluate held-out test and matching challenge
   subsets for selected variants. Do not rank candidate parameters using test
   metrics. No final refit is performed, keeping the evaluated model identical to
   the validation-selected model. This intentionally reserves validation data.

Native early-stopping metrics differ: LightGBM uses average precision and
XGBoost uses aucpr with different interpolation. Final cross-backend ranking uses
one shared scikit-learn AP definition. The two backends also differ in categorical
feature handling. A comparison is a practical baseline experiment, not proof that
only the algorithm itself caused a difference.

## Results and choosing the next experiment

The run prints the report path:

```text
data/runs/<run-id>/benchmark/
  benchmark.md          validation leaderboard and selected held-out results
  benchmark.json        full metrics/protocol/software/source hashes
  leaderboard.csv       validation ranking, seed variability and timing
  audit.json            identity selection, counts, hashes and duplicates
  settings.json         resolved configuration
  models/<trial>/       immutable model plus training metadata
  reports/<trial>/      test/challenge metrics, plots, predictions and error CSVs
  test_comparison.csv   per-file outcomes for all selected models/seeds
  challenge_comparison.csv
```

Use AP to study ranking under imbalance, and recall/FPR/precision to judge the
alarm threshold. Also inspect F1, balanced accuracy, MCC, ROC-AUC, Brier/log loss,
Wilson intervals, class counts, calibration plots, tree depths, training time,
prediction time and model size in the full metrics. Do not treat undefined metrics
as zero. The row-level CSVs show which exact samples are detected or missed.

Choose the preferred tradeoff, rather than declaring one universally best AI:
validation AP versus false alarms, missed malware, compute/memory cost and model
size. Training-fraction experiments show whether more data helps. Change one
factor at a time if you need to attribute the effect of class weighting, depth,
learning rate or a changed corpus. Repeatedly inspecting the same test set during
development turns it into an exploratory benchmark; reserve a fresh untouched
final holdout before making final performance claims. Related-family dependence
can make row-wise confidence intervals optimistic. No significance test is claimed.

## A future modified EMBER dataset

Keep original inputs unchanged. Write an EMBER-feature-schema-compatible variant
under a different data directory, change `[paths].data_dir`, give `variant` a new
name and disable the download step. Retain SHA-256 and correct file_type/labels.
Input hashes distinguish modified feature files even when sample identities are
unchanged. When comparing corpus versions, keep a fixed untouched evaluation set
and document whether identities/features/labels changed. New feature schemas,
new objectives, or unrelated datasets require a new adapter; this runner does
not silently claim compatibility with them.
