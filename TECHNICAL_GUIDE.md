# Technical guide: models, settings, and evaluation reports

This guide explains the terminology used by this repository. **Positive means
malicious (`1`); negative means benign (`0`).** Counts and metrics are calculated
against the dataset's supplied labels, not an independent forensic investigation.

## Reading the main results table

| Column | Meaning | Formula / interpretation |
| --- | --- | --- |
| Model | Classifier being evaluated, such as LightGBM or XGBoost. | Compare models on the same labeled samples and threshold. |
| Correct | Number of correctly classified samples. | `TP + TN`; this is a count, not a percentage. |
| TP | True positives: malicious samples correctly flagged. | Successful malware detections. |
| TN | True negatives: benign samples correctly accepted. | Successful benign classifications. |
| FP | False positives: benign samples incorrectly flagged as malicious. | False alarms. |
| FN | False negatives: malicious samples incorrectly accepted as benign. | Missed malware. |
| Recall | Fraction of malicious samples detected; also called sensitivity or true-positive rate. | `TP / (TP + FN)`; higher is better. |
| FPR | False-positive rate: fraction of benign samples incorrectly flagged. | `FP / (FP + TN)`; lower is better. |
| Precision | Fraction of flagged samples that really are malicious. | `TP / (TP + FP)`; depends on class prevalence. |
| F1 | Harmonic mean of precision and recall. | `2*TP / (2*TP + FP + FN)`; higher is better. It does not use TN. |
| ROC-AUC | Area under the receiver operating characteristic curve across score thresholds. | Measures ranking: a randomly selected malicious sample should score above a randomly selected benign sample. Tied scores receive half credit. |
| AP | Average precision across score thresholds. | Step-weighted precision–recall summary; higher is better. It is not trapezoidal PR area. |

A score of `0.499921` is approximately **49.9921%**, not 0.499921%.
A zero denominator produces `N/A` in Markdown and `null` in JSON. It does not
mean zero, a perfect result, a failed download, or missing predictions.

### Your example: precision 1.0 while half the malware is missed

| Correct | TP | TN | FP | FN | Recall | FPR | Precision | F1 | ROC-AUC | AP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3157 | 3157 | 0 | 0 | 3158 | 0.499921 | N/A | 1.000000 | 0.666596 | N/A | N/A |

There are **6,315 malicious samples** (`TP + FN`) and **no benign samples**
(`TN + FP = 0`). The model detects 3,157 and misses 3,158.

- Recall is `3157 / 6315`, about **49.99%**: approximately half the malware is missed.
- Precision is `3157 / (3157 + 0) = 1`: every alarm in this particular set is on
  malware. With no benign samples, that cannot establish real-world precision.
- FP is zero because this set has no benign samples to produce false alarms.
  **This does not demonstrate a zero false-positive rate.** FPR is undefined:
  `0 / (0 + 0)`.
- ROC-AUC needs both classes and is undefined here. This implementation also
  reports AP as N/A on single-class splits rather than presenting a trivial
  all-positive ranking score as evidence of discrimination.
- F1 is `6314 / 9472`, about **0.666596**. It is not the percentage of files
  classified correctly. Accuracy on this set is about **49.99%**.

To measure false alarms and discrimination, inspect the test split with benign
and malicious samples. The shared challenge set is a stress test and is **not
PDF-only**; its results should not be described as PDF-only accuracy.

## Other metrics and statistical terms

| Term | Meaning |
| --- | --- |
| Accuracy | `(TP + TN) / total`. Can look high when the majority class dominates. |
| Specificity | `TN / (TN + FP)`, the fraction of benign samples accepted correctly; equals `1 - FPR` when defined. |
| Balanced accuracy | Mean of recall and specificity; gives the two classes equal weight. Undefined here when a class is absent. |
| MCC | Matthews correlation coefficient: combines all four confusion counts. Ranges from -1 (opposite labels), through 0, to +1 (perfect). Undefined when its denominator is zero. |
| Prevalence | Fraction of labeled samples that are malicious. It affects precision and provides the reference baseline for AP. |
| Brier score | Mean squared error between malicious scores and binary labels. Lower is better. |
| Log loss | Probability error that strongly penalizes confident wrong predictions. Lower is better; exact 0/1 scores are numerically clipped by the metric implementation. |
| Calibration | Whether scores match observed frequencies. Among comparable samples scored near 0.8, roughly 80% should be malicious if well calibrated. This is not guaranteed by good ranking. |
| Reliability diagram | Compares mean score with observed malicious frequency in score bins. This report uses 10 equal-width bins; bin counts are in JSON. Small bins are uncertain. |
| 95% Wilson interval | Uncertainty interval for a binomial rate, such as recall or FPR. Even zero observed false positives generally gives a nonzero upper bound. |
| Point estimate | A single measured value without an uncertainty interval. AUC, AP, F1, MCC, Brier score and log loss are reported as point estimates here. |

Wilson intervals assume independent samples. Duplicate files, related malware
families and temporal correlation can make them too optimistic. A 95% confidence
interval is not a guarantee about the next dataset and is not a statement that
95% of predictions are correct. The reports do not perform significance tests or
establish a universally better algorithm.

## Scores, thresholds, and comparing the models

The model produces a score intended to represent malicious probability. A
**threshold** converts that score to a label: `score >= threshold` is malicious.
The default is 0.5. Lowering the threshold generally increases detections and
false alarms; raising it generally decreases both. Class weighting can distort
probability calibration, so a score of 0.9 is not automatically a validated 90%
chance of malware.

ROC and precision–recall plots summarize ranking over thresholds; the main
confusion table uses one fixed threshold. Choose thresholds using a separate
validation set, not the final test or challenge set.

The paired comparison distinguishes:

- **Both correct:** both models match the supplied label.
- **Only LightGBM / only XGBoost correct:** one model matches the label and the
  other makes an error on that same sample.
- **Both wrong:** both misclassify the sample, including shared missed malware.
- **Detection overlap:** which model detects each malicious sample.
- **False-positive overlap:** which model wrongly flags each benign sample.

Use the `*_samples.csv`, `*_errors.csv`, `*_disagreements.csv` and
`*_detections.csv` files to identify the actual rows. Each includes the sample
SHA-256 when available and its source JSONL path and 1-based line number.
A SHA-256 identifies the original file; it does not retrieve the original PDF.
These downloads contain precomputed features, not original documents.

Counts are **row counts**. Repeated file hashes remain separate rows; the
comparison matches source path and line without silently deduplicating them.
Dataset hashes, labels, thresholds and feature-extractor hashes must match before
results can be paired. Missing training manifests are explicitly unverified.

## Algorithm and fitted-model terminology

| Term | Meaning |
| --- | --- |
| LightGBM / LGBM | A gradient-boosted decision-tree library. This workflow uses its CPU backend and selected categorical feature indices. |
| XGBoost / XGB | Another gradient-boosted tree library. This workflow uses histogram training on CPU or a compatible NVIDIA GPU, treating the feature columns as numeric. |
| Feature | A numerical input describing a file. The bundled EMBER extractor produces 2,568 feature values per record. A feature is not a file or a label. |
| Decision tree | A sequence of feature-based splits ending in leaf values that contribute to the prediction. |
| Leaf | A terminal node; the sample follows tree splits until it reaches a leaf. |
| Depth | Number of edges from root to deepest leaf. A constant tree has depth 0; a stump with one split has depth 1. |
| Maximum depth | A configured upper limit, not necessarily the depth trees actually reach. A backend's negative/default value can mean no explicit limit. |
| Number of trees | Actual fitted tree count inspected from the model. Do not assume it always equals requested rounds. |
| Boosting round | An iteration that adds a correction to the current ensemble. Our binary, one-tree-per-round setup normally adds one tree per round; LightGBM can stop adding useful trees earlier. Other model configurations can add multiple trees per round. |
| Learning rate | Scales each tree's contribution (currently 0.1 for these baselines). Smaller values often need more rounds. |
| `num_leaves` | LightGBM's maximum leaves per tree. More leaves can increase capacity and overfitting. |
| `min_data_in_leaf` | LightGBM's minimum-data constraint for leaves. Larger values constrain tree growth. |
| `min_child_weight` | XGBoost's minimum sum of second-order gradient weights in a child; it is not simply a minimum row count. |
| L2 regularization | Penalty on leaf weights (`lambda_l2` in LightGBM, `lambda` in XGBoost) that constrains the fit. |
| Class weighting | Adjusts the influence of classes during training (`is_unbalance` / `scale_pos_weight`). It does not change evaluation labels or counts. |
| Hyperparameter | A setting chosen before fitting, such as learning rate, maximum depth, or requested rounds. |
| Overfitting | Learning training-specific detail that fails to generalize to new samples. More rounds or deeper trees are not automatically better. |

Reports distinguish **requested hyperparameters**, **actual fitted structure**
and **loaded runtime configuration**. Original training parameters are trusted
only when their metadata matches the model hash. In particular, XGBoost can load
runtime defaults that do not reproduce every original training parameter; those
are labeled separately. Per-tree depths and leaf counts are saved in JSON.

## TOML settings and execution

| Setting / term | Meaning |
| --- | --- |
| `runtime.threads` | CPU backend worker budget; default 10. `"auto"` resolves available CPU affinity. It does not make all Python feature extraction parallel. |
| `runtime.xgboost_device` | `auto` probes CUDA and falls back to CPU with a reason; `cuda` or `cuda:N` requires a usable visible GPU; `cpu` forces CPU. |
| CUDA | NVIDIA's GPU computing platform. A physical card, compatible library build/driver, and process/container GPU visibility are all necessary. |
| GPU utilization / VRAM | Compute activity / device memory. XGBoost manages the selected GPU without an artificial quota; constant 100% utilization is not guaranteed. |
| `training.rounds` | Requested boosting iterations (default 500), not neural-network epochs. The report records actual rounds too. |
| `training.seed` | Random seed for repeatability. It does not guarantee identical outputs across hardware/library versions or prove robustness across seeds. |
| `evaluation.threshold` | Score cutoff for classification (default 0.5). Does not affect how the model was originally fitted. |
| `evaluation.batch_size` | Records vectorized/predicted together during evaluation (default 1,024). Larger batches need more memory; this is not a training batch size. |
| `[[steps]]` | Ordered executable command arrays in TOML. `enabled = false` skips a step. Execution stops at the first failed command. |
| `--dry-run` | Prints enabled commands without running workflow steps or creating a run directory. |
| Cache / completion manifest | Reusable downloaded artifacts / hashes used to verify completed extraction before skipping work. A folder's mere existence does not prove completeness. |

## Data splits and reproducibility

**Training** fits the model. **Validation** should select settings or thresholds.
**Test** measures final performance on held-out records. **Challenge** measures
performance on the shared evasive-malware stress set. These roles must remain
separate; the current training scripts do not automatically create a validation
set or perform early stopping. They use only `*_train.jsonl` inputs.

Null and `-1` labels are excluded and counted; other non-binary labels cause an
error. The baselines use one seed and fixed hyperparameters, not an optimized
multi-seed benchmark. Equal tree counts do not make the two algorithms equally
complex or equally expensive.

Each batch saves reports under `data/runs/<timestamp>-<id>/reports/`, with
`run.json` recording settings, commands and status. Dataset and model hashes
identify the exact artifacts. Models/data remain in shared locations; the old
report's hash alone does not preserve an overwritten model. Compare report
versions and timestamps, especially after failed runs that can leave earlier
completed reports intact. A passing synthetic test suite verifies the workflow,
not real-dataset malware detection performance.
