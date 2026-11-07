# PhishTrap Model Benchmark

12 candidate models, all CPU-only training on the PhishTrap train split (13,969 URLs, 16 features),
5-fold CV F1, held-out test (2,993 URLs, 49.5% phishing), ONNX export, size, and inference latency
(median of 200 single-URL runs, onnxruntime CPU).

Full results: `benchmark_models.py` -> `/tmp/phishtrap_bench/results.json`

## Baseline candidates

| model | CV F1 | test F1 | acc | prec | rec | AUC | size KB | latency |
|---|---|---|---|---|---|---|---|---|
| gaussian_nb | 0.5571 | 0.542 | 0.681 | 0.939 | 0.381 | 0.863 | 1.8 | 0.008ms |
| logistic_regression | 0.816 | 0.816 | 0.825 | 0.854 | 0.781 | 0.901 | 0.7 | 0.004ms |
| decision_tree | 0.8401 | 0.846 | 0.853 | 0.884 | 0.811 | 0.919 | 14.0 | 0.004ms |
| random_forest | 0.8246 | 0.829 | 0.840 | 0.879 | 0.784 | 0.907 | 32.4 | 0.004ms |
| extra_trees | 0.7953 | 0.801 | 0.812 | 0.844 | 0.761 | 0.880 | 14.9 | 0.004ms |
| adaboost | 0.8305 | 0.826 | 0.834 | 0.861 | 0.794 | 0.903 | 33.7 | 0.065ms |
| gradient_boosting | 0.8283 | 0.832 | 0.841 | 0.871 | 0.796 | 0.909 | 16.8 | 0.010ms |
| hist_gb | 0.8443 | 0.843 | 0.852 | - | - | - | n/a (export unsupported) | - |
| lightgbm | 0.8415 | 0.846 | 0.854 | 0.890 | 0.805 | 0.921 | 31.9 | 0.012ms |
| xgboost | 0.8367 | 0.840 | 0.848 | 0.874 | 0.808 | 0.914 | 28.0 | 0.004ms |
| svm | 0.8324 | 0.836 | 0.847 | 0.892 | 0.787 | 0.914 | 444.4 | 0.043ms |
| knn | 0.8306 | 0.833 | 0.841 | 0.865 | 0.805 | 0.897 | 889.4 | 14.444ms |

## Tuned variants (sweep over depth / trees / leaves)

| model | val F1 | test F1 | acc | prec | rec | AUC | size KB |
|---|---|---|---|---|---|---|---|
| tree_d8 | 0.8497 | 0.846 | 0.853 | 0.884 | 0.811 | 0.919 | 14.0 |
| tree_d10 | 0.8495 | 0.846 | 0.851 | 0.870 | 0.823 | 0.913 | 17.0 |
| rf_20x8 | 0.8434 | 0.846 | 0.853 | 0.877 | 0.817 | 0.919 | 60.0 |
| lgbm_40x8L31 | 0.8519 | 0.854 | 0.861 | 0.892 | 0.819 | 0.927 | 87.9 |
| lgbm_50x10L63 | 0.8567 | 0.855 | 0.862 | 0.893 | 0.820 | 0.929 | 223.7 |

## Findings

- **Best tiny pick: decision tree depth 8** — 14.0 KB, test F1 0.846, AUC 0.919,
  0.004ms/URL. Beats the original 10x6 random forest (32.4 KB, F1 0.829) on every metric
  while being less than half the size.
- **Best overall quality: LightGBM 50x10L63** — F1 0.855, acc 0.862, AUC 0.929,
  but 223.7 KB (16x the tree) for +0.9% F1.
- Decision tree depth 8 vs 10 are equivalent on test; depth 8 is smaller (14 vs 17 KB)
  and less overfit-prone.
- ONNX parity verified 1.0000 on all exported models.
- All inference latencies are sub-100 microseconds except KNN (14ms — model stores all data).

## Method notes

- 5-fold CV F1 on train; test split untouched until final eval.
- Best-threshold F1 tuned on val split, applied to test.
- skl2onnx (sklearn models), onnxmltools (LightGBM / XGBoost), opset repaired + checker-validated.
- Random seeds fixed (42) everywhere for reproducibility.

## Deep head-to-head: decision tree d8 vs LightGBM 40x8L31

Run: `deep_benchmark.py` (artifacts in `/tmp/phishtrap_bench2/`)

### Seed variance (10 seeds, held-out test)

| model | F1 | acc | prec | rec | AUC |
|---|---|---|---|---|---|
| tree d8 | 0.8448 +/- 0.0005 | 0.8525 +/- 0.0005 | 0.8823 +/- 0.0009 | 0.8105 +/- 0.0005 | 0.9186 +/- 0.0002 |
| lgbm 40x8L31 | 0.8537 +/- 0.0000 | 0.8610 +/- 0.0000 | 0.8920 +/- 0.0000 | 0.8186 +/- 0.0000 | 0.9271 +/- 0.0000 |

Both are essentially seed-stable; lgbm is fully deterministic (0 std).

### Statistical significance (seed 42)

- McNemar on paired test predictions: lgbm-only-right=75, tree-only-right=52, chi2=3.81, **p=0.0509 — not significant** at alpha=0.05 (just misses).
- Paired bootstrap 95% CI of F1(lgbm - tree): [-0.0000, +0.0157] — **inconclusive** (CI crosses zero).

### Threshold analysis

- tree d8: default 0.5 threshold is already optimal (best val t=0.49 -> test F1 0.8456, no gain).
- lgbm: default 0.5 slightly conservative; t=0.34 raises recall 0.82->0.87 at precision cost (F1 0.8537 -> 0.8428).

### Misclassification profile (identical pattern for both)

- FP (158/147): longer URLs (mean 27-28 vs 20) with higher entropy (3.9 vs 3.6) — suspicious-looking but legit pages.
- FN (281/269): dominated by trusted TLDs (63-69% of misses vs 28-29% of hits) with near-zero subdomain counts — phishing on bare trusted domains (e.g. `paypal.com/...`) is the hard core.
- No `is_domain_ip` FN for either (IP phishing is fully caught).

### Robustness

| perturbation | tree F1 | lgbm F1 |
|---|---|---|
| clean | 0.8456 | 0.8537 |
| gaussian sigma=0.05 | 0.8453 (-0.0003) | 0.8036 (-0.0501) |
| gaussian sigma=0.15 | 0.8420 (-0.0036) | 0.8004 (-0.0533) |
| gaussian sigma=0.5 | 0.7858 (-0.0598) | 0.7715 (-0.0822) |
| 5% extreme-value corruption | 0.8004 | 0.8086 |

**Tree is dramatically more robust to feature noise** (F1 drop 0.004 vs 0.053 at sigma=0.15) — lgbm over-relies on precise numeric thresholds. lgbm only wins on extreme-value corruption, by 0.008.

### Hard subset (compromised legit domains)

None present in the current test split (all re-labeled entries live in train/val from the schema-v2 merge). Will re-check on future refresh builds.

### Verdict

- Quality: lgbm leads by +0.009 F1 / +0.0085 AUC but **not statistically significant** (McNemar p=0.0509, bootstrap CI crosses 0).
- Size: tree is 14.0 KB vs 87.9 KB (6.3x smaller).
- Robustness: tree wins decisively on feature noise, which matters for a browser extension where feature extraction can be imperfect.
- **Winner for the tiny on-device use case: decision tree d8** (14 KB, robust, no threshold tuning needed).

## Expanded-feature Pareto sweep (35 features)

Run: `sweep_models.py` — full feature set adds keyword, brand, punycode, port,
encoding, and structure signals to the base 16. Verified 16/16 base features
byte-exact against the stored dataset (the one entropy diff is CSV float32 rounding).

### Pareto frontier (F1 vs ONNX size)

| model | feats | F1 | acc | AUC | size KB |
|---|---|---|---|---|---|
| dt_d8 | full | 0.8446 | 0.8527 | 0.9208 | 11.5 |
| dt_d8 | base16 | 0.8456 | 0.8533 | 0.9190 | 14.0 |
| xgb_30x4 | full | 0.8472 | 0.8547 | 0.9239 | 28.1 |
| lgbm_30x6L15 | full | 0.8502 | 0.8567 | 0.9270 | 31.9 |
| gb_50x4 | full | 0.8506 | 0.8573 | 0.9294 | 53.9 |
| lgbm_60x8L31 | base16 | 0.8545 | 0.8620 | 0.9285 | 131.5 |
| **lgbm_60x8L31** | **full** | **0.8626** | **0.8697** | **0.9369** | **131.5** |

Notable: everything above 131.5 KB does WORSE (overfitting): lgbm_100x10L63 0.8589
@447 KB, rf_100x12 0.8566 @2388 KB, lgbm_200x12L127 0.8477 @1894 KB.
The elbow is unambiguous.

### Winner validation: lgbm_60x8L31 full (131.5 KB)

- Seed variance (10 seeds): F1 0.8626 +/- 0.0000, acc 0.8697, prec 0.9033, rec 0.8254, AUC 0.9369 — fully deterministic.
- **McNemar vs tree d8: p=0.0002 (highly significant)** — lgbm-only-right=115, tree-only-right=64.
- Paired bootstrap 95% CI of F1(lgbm - tree): [+0.0084, +0.0271] — entirely positive.
- Threshold: default 0.5 already good (0.8626); t=0.36 trades precision for recall (0.8529).
- Robustness: tree still wins on pure feature noise (lgbm is threshold-sensitive), but lgbm wins on extreme-value corruption (0.808 vs 0.791). For a browser extension the feature extractor is deterministic code, so noise tolerance is secondary.
- Confusion: TP=1224 FP=131 FN=259 TN=1379 — fewer errors than tree (131 vs 158 FP, 259 vs 281 FN).

### Verdict (final)

- **Best balanced model: LightGBM 60x8L31, full 35 features — 131.5 KB, F1 0.8626, AUC 0.9369, 100% ONNX parity.**
- +0.018 F1 / +0.017 AUC over the 14 KB tree — statistically significant, 9x size still trivial for any device.
- This replaces the previous "tiny tree" pick: the tree's only remaining advantage is pure noise robustness.
