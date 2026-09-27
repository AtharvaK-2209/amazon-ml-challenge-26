# Phase 7 Member 3 Final Report

## Executive Summary
All controlled decision-rule experiments (E01 to E10) were executed on the frozen Phase 5/6 validation dataset (25,074 candidate pairs, 500 S1 entities, 1,646 true positive candidate pairs). 

The optimal decision configuration is:
- **`calibration_method`**: `raw`
- **`threshold` ($T$)**: `0.94` (or `0.935`)
- **`margin` ($M$)**: `0.00`
- **`precision`**: `1.000000` (Zero False Positives)
- **`recall`**: `0.974484`
- **`F0.5 Score`**: `0.994790`
- **`F1 Score`**: `0.987077`

---

## 1. E01 — Baseline Reproduction
- **Phase 6 Threshold**: `0.94`
- **Phase 6 Margin**: `0.00`
- **Precision**: `1.000000`
- **Recall**: `0.974484`
- **F0.5**: `0.994790`
- **Match Rate**: `92.8%` (464 S1 matched)
- **Status**: **`PASS`** (100% exact numerical reproduction of Phase 6 reference values).

---

## 2. E05 — Threshold Search
- **Search Range**: `0.800` to `0.990` (Resolution `0.005`)
- **Peak F0.5**: `0.994790` achieved at `T = 0.935` to `0.940`
- **Precision at Peak**: `1.000000` (`0` FP)
- **Recall at Peak**: `0.974484` (`1604` TP, `42` FN)

---

## 3. E06 — Margin Optimization
- **Search Range**: `0.00` to `0.50`
- **Selected Margin**: `0.00`
- **Observation**: Applying margin penalty $M > 0.0$ on multi-candidate top selections reduces recall without improving precision (precision is already 1.000000).

---

## 4. E07 — Singleton Analysis
- **True Singletons (0 GT Positives)**: `36 S1 entities`
- **Non-Singletons (>=1 GT Positives)**: `464 S1 entities`
- **Singleton Accuracy at T=0.94**: `100.0%` (All 36 true singletons fail threshold $<0.94$ and get correctly designated as NO_MATCH).

---

## 5. E08 — Candidate Count Analysis
- Candidate count ranges from 7 to 77 candidates per S1 entity.
- The standard decision rule ($T=0.94, M=0.00$) maintains 100% precision across all candidate count buckets.

---

## 6. E09 — S2/S3 Conflict Analysis
- **Conflict S1 Entities**: `500 / 500` (100% of S1s contain candidates from both S2 and S3).
- **S2 Top Candidates**: `270` (Accuracy `94.07%`)
- **S3 Top Candidates**: `230` (Accuracy `91.30%`)
- **Conclusion**: Pure model probability selection without artificial source bias achieves optimal F0.5.

---

## 7. E10 — Combined Decision Rules
- Tested combinations E10-A ($T=0.94, M=0.00$), E10-B ($T=0.935, M=0.00$), E10-C ($T=0.95, M=0.00$).
- Combination E10-A / E10-B yields optimal $F_0.5 = 0.994790$.

---

## 8. Experiment Comparison Table

| experiment_id       | experiment_name                |   threshold |   margin |   precision |   recall |     f05 |       f1 |   tp |   fp |   fn |   runtime_seconds | status    |
|:--------------------|:-------------------------------|------------:|---------:|------------:|---------:|--------:|---------:|-----:|-----:|-----:|------------------:|:----------|
| E01_baseline        | Phase 6 Baseline Reproduction  |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.0001 | PASS      |
| E05_threshold       | Fine Threshold Search          |       0.935 |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.009  | COMPLETED |
| E06_margin          | Margin Optimization            |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.4557 | COMPLETED |
| E07_singleton       | Singleton Analysis             |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.0043 | COMPLETED |
| E08_candidate_count | Candidate-Count-Aware Analysis |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.0061 | COMPLETED |
| E09_s2_s3           | S2/S3 Conflict Analysis        |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.0051 | COMPLETED |
| E10_combined        | Combined Decision Rules        |       0.94  |        0 |           1 | 0.974484 | 0.99479 | 0.987077 | 1604 |    0 |   42 |            0.001  | COMPLETED |

---

## 9. Selected Decision Configuration
- **`calibration_method`**: `"raw"`
- **`threshold` ($T$)**: `0.94`
- **`margin` ($M$)**: `0.00`
- **`singleton_rule`**: Reject if $P < 0.94$ (Yields 100% singleton accuracy)
- **`source_rule`**: Pure model probability rank order

---

## 10. Reproducibility
- **Git Commit**: `2c0cdb7`
- **Input Artifact**: `experiments/phase5/improved_validation_predictions.parquet`
- **Validation Rows**: `25074`
- **Unique S1 Entities**: `500`
