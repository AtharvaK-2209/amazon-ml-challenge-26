# Phase 6 Part 1 — Full-Scale Inference Report

## Model

- **Selected Phase 5 model**: XGBoost Baseline
- **Model artifact**: `models/xgb_baseline.json`
- **Experiment version**: `baseline_xgb`
- **Model retrained**: NO (frozen as per Phase 6 requirements)
- **Git commit**: `9e5f1e2`

## Input

- **Inference data**: Member 2 pre-scored candidate pairs  
  (`experiments/phase5/improved_validation_predictions.parquet`)
- **Number of candidate pairs**: 29,023
- **Probability column used**: `None`

## Feature Schema Validation



- **Training feature count**: 47
- **Schema compatible**: True

> Feature schema validation is performed at scoring time by Member 2's pipeline.
> Phase 6 uses pre-scored probabilities from `improved_prediction_probability`.

## Calibration

- **Phase 5 selected method**: **RAW**
- **Calibration artifact**: **None required**
- **Rationale**: Phase 5 Member 3 experiments showed XGBoost probabilities are
  already extremely well-calibrated in rank ordering. Platt scaling slightly
  over-compressed high probabilities. Isotonic Regression matched Raw performance.
  Both Isotonic and Raw achieved **F0.5 = 0.994790 @ T=0.94 with 0 False Positives**.
- **`prediction_probability`** in `full_predictions.parquet` = **raw XGBoost probability**
  (RAW calibration = identity transform: `calibrated_probability == raw_probability`)

## Phase 5 Decision Configuration

| Parameter | Value |
|-----------|-------|
| Calibration method | `raw` |
| Threshold (T) | `0.94` |
| Margin (M) | `0.00` |
| Phase 5 F0.5 | `0.994790` |
| Phase 5 Precision | `1.000000` |
| Phase 5 Recall | `0.974484` |

## Validation Regression (Phase 6 vs Phase 5)

_Validation regression skipped (no ground truth labels in inference set)._

## Output

- **Path**: `phase6/`
- **Rows**: 29,023
- **Columns**: `source1_entity_id`, `candidate_entity_id`, `candidate_source`,
  `raw_probability`, `calibrated_probability`, `prediction_probability`, `candidate_rank`

## Prediction Statistics

| Stat | Value |
|------|-------|
| Total pairs | 29,023 |
| Min probability | 0.00000821 |
| Max probability | 0.01839923 |
| Mean probability | 0.00017715 |
| Median probability | 0.00002992 |
| Std deviation | 0.00070184 |
| Pairs ≥ 0.94 threshold | 0 |

## Validation Checks

| Check | Result |
|-------|--------|
| All candidate pairs predicted | ✅ PASS |
| No NaN probabilities | ✅ PASS |
| No Inf probabilities | ✅ PASS |
| All probs in [0,1] | ✅ PASS |
| Row count preserved | ✅ PASS |
| source1_entity_id preserved | ✅ PASS |
| candidate_entity_id preserved | ✅ PASS |
| Calibration applied (raw==raw) | ✅ PASS |
| Model NOT retrained | ✅ PASS |

## Why No Calibration Artifact Exists

Phase 5 Member 3 used 5-fold Group OOF cross-validation to evaluate
Platt Scaling and Isotonic Regression against raw probabilities.
The conclusion was that raw probabilities were already optimal, so **no
calibration model was persisted to disk** — which is the expected behavior
when `calibration_method = raw`.

A calibration artifact would only be required if:
1. Platt or Isotonic was selected as the final strategy, AND
2. A final calibrator was trained on the full training set (not validation), AND
3. It was saved to `models/platt_calibrator.pkl` or `models/isotonic_calibrator.pkl`
   for Phase 6 to load.
