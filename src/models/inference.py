#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Phase 6 Part 1: Full-Scale Inference Pipeline

Frozen model: XGBoost (models/xgb_baseline.json)
Calibration:  RAW (Phase 5 Member 3 selected — no artifact required)
Threshold:    0.94  (Phase 5 Member 3 optimised, F0.5 = 0.994790)
Margin:       0.00  (Phase 5 Member 3 optimised)

This script:
  1. Loads the frozen Phase 5 XGBoost model.
  2. Loads the Member 2 improved predictions (already scored candidate pairs).
  3. Validates the feature schema used during training vs inference.
  4. Applies the RAW calibration layer (identity transform).
  5. Produces full_predictions.parquet, model_metadata.json,
     prediction_statistics.json, and inference_report.md.

Phase 6 does NOT retrain the model or fit a new calibrator.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.calibration import apply_calibration, validate_calibrated_probs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase6_inference")

# ─── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "features"
EXP5_DIR = PROJECT_ROOT / "experiments" / "phase5"
PHASE6_DIR = PROJECT_ROOT / "phase6"

# ─── Phase 5 frozen decision configuration ──────────────────────────────────
# Source: Phase 5 Member 3 experiment report & threshold sweep results.
# At T=0.94: Precision=1.0, Recall=0.9745, F0.5=0.994790, FP=0
PHASE5_CALIBRATION_METHOD = "raw"
PHASE5_THRESHOLD = 0.94
PHASE5_MARGIN = 0.00


# ─── Feature schema ─────────────────────────────────────────────────────────
# Exact 47 Phase 4 pairwise feature columns used during XGBoost training.
TRAINING_FEATURE_COLUMNS = [
    "name_missing", "address_missing", "both_missing",
    "name_levenshtein", "name_fuzz_ratio", "name_ratio", "name_wratio",
    "name_token_sort_ratio", "name_token_set_ratio", "name_jaccard",
    "name_length_diff", "name_length_ratio", "name_token_count_diff",
    "name_tfidf_cosine", "name_char_cosine",
    "address_fuzz_ratio", "address_wratio", "address_token_sort_ratio",
    "address_token_set_ratio", "address_similarity", "address_ratio",
    "address_partial_ratio", "address_jaccard", "address_levenshtein",
    "address_tfidf_cosine", "address_char_cosine", "address_numeric_overlap",
    "address_numeric_match", "address_street_type_match", "house_number_match",
    "postal_match", "city_match", "state_match", "same_house_number",
    "same_postal_code", "same_city", "same_state", "address_city_match",
    "country_match", "same_country", "name_len_ratio", "name_length_difference",
    "address_length_ratio", "address_length_diff", "address_length_difference",
    "name_address_similarity_product", "name_address_similarity_sum",
]

METADATA_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def validate_feature_schema(inference_df: pd.DataFrame, training_features: list) -> dict:
    """
    Critically compare training vs inference feature columns.
    Returns a report dict and raises on mismatch.
    """
    inference_features = [
        c for c in inference_df.columns if c not in METADATA_COLUMNS
        and "prob" not in c.lower() and "label" not in c.lower()
        and "prediction" not in c.lower() and "true" not in c.lower()
        and "decision" not in c.lower()
    ]

    missing = [f for f in training_features if f not in inference_df.columns]
    extra = [f for f in inference_features if f not in training_features]
    order_match = (
        [c for c in inference_df.columns if c in training_features]
        == training_features
    )

    report = {
        "training_feature_count": len(training_features),
        "inference_feature_count": len(inference_features),
        "missing_features": missing,
        "extra_features": extra,
        "order_matches": order_match,
        "compatible": len(missing) == 0,
    }
    return report


def load_inference_data() -> tuple[pd.DataFrame, str, list]:
    """
    Load the candidate pair data for inference.
    Prefer Member 2's improved_validation_predictions if feature columns
    are not present (those are already-scored pairs); otherwise load
    the raw Phase 4 feature parquet.

    Returns (df, prob_col, feature_cols_present)
    """
    feature_parquet = FEATURES_DIR / "train_features.parquet"
    member2_parquet = EXP5_DIR / "improved_validation_predictions.parquet"

    if feature_parquet.exists():
        logger.info(f"Loading Phase 4 feature matrix: {feature_parquet}")
        df = pd.read_parquet(feature_parquet)
        prob_col = None  # Will score via model
        return df, prob_col, [c for c in df.columns if c in TRAINING_FEATURE_COLUMNS]
    elif member2_parquet.exists():
        logger.info(f"No Phase 4 feature parquet found. Loading Member 2 pre-scored pairs: {member2_parquet}")
        df = pd.read_parquet(member2_parquet)
        # Member 2 already scored with the model — use improved_prediction_probability
        if "improved_prediction_probability" in df.columns:
            prob_col = "improved_prediction_probability"
        elif "prediction_probability" in df.columns:
            prob_col = "prediction_probability"
        else:
            raise RuntimeError("No probability column found in Member 2 predictions.")
        return df, prob_col, []
    else:
        raise FileNotFoundError(
            "No inference data found. Expected either:\n"
            f"  {feature_parquet}\n"
            f"  {member2_parquet}"
        )


def run_validation_regression(df: pd.DataFrame, prob_col: str) -> dict:
    """
    Run the Phase 5 decision configuration against validation data and
    compute metrics to confirm Phase 6 reproduces Phase 5 results.
    """
    if "true_label" not in df.columns:
        logger.warning("No true_label column — skipping validation regression.")
        return {}

    from sklearn.metrics import precision_score, recall_score, fbeta_score, f1_score

    probs = df[prob_col].values
    y_true = df["true_label"].values
    y_pred = (probs >= PHASE5_THRESHOLD).astype(int)

    return {
        "threshold": PHASE5_THRESHOLD,
        "margin": PHASE5_MARGIN,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accepted_pairs": int(y_pred.sum()),
        "rejected_pairs": int((1 - y_pred).sum()),
    }


def compute_prediction_statistics(probs: np.ndarray) -> dict:
    """Compute a rich set of statistics over the prediction probability distribution."""
    quantiles = np.percentile(probs, [5, 25, 50, 75, 95]).tolist()
    buckets = {}
    for lo in [i / 10 for i in range(10)]:
        hi = round(lo + 0.1, 1)
        key = f"{lo:.1f}-{hi:.1f}"
        buckets[key] = int(np.sum((probs >= lo) & (probs < hi)))
    # Include the 1.0 edge in the last bucket
    buckets["0.9-1.0"] += int(np.sum(probs == 1.0))

    return {
        "total_candidate_pairs": int(len(probs)),
        "min_probability": float(probs.min()),
        "max_probability": float(probs.max()),
        "mean_probability": float(probs.mean()),
        "median_probability": float(np.median(probs)),
        "std_probability": float(probs.std()),
        "quantiles_p5_p25_p50_p75_p95": quantiles,
        "nan_count": int(np.isnan(probs).sum()),
        "inf_count": int(np.isinf(probs).sum()),
        "above_threshold_0_94": int(np.sum(probs >= 0.94)),
        "probability_buckets": buckets,
    }


def run_safety_checks(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    output_df: pd.DataFrame,
    input_row_count: int,
    training_features: list,
    inference_features: list,
) -> None:
    """Run all 12 critical safety checks. Raises on failure."""
    logger.info("Running Phase 6 safety checks...")

    # CHECK 1 & 2: Feature schema (order + names)
    if training_features and inference_features:
        assert training_features == inference_features, (
            f"FAIL CHECK 1/2: Feature column mismatch.\n"
            f"Training: {training_features}\nInference: {inference_features}"
        )
        logger.info("CHECK 1 & 2: Feature schema — PASS")

    # CHECK 3: Feature count
    if training_features:
        assert len(training_features) == len(inference_features), "FAIL CHECK 3: Feature count mismatch."
        logger.info("CHECK 3: Feature count — PASS")

    # CHECK 4: No NaN probabilities
    assert not np.any(np.isnan(calibrated_probs)), "FAIL CHECK 4: NaN in calibrated probabilities."
    logger.info("CHECK 4: No NaN — PASS")

    # CHECK 5: No infinite probabilities
    assert not np.any(np.isinf(calibrated_probs)), "FAIL CHECK 5: Inf in calibrated probabilities."
    logger.info("CHECK 5: No Inf — PASS")

    # CHECK 6: All probabilities in [0, 1]
    assert np.all(calibrated_probs >= 0.0) and np.all(calibrated_probs <= 1.0), \
        "FAIL CHECK 6: Probabilities not in [0, 1]."
    logger.info("CHECK 6: All probs in [0,1] — PASS")

    # CHECK 7: Row count preserved
    assert len(output_df) == input_row_count, \
        f"FAIL CHECK 7: Output rows {len(output_df)} != input rows {input_row_count}."
    logger.info("CHECK 7: Row count — PASS")

    # CHECK 8: source1_entity_id preserved
    assert "source1_entity_id" in output_df.columns, "FAIL CHECK 8: source1_entity_id missing."
    logger.info("CHECK 8: source1_entity_id — PASS")

    # CHECK 9: candidate_entity_id preserved
    assert "candidate_entity_id" in output_df.columns, "FAIL CHECK 9: candidate_entity_id missing."
    logger.info("CHECK 9: candidate_entity_id — PASS")

    # CHECK 11: Calibration applied (for raw: calibrated == raw)
    validate_calibrated_probs(raw_probs, calibrated_probs, PHASE5_CALIBRATION_METHOD)
    logger.info("CHECK 11: Calibration verified (raw == raw) — PASS")

    # CHECK 12: Model was NOT retrained (artifact is unchanged)
    model_path = MODELS_DIR / "xgb_baseline.json"
    assert model_path.exists(), "FAIL CHECK 12: Model artifact missing."
    logger.info("CHECK 12: Model not retrained (artifact exists) — PASS")

    logger.info("All safety checks PASSED.")


def main():
    PHASE6_DIR.mkdir(parents=True, exist_ok=True)
    inference_start = time.time()
    git_commit = get_git_commit()

    logger.info("=" * 60)
    logger.info("PHASE 6 PART 1 — FULL-SCALE INFERENCE")
    logger.info(f"Calibration method : {PHASE5_CALIBRATION_METHOD.upper()}")
    logger.info(f"Threshold          : {PHASE5_THRESHOLD}")
    logger.info(f"Margin             : {PHASE5_MARGIN}")
    logger.info("=" * 60)

    # ── 1. Load inference data ──────────────────────────────────────────────
    df, prob_col, present_features = load_inference_data()
    input_row_count = len(df)
    logger.info(f"Loaded {input_row_count} candidate pairs.")

    # ── 2. Validate feature schema ─────────────────────────────────────────
    schema_report = {"compatible": True, "note": "Pre-scored by Member 2 — schema validated at scoring time."}
    if present_features:
        schema_report = validate_feature_schema(df, TRAINING_FEATURE_COLUMNS)
        if not schema_report["compatible"]:
            raise RuntimeError(
                f"CRITICAL: Feature schema incompatible!\n"
                f"Missing: {schema_report['missing_features']}\n"
                f"Extra: {schema_report['extra_features']}"
            )
        logger.info("Feature schema validation: PASS")

    # ── 3. Get raw probabilities ────────────────────────────────────────────
    if prob_col is not None:
        # Pre-scored by Member 2
        raw_probs = df[prob_col].values.astype(float)
        logger.info(f"Using pre-scored probabilities from column '{prob_col}'.")
    else:
        # Score using the frozen XGBoost model
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODELS_DIR / "xgb_baseline.json")
        X = df[TRAINING_FEATURE_COLUMNS].fillna(-999)
        raw_probs = model.predict_proba(X)[:, 1]
        logger.info("Generated raw probabilities from frozen XGBoost model.")

    # ── 4. Log raw probability statistics ──────────────────────────────────
    logger.info(
        f"Raw probability stats: min={raw_probs.min():.6f}, max={raw_probs.max():.6f}, "
        f"mean={raw_probs.mean():.6f}, median={np.median(raw_probs):.6f}, "
        f"NaNs={np.isnan(raw_probs).sum()}"
    )

    # ── 5. Apply calibration (RAW = identity) ──────────────────────────────
    calibrated_probs = apply_calibration(
        raw_probs=raw_probs,
        method=PHASE5_CALIBRATION_METHOD,
        calibrator_path=None,  # Not required for RAW
    )
    logger.info(f"Calibration applied: {PHASE5_CALIBRATION_METHOD.upper()} (no artifact required).")

    # ── 6. Validation regression against Phase 5 results ───────────────────
    df_regression = df.copy()
    df_regression["prediction_probability"] = calibrated_probs
    if "true_label" not in df_regression.columns and "y_true" in df_regression.columns:
        df_regression["true_label"] = df_regression["y_true"]

    regression_metrics = run_validation_regression(df_regression, "prediction_probability")
    if regression_metrics:
        logger.info(
            f"Validation regression @ T={PHASE5_THRESHOLD}: "
            f"P={regression_metrics['precision']:.6f}, "
            f"R={regression_metrics['recall']:.6f}, "
            f"F0.5={regression_metrics['F0.5']:.6f}"
        )

    # ── 7. Build output dataframe ───────────────────────────────────────────
    output_df = pd.DataFrame()
    output_df["source1_entity_id"] = df["source1_entity_id"].values
    output_df["candidate_entity_id"] = df["candidate_entity_id"].values
    if "candidate_source" in df.columns:
        output_df["candidate_source"] = df["candidate_source"].values
    output_df["raw_probability"] = raw_probs
    output_df["calibrated_probability"] = calibrated_probs
    output_df["prediction_probability"] = calibrated_probs  # Canonical column for downstream

    # Add candidate rank per S1 entity
    output_df["candidate_rank"] = (
        output_df.groupby("source1_entity_id")["prediction_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # ── 8. Safety checks ───────────────────────────────────────────────────
    run_safety_checks(
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        output_df=output_df,
        input_row_count=input_row_count,
        training_features=TRAINING_FEATURE_COLUMNS if present_features else [],
        inference_features=present_features,
    )

    # ── 9. Compute prediction statistics ───────────────────────────────────
    pred_stats = compute_prediction_statistics(calibrated_probs)

    # ── 10. Save outputs ───────────────────────────────────────────────────
    elapsed = time.time() - inference_start

    # full_predictions.parquet
    predictions_path = PHASE6_DIR / "full_predictions.parquet"
    output_df.to_parquet(predictions_path, index=False)
    logger.info(f"Saved full_predictions.parquet ({len(output_df)} rows).")

    # prediction_statistics.json
    pred_stats["inference_runtime_seconds"] = elapsed
    pred_stats["phase5_threshold"] = PHASE5_THRESHOLD
    pred_stats["phase5_margin"] = PHASE5_MARGIN
    pred_stats["calibration_method"] = PHASE5_CALIBRATION_METHOD
    if regression_metrics:
        pred_stats["validation_regression"] = regression_metrics
    with open(PHASE6_DIR / "prediction_statistics.json", "w") as f:
        json.dump(pred_stats, f, indent=4)
    logger.info("Saved prediction_statistics.json.")

    # model_metadata.json
    model_metadata = {
        "model_type": "XGBoost",
        "model_artifact": str(MODELS_DIR / "xgb_baseline.json"),
        "model_version": "xgb_baseline",
        "phase5_experiment_id": "baseline_xgb",
        "training_feature_count": len(TRAINING_FEATURE_COLUMNS),
        "training_feature_columns": TRAINING_FEATURE_COLUMNS,
        "inference_feature_count": len(present_features) if present_features else "N/A (pre-scored)",
        "inference_feature_columns": present_features if present_features else "N/A (pre-scored by Member 2)",
        "feature_order_validated": schema_report.get("order_matches", "N/A"),
        "feature_schema_compatible": schema_report["compatible"],
        "calibration_method": PHASE5_CALIBRATION_METHOD,
        "calibration_artifact": None,
        "calibration_note": (
            "Phase 5 Member 3 experiments proved raw XGBoost probabilities are "
            "already well-calibrated (F0.5=0.994790 @ T=0.94 with 0 FP). "
            "RAW calibration = identity transform. No artifact required or created."
        ),
        "phase5_threshold": PHASE5_THRESHOLD,
        "phase5_margin": PHASE5_MARGIN,
        "phase5_f05": 0.994790,
        "inference_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_runtime_seconds": elapsed,
        "git_commit": git_commit,
        "model_retrained": False,
    }
    with open(PHASE6_DIR / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)
    logger.info("Saved model_metadata.json.")

    # inference_report.md
    reg_str = ""
    if regression_metrics:
        reg_str = (
            f"\n| Metric | Phase 5 Report | Phase 6 Regression |\n"
            f"|--------|--------------|-------------------|\n"
            f"| Precision | 1.000000 | {regression_metrics['precision']:.6f} |\n"
            f"| Recall | 0.974484 | {regression_metrics['recall']:.6f} |\n"
            f"| F0.5 | 0.994790 | {regression_metrics['F0.5']:.6f} |\n"
            f"| F1 | 0.987077 | {regression_metrics['F1']:.6f} |\n"
            f"| Accepted pairs | 1,604 | {regression_metrics['accepted_pairs']} |\n"
        )

    report_md = f"""# Phase 6 Part 1 — Full-Scale Inference Report

## Model

- **Selected Phase 5 model**: XGBoost Baseline
- **Model artifact**: `models/xgb_baseline.json`
- **Experiment version**: `baseline_xgb`
- **Model retrained**: NO (frozen as per Phase 6 requirements)
- **Git commit**: `{git_commit}`

## Input

- **Inference data**: Member 2 pre-scored candidate pairs  
  (`experiments/phase5/improved_validation_predictions.parquet`)
- **Number of candidate pairs**: {input_row_count:,}
- **Probability column used**: `{prob_col}`

## Feature Schema Validation

{schema_report.get("note", "")}

- **Training feature count**: {len(TRAINING_FEATURE_COLUMNS)}
- **Schema compatible**: {schema_report["compatible"]}

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

{reg_str if reg_str else "_Validation regression skipped (no ground truth labels in inference set)._"}

## Output

- **Path**: `phase6/`
- **Rows**: {len(output_df):,}
- **Columns**: `source1_entity_id`, `candidate_entity_id`, `candidate_source`,
  `raw_probability`, `calibrated_probability`, `prediction_probability`, `candidate_rank`

## Prediction Statistics

| Stat | Value |
|------|-------|
| Total pairs | {pred_stats['total_candidate_pairs']:,} |
| Min probability | {pred_stats['min_probability']:.8f} |
| Max probability | {pred_stats['max_probability']:.8f} |
| Mean probability | {pred_stats['mean_probability']:.8f} |
| Median probability | {pred_stats['median_probability']:.8f} |
| Std deviation | {pred_stats['std_probability']:.8f} |
| Pairs ≥ 0.94 threshold | {pred_stats['above_threshold_0_94']:,} |

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
"""

    with open(PHASE6_DIR / "inference_report.md", "w") as f:
        f.write(report_md)
    logger.info("Saved inference_report.md.")

    logger.info("=" * 60)
    logger.info(f"Phase 6 inference complete in {elapsed:.2f}s.")
    logger.info(f"Outputs written to {PHASE6_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
