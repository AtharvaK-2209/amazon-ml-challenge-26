"""
test_phase7_member2.py — Verification unit tests for Phase 7 Member 2.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("experiments/phase7/member2")


def test_deliverable_files_exist():
    """Verify that all required Phase 7 Member 2 deliverable files exist."""
    required_files = [
        OUT_DIR / "hard_negative_dataset.parquet",
        OUT_DIR / "error_analysis.csv",
        OUT_DIR / "blocking_recall.json",
        OUT_DIR / "predictions.parquet",
        OUT_DIR / "metrics.json",
        OUT_DIR / "experiment_report.md"
    ]
    for filepath in required_files:
        assert filepath.exists(), f"Missing required deliverable: {filepath}"
        assert filepath.stat().st_size > 0, f"File is empty: {filepath}"


def test_error_analysis_schema():
    """Verify error_analysis.csv schema and absence of NaN values."""
    err_df = pd.read_csv(OUT_DIR / "error_analysis.csv")
    required_cols = [
        "s1_entity_id", "candidate_entity_id", "candidate_source",
        "prediction_probability", "true_label", "predicted_label",
        "error_type", "error_category", "name_similarity", "address_similarity",
        "postal_match", "city_match", "state_match", "country_match",
        "house_number_match", "numeric_overlap"
    ]
    for col in required_cols:
        assert col in err_df.columns, f"Missing column {col} in error_analysis.csv"
    
    assert err_df.isnull().sum().sum() == 0, "error_analysis.csv contains NaN values"
    assert set(err_df["error_type"].unique()).issubset({"FALSE_POSITIVE", "FALSE_NEGATIVE"})


def test_blocking_recall_json():
    """Verify blocking_recall.json content and schema."""
    with open(OUT_DIR / "blocking_recall.json") as f:
        br = json.load(f)
    
    assert "total_true_matches" in br
    assert "true_matches_in_candidate_set" in br
    assert "true_matches_missing_from_candidate_set" in br
    assert "blocking_recall" in br
    assert 0.0 <= br["blocking_recall"] <= 1.0
    assert br["total_true_matches"] == br["true_matches_in_candidate_set"] + br["true_matches_missing_from_candidate_set"]


def test_metrics_json():
    """Verify metrics.json baseline vs experimental metrics."""
    with open(OUT_DIR / "metrics.json") as f:
        met = json.load(f)
    
    for key in ["baseline", "hard_negative_experiment", "experiment_metadata"]:
        assert key in met, f"Missing section {key} in metrics.json"
        
    base = met["baseline"]
    exp = met["hard_negative_experiment"]
    
    for m in ["precision", "recall", "f0_5", "f1", "fp", "fn"]:
        assert m in base, f"Missing baseline metric {m}"
        assert m in exp, f"Missing experimental metric {m}"
        assert not np.isnan(base[m]), f"NaN found in baseline {m}"
        assert not np.isnan(exp[m]), f"NaN found in experimental {m}"


def test_predictions_parquet_schema():
    """Verify predictions.parquet schema and integrity."""
    df_preds = pd.read_parquet(OUT_DIR / "predictions.parquet")
    required_cols = [
        "source1_entity_id", "candidate_entity_id",
        "baseline_prediction_probability", "experimental_prediction_probability",
        "baseline_predicted_label", "experimental_predicted_label",
        "true_label", "experiment_id", "model_version"
    ]
    for col in required_cols:
        assert col in df_preds.columns, f"Missing column {col} in predictions.parquet"
    
    assert len(df_preds) > 0, "predictions.parquet is empty"
    assert df_preds["baseline_prediction_probability"].isnull().sum() == 0
    assert df_preds["experimental_prediction_probability"].isnull().sum() == 0
