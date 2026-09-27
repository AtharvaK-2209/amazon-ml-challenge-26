"""predict.py — Phase 5/6: Run inference with a trained model."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.config import MODELS_DIR


def predict_proba(features_df: pd.DataFrame,
                  model_path: str | Path | None = None) -> np.ndarray:
    """
    Run trained XGBoost model on a feature DataFrame.
    Returns array of shape (N,) with match probabilities.
    """
    path = model_path or (MODELS_DIR / "model_v1.joblib")
    clf  = joblib.load(path)
    feature_cols = [c for c in features_df.columns
                    if c not in ("source1_entity_id","candidate_entity_id","label")]
    return clf.predict_proba(features_df[feature_cols].values)[:, 1]


def predict_matches(candidates_df: pd.DataFrame,
                    features_df: pd.DataFrame,
                    threshold: float = 0.70,
                    model_path: str | Path | None = None) -> pd.DataFrame:
    """
    Full prediction pipeline:
    1. Score all candidate pairs
    2. Apply threshold
    3. Return matching_results DataFrame: [source1_entity_id, matched_entity_ids]
    """
    scores = predict_proba(features_df, model_path)
    candidates_df = candidates_df.copy()
    candidates_df["match_score"] = scores

    matched = candidates_df[candidates_df["match_score"] >= threshold]

    results = (
        matched
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(ids.tolist()))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )
    return results
