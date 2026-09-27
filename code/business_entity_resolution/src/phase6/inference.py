"""
inference.py — Phase 6 Member 3: Model Inference Component.

Wraps Member 1's model prediction logic to generate raw probabilities on features dataframe.
"""

import pathlib
import sys
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_proba, BaselinePredictor


def run_phase6_inference(
    features_df: pd.DataFrame,
    model_path: str | pathlib.Path = "models/xgb_baseline.json"
) -> pd.DataFrame:
    """
    Run inference on input features DataFrame using Member 1's trained model.
    
    Returns DataFrame with:
    - source1_entity_id
    - candidate_entity_id
    - candidate_source
    - raw_probability
    """
    model_path = pathlib.Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    # Compute prediction probabilities
    probs = predict_proba(features_df, model_path=model_path)
    
    preds_df = features_df[["source1_entity_id", "candidate_entity_id"]].copy()
    
    if "candidate_source" in features_df.columns:
        preds_df["candidate_source"] = features_df["candidate_source"]
    else:
        preds_df["candidate_source"] = preds_df["candidate_entity_id"].apply(
            lambda c: "S2" if str(c).startswith("S2") else ("S3" if str(c).startswith("S3") else "UNKNOWN")
        )
        
    preds_df["raw_probability"] = probs
    
    if "y_true" in features_df.columns:
        preds_df["true_label"] = features_df["y_true"]
    elif "true_label" in features_df.columns:
        preds_df["true_label"] = features_df["true_label"]
        
    return preds_df
