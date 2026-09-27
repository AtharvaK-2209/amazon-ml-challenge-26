"""
calibration.py — Phase 6 Member 3: Calibration Component.

Passes through or applies probability calibration.
Phase 5 selected method = 'raw' (calibrated_probability = raw_probability).
"""

import pandas as pd
import numpy as np
from typing import Optional, Any


def apply_calibration(
    preds_df: pd.DataFrame,
    method: str = "raw",
    calibrator_artifact: Optional[Any] = None
) -> pd.DataFrame:
    """
    Apply probability calibration layer.
    
    For method == 'raw':
    calibrated_probability = raw_probability
    
    Returns DataFrame with added column: 'calibrated_probability' & 'prediction_probability'.
    """
    df = preds_df.copy()
    
    if "raw_probability" not in df.columns:
        if "prediction_probability" in df.columns:
            df["raw_probability"] = df["prediction_probability"]
        else:
            raise KeyError("raw_probability column missing from predictions DataFrame.")

    if method.lower() == "raw":
        df["calibrated_probability"] = df["raw_probability"]
    elif calibrator_artifact is not None:
        probs = df["raw_probability"].values.reshape(-1, 1)
        df["calibrated_probability"] = calibrator_artifact.predict(probs)
    else:
        raise ValueError(f"Calibration method '{method}' requested but no calibrator artifact provided.")

    df["prediction_probability"] = df["calibrated_probability"]
    return df
