import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import List, Tuple, Optional, Union
from src.config import MODELS_DIR


class BaselinePredictor:
    """
    Reusable prediction logic for the baseline model.
    Does not perform threshold optimization or calibration.
    """
    
    def __init__(self, model_path: str = "models/xgb_baseline.json"):
        self.model_path = Path(model_path)
        self.model = xgb.XGBClassifier()
        
    def load_model(self):
        """Load the saved XGBoost model artifact."""
        self.model.load_model(self.model_path)
        
    def predict(self, feature_df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate positive class probabilities and binary predictions.
        
        Args:
            feature_df: DataFrame containing the pairwise features.
            feature_cols: The exact list of columns to feed the model.
            threshold: The default classification threshold.
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        X = feature_df[feature_cols].copy()
        X.fillna(-999, inplace=True)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return probs, preds


def predict_proba(features_df: pd.DataFrame,
                  model_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Run trained XGBoost model on a feature DataFrame.
    Returns array of shape (N,) with match probabilities.
    """
    path = Path(model_path) if model_path else (MODELS_DIR / "xgb_baseline.json")
    if not path.exists():
        path = MODELS_DIR / "model_v1.joblib"
    
    if path.suffix == ".json":
        clf = xgb.XGBClassifier()
        clf.load_model(path)
        meta_cols = ["source1_entity_id", "candidate_entity_id", "label", "candidate_source", "y_true"]
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        X = features_df[feature_cols].fillna(-999)
        return clf.predict_proba(X)[:, 1]
    else:
        clf = joblib.load(path)
        meta_cols = ["source1_entity_id", "candidate_entity_id", "label", "candidate_source", "y_true"]
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        return clf.predict_proba(features_df[feature_cols].values)[:, 1]


def predict_matches(candidates_df: pd.DataFrame,
                    features_df: pd.DataFrame,
                    threshold: float = 0.70,
                    model_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
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


if __name__ == "__main__":
    predictor = BaselinePredictor()
    try:
        predictor.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
