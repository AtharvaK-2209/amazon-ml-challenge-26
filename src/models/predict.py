import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Tuple

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
        # Ensure NaNs are safely handled before prediction
        X.fillna(-999, inplace=True)
        
        # Generate positive class probability
        probs = self.model.predict_proba(X)[:, 1]
        
        # Generate binary predictions using the default threshold
        preds = (probs >= threshold).astype(int)
        
        return probs, preds

if __name__ == "__main__":
    import numpy as np
    # Quick ad-hoc test
    predictor = BaselinePredictor()
    try:
        predictor.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
