import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import time
from pathlib import Path
import warnings

# Define paths
FEATURES_DIR = Path("features")
MODELS_DIR = Path("models")
EXP_DIR = Path("experiments/phase5")

MODELS_DIR.mkdir(exist_ok=True, parents=True)
EXP_DIR.mkdir(exist_ok=True, parents=True)

def train_and_evaluate():
    print("Loading data...")
    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val_df = pd.read_parquet(FEATURES_DIR / "validation_features.parquet")
    
    # Identify metadata & target
    metadata_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
    target_col = "y_true"
    
    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in metadata_cols + [target_col]]
    
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Number of feature columns: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Extract data
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()
    
    # Handle NaN values explicitly
    X_train.fillna(-999, inplace=True)
    X_val.fillna(-999, inplace=True)
    
    # Target Distribution
    positives = y_train.sum()
    negatives = len(y_train) - positives
    print(f"\n--- Target Distribution (Train) ---")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"Positive %: {positives/len(y_train)*100:.2f}%")
    print(f"Negative %: {negatives/len(y_train)*100:.2f}%")
    if positives > 0:
        print(f"Imbalance Ratio: 1:{negatives/positives:.2f}")
    
    # ==========================
    # XGBoost Baseline
    # ==========================
    print("\nTraining XGBoost baseline...")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_model.save_model(MODELS_DIR / "xgb_baseline.json")
    
    # Evaluate XGBoost
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    
    # XGBoost Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df['importance_rank'] = range(1, len(feature_cols) + 1)
    importance_df.to_csv(EXP_DIR / "feature_importance.csv", index=False)
    
    # Metrics
    xgb_metrics = {
        "model": "XGBoost",
        "parameters": xgb_model.get_params(),
        "features_used": feature_cols,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "precision": float(precision_score(y_val, xgb_preds, zero_division=0)),
        "recall": float(recall_score(y_val, xgb_preds, zero_division=0)),
        "F0.5": float(fbeta_score(y_val, xgb_preds, beta=0.5, zero_division=0)),
        "F1": float(f1_score(y_val, xgb_preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_val, xgb_probs)),
        "PR-AUC": float(average_precision_score(y_val, xgb_probs)),
        "runtime": xgb_time,
        "git_commit": "HEAD"  # Placeholder
    }
    with open(EXP_DIR / "baseline_xgb.json", "w") as f:
        json.dump(xgb_metrics, f, indent=4)
        
    print(f"XGBoost F0.5: {xgb_metrics['F0.5']:.4f}")

    # ==========================
    # LightGBM Baseline
    # ==========================
    print("\nTraining LightGBM baseline...")
    start_time = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        force_row_wise=True
    )
    # LGBM likes feature names to have no special characters, but our simple features are fine.
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start_time
    
    lgb_model.booster_.save_model(MODELS_DIR / "lgbm_baseline.txt")
    
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    lgb_preds = (lgb_probs >= 0.5).astype(int)
    
    lgb_metrics = {
        "model": "LightGBM",
        "parameters": lgb_model.get_params(),
        "features_used": feature_cols,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "precision": float(precision_score(y_val, lgb_preds, zero_division=0)),
        "recall": float(recall_score(y_val, lgb_preds, zero_division=0)),
        "F0.5": float(fbeta_score(y_val, lgb_preds, beta=0.5, zero_division=0)),
        "F1": float(f1_score(y_val, lgb_preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_val, lgb_probs)),
        "PR-AUC": float(average_precision_score(y_val, lgb_probs)),
        "runtime": lgb_time,
        "git_commit": "HEAD"
    }
    with open(EXP_DIR / "baseline_lgbm.json", "w") as f:
        json.dump(lgb_metrics, f, indent=4)
        
    print(f"LightGBM F0.5: {lgb_metrics['F0.5']:.4f}")
    
    # ==========================
    # Generate Validation Output
    # ==========================
    # We will use XGBoost as the primary model to generate the main prediction & error files
    val_out = val_df[metadata_cols + [target_col]].copy()
    val_out['prediction_probability'] = xgb_probs
    val_out['prediction'] = xgb_preds
    val_out['model'] = 'XGBoost'
    
    val_out.to_parquet(EXP_DIR / "validation_predictions.parquet", index=False)
    
    # Error dataset
    errors = val_out[val_out['y_true'] != val_out['prediction']].copy()
    def get_error_type(row):
        if row['y_true'] == 0 and row['prediction'] == 1:
            return 'FALSE_POSITIVE'
        elif row['y_true'] == 1 and row['prediction'] == 0:
            return 'FALSE_NEGATIVE'
        return 'UNKNOWN'
    
    errors['error_type'] = errors.apply(get_error_type, axis=1)
    errors.to_csv(EXP_DIR / "errors.csv", index=False)
    print(f"\nGenerated {len(errors)} errors for analysis.")
    print("Done!")

if __name__ == "__main__":
    train_and_evaluate()
