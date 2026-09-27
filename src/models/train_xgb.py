"""train_xgb.py — Phase 5: XGBoost pairwise matching model with hard negative mining."""
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
import joblib
from src.config import MODEL, MODELS_DIR


def train(features_path: str | Path,
          model_save_path: str | Path | None = None,
          use_hard_negatives: bool = True) -> XGBClassifier:
    """
    Train XGBoost on pairwise features.
    features_path: parquet with columns [source1_entity_id, label, ...features...]
    """
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns
                    if c not in ("source1_entity_id","candidate_entity_id","label")]

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["source1_entity_id"].values

    # 80/20 split by S1 entity (not arbitrary pair split)
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=MODEL["validation_split"],
                            random_state=MODEL["random_seed"])
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    if use_hard_negatives:
        X_train, y_train = _mine_hard_negatives(X_train, y_train,
                                                df.iloc[train_idx], feature_cols)

    clf = XGBClassifier(**MODEL["xgb_params"])
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50)

    save_path = model_save_path or (MODELS_DIR / "model_v1.joblib")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"Model saved → {save_path}")
    return clf


def _mine_hard_negatives(X: np.ndarray, y: np.ndarray,
                          df_subset: pd.DataFrame,
                          feature_cols: list) -> tuple:
    """
    Hard negative mining: oversample negatives with high name_ratio scores.
    Mimics confusing near-matches (same name, different city).
    """
    pos_mask = y == 1
    neg_mask = y == 0

    if "name_ratio" in feature_cols:
        name_ratio_col = feature_cols.index("name_ratio")
        hard_neg_mask = neg_mask & (X[:, name_ratio_col] > 0.7)
    else:
        hard_neg_mask = neg_mask

    hard_neg_idx = np.where(hard_neg_mask)[0]
    pos_idx      = np.where(pos_mask)[0]

    # Oversample hard negatives to 2× positives
    if len(hard_neg_idx) > 0:
        oversample_n = min(len(pos_idx) * 2, len(hard_neg_idx))
        chosen = np.random.choice(hard_neg_idx, oversample_n, replace=False)
        keep   = np.concatenate([pos_idx,
                                  np.where(~hard_neg_mask & neg_mask)[0],
                                  chosen])
        keep   = np.sort(keep)
        return X[keep], y[keep]

    return X, y
