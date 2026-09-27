"""evaluate_f05.py — F_0.5 macro-averaged evaluation (matches Amazon's scoring)."""
import pandas as pd
import numpy as np


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def compute_pair_metrics(y_true, y_pred, beta: float = 0.5) -> dict:
    """
    Compute pair-level classification metrics (Precision, Recall, F0.5, F1, TP, FP, FN, TN).
    
    y_true: array-like of true binary labels (0 or 1)
    y_pred: array-like of predicted binary labels (0 or 1)
    beta: F-beta weight parameter (default: 0.5 for precision priority)
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f05       = float(f_beta(precision, recall, beta=beta))
    f1        = float(f_beta(precision, recall, beta=1.0))
    
    return {
        "precision": round(precision, 6),
        "recall":    round(recall, 6),
        "f05":       round(f05, 6),
        "f1":        round(f1, 6),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
        "total":     len(y_t)
    }


def evaluate_f05(predictions: pd.DataFrame,
                 ground_truth: pd.DataFrame,
                 beta: float = 0.5) -> dict:
    """
    Compute macro-averaged F_0.5, Precision, Recall at the entity level.

    predictions:  [source1_entity_id, matched_entity_ids]  (comma-sep string or list)
    ground_truth: [source1_entity_id, matched_entity_ids]  (comma-sep string or list)

    Returns dict: {f05, precision, recall, false_merges, singleton_accuracy, n_entities}
    """
    def parse_ids(s):
        if pd.isna(s) or str(s).strip() == "":
            return set()
        if isinstance(s, (list, set, tuple)):
            return set(s)
        return set(str(s).split(","))

    gt_dict   = {r.source1_entity_id: parse_ids(r.matched_entity_ids)
                 for _, r in ground_truth.iterrows()}
    pred_dict = {r.source1_entity_id: parse_ids(r.matched_entity_ids)
                 for _, r in predictions.iterrows()}

    entity_ids = set(gt_dict.keys()).union(set(pred_dict.keys()))
    precisions, recalls, f05s = [], [], []
    false_merges = 0
    singleton_correct = 0
    singleton_total = 0

    for eid in entity_ids:
        true_set = gt_dict.get(eid, set())
        pred_set = pred_dict.get(eid, set())

        # Singleton tracking
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            else:
                false_merges += len(pred_set)

        tp = len(true_set & pred_set)
        p  = tp / len(pred_set) if pred_set else (1.0 if not true_set else 0.0)
        r  = tp / len(true_set) if true_set else (1.0 if not pred_set else 0.0)
        f  = f_beta(p, r, beta)

        precisions.append(p)
        recalls.append(r)
        f05s.append(f)

    return {
        "f05":                round(float(np.mean(f05s)) if f05s else 0.0, 6),
        "precision":          round(float(np.mean(precisions)) if precisions else 0.0, 6),
        "recall":             round(float(np.mean(recalls)) if recalls else 0.0, 6),
        "false_merges":       false_merges,
        "singleton_accuracy": round(singleton_correct / max(singleton_total, 1), 4),
        "n_entities":         len(entity_ids),
    }

