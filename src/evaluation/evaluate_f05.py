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


def evaluate_f05(predictions: pd.DataFrame,
                 ground_truth: pd.DataFrame,
                 beta: float = 0.5) -> dict:
    """
    Compute macro-averaged F_0.5, Precision, Recall.

    predictions:  [source1_entity_id, matched_entity_ids]  (comma-sep string)
    ground_truth: [source1_entity_id, matched_entity_ids]  (comma-sep string)

    Returns dict: {f05, precision, recall, false_merges, singleton_accuracy, n_entities}
    """
    def parse_ids(s):
        if pd.isna(s) or str(s).strip() == "":
            return set()
        return set(str(s).split(","))

    gt_dict   = {r.source1_entity_id: parse_ids(r.matched_entity_ids)
                 for _, r in ground_truth.iterrows()}
    pred_dict = {r.source1_entity_id: parse_ids(r.matched_entity_ids)
                 for _, r in predictions.iterrows()}

    entity_ids = set(gt_dict.keys())
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
        "f05":               round(np.mean(f05s), 6),
        "precision":         round(np.mean(precisions), 6),
        "recall":            round(np.mean(recalls), 6),
        "false_merges":      false_merges,
        "singleton_accuracy": round(singleton_correct / max(singleton_total, 1), 4),
        "n_entities":        len(entity_ids),
    }
