"""threshold.py — Phase 6: Threshold sweep for precision-first decision."""
import numpy as np
import pandas as pd
from src.config import DECISION
from src.evaluation.evaluate_f05 import evaluate_f05


def sweep_thresholds(candidates_df: pd.DataFrame,
                     ground_truth: pd.DataFrame,
                     thresholds: list | None = None) -> pd.DataFrame:
    """
    Sweep match probability thresholds on a validation split.
    Returns DataFrame with [threshold, precision, recall, f05, false_merges].
    """
    thresholds = thresholds or DECISION["threshold_sweep"]
    results = []

    for t in thresholds:
        matched = (
            candidates_df[candidates_df["match_score"] >= t]
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(ids.tolist()))
            .reset_index()
            .rename(columns={"candidate_entity_id": "matched_entity_ids"})
        )
        metrics = evaluate_f05(matched, ground_truth)
        results.append({"threshold": t, **metrics})
        print(f"  t={t:.2f}  F0.5={metrics['f05']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    return pd.DataFrame(results)
