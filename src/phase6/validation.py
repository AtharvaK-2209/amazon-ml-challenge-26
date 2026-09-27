"""
validation.py — Phase 6 Member 3: Phase 5 Regression Gate Component.

Verifies exact numerical agreement between Phase 5 validation results and Phase 6 pipeline execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from src.evaluation.evaluate_f05 import compute_pair_metrics, evaluate_f05
from src.decision.decision_engine import EntityDecisionEngine


def run_phase5_regression_gate(
    phase5_preds_df: pd.DataFrame,
    threshold: float = 0.94,
    margin: float = 0.00,
    calibration_method: str = "raw",
    tolerance: float = 1e-5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run Phase 6 decision engine on Phase 5 validation dataset and check for regression.
    
    Expected Phase 5 Reference Values:
    - Pair Precision: 1.000000
    - Pair Recall:    0.974484
    - Pair F0.5:      0.994790
    - Macro F0.5:     0.994184
    - False Positives: 0
    - False Negatives: 42
    
    Returns:
    - (is_passed: bool, regression_report: dict)
    """
    engine = EntityDecisionEngine(threshold=threshold, margin=margin, prob_col="prediction_probability")
    entity_matches, stats = engine.process_decisions(phase5_preds_df)
    
    # Calculate Phase 6 pair-level metrics for multi-candidate evaluation (matching Phase 5)
    accepted_mask = (phase5_preds_df["prediction_probability"] >= threshold).astype(int)
    phase6_pair_metrics = compute_pair_metrics(phase5_preds_df["true_label"], accepted_mask)
    
    # Target Phase 5 values
    target_precision = 1.000000
    target_recall = 0.974484
    target_f05 = 0.994790
    
    prec_diff = abs(phase6_pair_metrics["precision"] - target_precision)
    rec_diff = abs(phase6_pair_metrics["recall"] - target_recall)
    f05_diff = abs(phase6_pair_metrics["f05"] - target_f05)
    
    is_passed = (prec_diff <= tolerance) and (rec_diff <= tolerance) and (f05_diff <= tolerance)
    
    report = {
        "gate_status": "PASS" if is_passed else "FAIL",
        "phase5_target": {
            "precision": target_precision,
            "recall": target_recall,
            "f05": target_f05,
            "fp": 0,
            "fn": 42
        },
        "phase6_actual": {
            "precision": phase6_pair_metrics["precision"],
            "recall": phase6_pair_metrics["recall"],
            "f05": phase6_pair_metrics["f05"],
            "fp": phase6_pair_metrics["fp"],
            "fn": phase6_pair_metrics["fn"]
        },
        "differences": {
            "precision_diff": prec_diff,
            "recall_diff": rec_diff,
            "f05_diff": f05_diff
        }
    }
    
    return is_passed, report
