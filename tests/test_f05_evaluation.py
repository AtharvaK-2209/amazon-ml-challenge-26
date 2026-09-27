"""
test_f05_evaluation.py — Unit test for F0.5 evaluation logic against scikit-learn.
"""

import pytest
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from src.evaluation.evaluate_f05 import compute_pair_metrics, f_beta


def test_f_beta_formula():
    # Test manual calculation
    # P = 1.0, R = 0.5, Beta = 0.5
    # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.833333...
    res = f_beta(precision=1.0, recall=0.5, beta=0.5)
    assert round(res, 6) == round(0.625 / 0.75, 6)


def test_compute_pair_metrics_against_sklearn():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    
    sk_prec = float(precision_score(y_true, y_pred))
    sk_rec = float(recall_score(y_true, y_pred))
    sk_f05 = float(fbeta_score(y_true, y_pred, beta=0.5))
    sk_f1 = float(f1_score(y_true, y_pred))
    
    metrics = compute_pair_metrics(y_true, y_pred, beta=0.5)
    
    assert metrics["precision"] == round(sk_prec, 6)
    assert metrics["recall"] == round(sk_rec, 6)
    assert metrics["f05"] == round(sk_f05, 6)
    assert metrics["f1"] == round(sk_f1, 6)
    assert metrics["tp"] == 5
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
