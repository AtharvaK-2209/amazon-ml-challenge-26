"""
Amazon ML Challenge 2026 — Calibration Layer (Phase 6)

Supports three calibration strategies:
  raw      — Pass raw model probabilities through unchanged. No artifact needed.
             Selected by Phase 5 Member 3 as optimal (matches Isotonic, beats Platt).
  platt    — Load a saved Platt/Sigmoid calibrator artifact and apply it.
  isotonic — Load a saved Isotonic Regression calibrator artifact and apply it.

For the Phase 6 production run, calibration_method = "raw" is used because
Phase 5 Member 3 experiments proved raw XGBoost probabilities are already
well-calibrated in rank ordering (F0.5 = 0.994790 at T=0.94 with 0 FP).

No calibration artifact is created or required for raw calibration.
A calibration artifact would only be required if Platt or Isotonic were selected
after a dedicated training/calibration stage (never fitted on the full data).
"""

import numpy as np
from pathlib import Path
from typing import Optional


SUPPORTED_METHODS = ("raw", "platt", "isotonic")


def apply_calibration(
    raw_probs: np.ndarray,
    method: str,
    calibrator_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Apply the selected calibration strategy to raw model probabilities.

    Args:
        raw_probs:        1D array of raw positive-class probabilities from the model.
        method:           One of 'raw', 'platt', 'isotonic'.
        calibrator_path:  Path to a serialised calibrator artifact (joblib).
                          Required for 'platt' and 'isotonic'; ignored for 'raw'.

    Returns:
        1D array of calibrated probabilities (same shape as raw_probs).

    Raises:
        ValueError:  Unknown method or artifact missing when required.
    """
    method = method.strip().lower()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown calibration method: '{method}'. "
            f"Supported methods: {SUPPORTED_METHODS}"
        )

    if method == "raw":
        # Phase 5 selected strategy: raw probabilities are already well-ranked.
        # No artifact is required or expected.
        return raw_probs.copy()

    # For platt / isotonic an artifact must exist.
    if calibrator_path is None or not Path(calibrator_path).exists():
        raise ValueError(
            f"{method.capitalize()} calibration selected but no calibration artifact "
            f"was provided at '{calibrator_path}'. "
            "A calibration artifact must be created during an explicit "
            "training/calibration stage and saved before Phase 6 inference. "
            "If Phase 5 selected 'raw', set calibration_method='raw' instead."
        )

    import joblib
    calibrator = joblib.load(calibrator_path)

    if method == "platt":
        # Platt: Logistic Regression on logit of raw probabilities.
        eps = 1e-7
        clipped = np.clip(raw_probs, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logits)[:, 1]

    if method == "isotonic":
        # Isotonic Regression: monotone mapping.
        return calibrator.predict(raw_probs)


def validate_calibrated_probs(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    method: str,
) -> None:
    """
    Run safety assertions on the calibrated probabilities.

    For 'raw' method, asserts that calibrated == raw (within floating-point tolerance).
    For all methods, asserts values are in [0, 1] and no NaN/Inf.
    """
    assert calibrated_probs.shape == raw_probs.shape, (
        f"Calibrated probs shape {calibrated_probs.shape} != raw probs shape {raw_probs.shape}"
    )
    assert not np.any(np.isnan(calibrated_probs)), "NaN found in calibrated probabilities."
    assert not np.any(np.isinf(calibrated_probs)), "Inf found in calibrated probabilities."
    assert np.all(calibrated_probs >= 0.0) and np.all(calibrated_probs <= 1.0), (
        "Calibrated probabilities are not in [0, 1]."
    )

    if method == "raw":
        assert np.allclose(calibrated_probs, raw_probs, atol=1e-7), (
            "RAW calibration: calibrated_probability must equal raw_probability. "
            "Values diverged beyond floating-point tolerance."
        )
