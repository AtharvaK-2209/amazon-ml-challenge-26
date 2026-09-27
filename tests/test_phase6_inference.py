"""
Tests for Phase 6 calibration layer and inference pipeline.

Covers:
  TEST 1: RAW calibration — output equals input
  TEST 2: Isotonic selected but no artifact → clear error
  TEST 3: Platt selected but no artifact → clear error
  TEST 4: Invalid calibration method → clear error
  TEST 5: Phase 5 threshold is correctly defined
  TEST 6: Phase 5 margin is correctly defined
  TEST 7: validate_calibrated_probs passes for raw (identical arrays)
  TEST 8: validate_calibrated_probs raises for raw with diverged arrays
  TEST 9: Safety check on output dataframe row count
  TEST 10: Probability range [0, 1] validated
"""

import numpy as np
import pytest

from src.models.calibration import (
    apply_calibration,
    validate_calibrated_probs,
    SUPPORTED_METHODS,
)
from src.models.inference import (
    PHASE5_CALIBRATION_METHOD,
    PHASE5_THRESHOLD,
    PHASE5_MARGIN,
    compute_prediction_statistics,
)


# ── Test data ─────────────────────────────────────────────────────────────────
RAW_PROBS = np.array([0.0001, 0.02, 0.15, 0.50, 0.85, 0.9412, 0.999])


# ── TEST 1: RAW calibration is identity ───────────────────────────────────────
def test_raw_calibration_is_identity():
    """method='raw' must return a value numerically equal to the input."""
    calibrated = apply_calibration(RAW_PROBS, method="raw", calibrator_path=None)
    assert np.allclose(calibrated, RAW_PROBS, atol=1e-7), (
        "RAW calibration must not alter probabilities."
    )


# ── TEST 2: Isotonic without artifact raises informative error ─────────────────
def test_isotonic_missing_artifact_raises():
    """Isotonic selected but no artifact path → ValueError with clear message."""
    with pytest.raises(ValueError, match="calibration artifact"):
        apply_calibration(RAW_PROBS, method="isotonic", calibrator_path=None)


# ── TEST 3: Platt without artifact raises informative error ────────────────────
def test_platt_missing_artifact_raises():
    """Platt selected but no artifact path → ValueError with clear message."""
    with pytest.raises(ValueError, match="calibration artifact"):
        apply_calibration(RAW_PROBS, method="platt", calibrator_path=None)


# ── TEST 4: Invalid method raises informative error ────────────────────────────
def test_invalid_method_raises():
    """Unknown calibration method → ValueError naming supported methods."""
    with pytest.raises(ValueError, match="Unknown calibration method"):
        apply_calibration(RAW_PROBS, method="sigmoid", calibrator_path=None)


# ── TEST 5: Phase 5 threshold is correctly defined ────────────────────────────
def test_phase5_threshold():
    """PHASE5_THRESHOLD must be 0.94 (Member 3 optimised, F0.5=0.994790)."""
    assert PHASE5_THRESHOLD == 0.94, (
        f"Expected threshold 0.94, got {PHASE5_THRESHOLD}"
    )


# ── TEST 6: Phase 5 margin is correctly defined ───────────────────────────────
def test_phase5_margin():
    """PHASE5_MARGIN must be 0.00 (Member 3 optimised — multi-candidate thresholding)."""
    assert PHASE5_MARGIN == 0.00, (
        f"Expected margin 0.00, got {PHASE5_MARGIN}"
    )


# ── TEST 7: validate_calibrated_probs passes for identical arrays ──────────────
def test_validate_calibrated_probs_passes_for_raw():
    """validate_calibrated_probs raises no error when calibrated == raw."""
    calibrated = apply_calibration(RAW_PROBS, method="raw")
    # Must not raise
    validate_calibrated_probs(RAW_PROBS, calibrated, method="raw")


# ── TEST 8: validate_calibrated_probs raises when raw arrays diverge ──────────
def test_validate_calibrated_probs_fails_when_diverged():
    """validate_calibrated_probs raises AssertionError when raw != calibrated for 'raw'."""
    diverged = RAW_PROBS + 0.1  # Artificially shift values
    diverged = np.clip(diverged, 0, 1)
    with pytest.raises(AssertionError, match="calibrated_probability must equal"):
        validate_calibrated_probs(RAW_PROBS, diverged, method="raw")


# ── TEST 9: compute_prediction_statistics returns correct row count ────────────
def test_prediction_statistics_row_count():
    """Total candidate pairs in stats must match input array length."""
    stats = compute_prediction_statistics(RAW_PROBS)
    assert stats["total_candidate_pairs"] == len(RAW_PROBS)


# ── TEST 10: Probability range validation ─────────────────────────────────────
def test_probability_range_all_valid():
    """All calibrated probs must lie in [0, 1]."""
    calibrated = apply_calibration(RAW_PROBS, method="raw")
    assert np.all(calibrated >= 0.0), "Probabilities below 0."
    assert np.all(calibrated <= 1.0), "Probabilities above 1."


# ── TEST 11: Phase 5 calibration method ───────────────────────────────────────
def test_phase5_calibration_method_is_raw():
    """PHASE5_CALIBRATION_METHOD must be 'raw' as selected by Member 3."""
    assert PHASE5_CALIBRATION_METHOD == "raw"


# ── TEST 12: Supported methods list ───────────────────────────────────────────
def test_supported_methods_contains_all_three():
    """Calibration module must support raw, platt, and isotonic for future use."""
    for m in ("raw", "platt", "isotonic"):
        assert m in SUPPORTED_METHODS
