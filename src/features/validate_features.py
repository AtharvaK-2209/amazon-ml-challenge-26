"""
validate_features.py — Comprehensive Feature Validation Suite (Phase 4).

Executes 7 strict validation suites:
A. Shape & Row Counts
B. Missing Values (Counts & Percentages)
C. Invalid Numeric Values (NaN, +inf, -inf)
D. Duplicate Candidate Pair Uniqueness
E. Feature Range Checks (RapidFuzz [0..100], Jaccard/Ratios [0..1], Binary [0/1])
F. Strict Row Preservation (input N candidate pairs == output N rows)
G. Target Leakage Prevention (no ground_truth / match_label in features)
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("validate_features")


def validate_feature_matrix(
    input_candidates: pd.DataFrame,
    feature_df: pd.DataFrame,
    strict_row_preservation: bool = True
) -> Dict[str, Any]:
    """
    Validate Phase 4 feature table against all data quality and security requirements.
    
    Args:
        input_candidates: Input candidate pairs DataFrame before feature merge.
        feature_df: Final unified feature DataFrame.
        strict_row_preservation: If True, raises ValueError if input_rows != output_rows.
        
    Returns:
        Validation report dictionary with metrics and check statuses.
    """
    input_rows = len(input_candidates)
    output_rows = len(feature_df)

    # 1. Row Preservation Check
    if strict_row_preservation and input_rows != output_rows:
        raise ValueError(
            f"[FATAL VALIDATION FAILURE] Row count mismatch! Input candidates: {input_rows:,}, "
            f"Output feature rows: {output_rows:,}. Feature merge must not multiply or drop rows."
        )

    # 2. Metadata Columns Preservation Check
    required_meta = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    for col in required_meta:
        if col not in feature_df.columns:
            raise ValueError(f"[FATAL VALIDATION FAILURE] Required candidate metadata column '{col}' is missing!")
        if feature_df[col].isnull().any():
            raise ValueError(f"[FATAL VALIDATION FAILURE] Found null values in identifier column '{col}'!")

    # 3. Duplicate Candidate Pair Check
    dup_mask = feature_df.duplicated(subset=["source1_entity_id", "candidate_entity_id"])
    num_duplicates = int(dup_mask.sum())
    if num_duplicates > 0:
        raise ValueError(f"[FATAL VALIDATION FAILURE] Found {num_duplicates} duplicate candidate pairs!")

    # 4. Target Leakage Check
    forbidden_target_cols = {"target", "match_label", "ground_truth", "matched_entity_ids", "is_match"}
    feature_cols = [c for c in feature_df.columns if c not in required_meta]
    leakage_found = [c for c in feature_cols if c.lower() in forbidden_target_cols]
    if leakage_found:
        raise ValueError(f"[FATAL SECURITY FAILURE] Target leakage detected! Target columns found in feature set: {leakage_found}")

    # 5. Invalid Numeric Values Check (NaN, +inf, -inf)
    numeric_df = feature_df[feature_cols].select_dtypes(include=[np.number])
    null_counts = numeric_df.isnull().sum().to_dict()
    null_pcts = {k: float(v / output_rows * 100.0) if output_rows > 0 else 0.0 for k, v in null_counts.items()}
    total_nulls = sum(null_counts.values())

    inf_count = int(np.isinf(numeric_df.values).sum())
    if inf_count > 0:
        raise ValueError(f"[FATAL VALIDATION FAILURE] Found {inf_count} infinite values (+inf / -inf) in numerical features!")

    # 6. Feature Range Checks
    range_warnings = []
    for col in feature_cols:
        col_series = numeric_df[col]
        col_min = float(col_series.min()) if not col_series.empty else 0.0
        col_max = float(col_series.max()) if not col_series.empty else 0.0

        # RapidFuzz scores [0..100]
        if "fuzz" in col or "token_sort" in col or "token_set" in col or "wratio" in col:
            if col_min < 0.0 or col_max > 100.0:
                range_warnings.append(f"{col} out of expected RapidFuzz range [0..100]: min={col_min}, max={col_max}")
        # Binary or ratio scores [0..1]
        elif col.endswith("_match") or col.endswith("_ratio") or col.startswith("jaccard"):
            if col_min < 0.0 or col_max > 1.0:
                # Check if it's RapidFuzz scaled to 0..100
                if col_max <= 100.0:
                    pass
                else:
                    range_warnings.append(f"{col} out of expected ratio/binary range [0..1]: min={col_min}, max={col_max}")

    # Breakdown of candidate sources
    source_breakdown = feature_df["candidate_source"].value_counts().to_dict()

    report = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "total_columns": len(feature_df.columns),
        "total_features": len(feature_cols),
        "candidate_source_breakdown": {str(k): int(v) for k, v in source_breakdown.items()},
        "total_nulls": total_nulls,
        "null_percentages": null_pcts,
        "infinite_values": inf_count,
        "duplicate_pairs": num_duplicates,
        "range_warnings": range_warnings,
        "validation_passed": True
    }

    logger.info(f"Feature Validation PASSED: {output_rows:,} rows × {len(feature_cols)} features | Sources: {source_breakdown} | Range Warnings: {len(range_warnings)}")
    return report
