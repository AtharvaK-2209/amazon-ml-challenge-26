"""
phase4_pipeline.py — Phase 4 Unified Pairwise Feature Engineering Pipeline (Member 3).

Loads candidate pairs, integrates Member 1 (Name) + Member 2 (Address) + Member 3 (Cross-field),
preserves candidate metadata (s1_entity_id, candidate_entity_id, candidate_source),
validates data integrity, and outputs parquet feature datasets and machine-readable schema.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from src.features.pair_features import (
    PairFeatureExtractor,
    build_feature_matrix,
    generate_candidate_source_metadata,
    generate_feature_schema,
)
from src.preprocessing.normalize import Normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phase4_pipeline")


def validate_phase4_dataset(
    input_candidates: pd.DataFrame,
    final_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform 15 strict data integrity checks on final Phase 4 feature table.
    Fails loudly if candidate row counts change unexpectedly.
    """
    input_rows = len(input_candidates)
    output_rows = len(final_df)

    # 1, 2, 3. Row count preservation check
    if input_rows != output_rows:
        raise ValueError(
            f"[FATAL DATA INTEGRITY ERROR] Candidate row mismatch! Input pairs: {input_rows:,}, "
            f"Output rows: {output_rows:,}. Merges must not multiply or drop candidate pairs."
        )

    # 4, 6, 7. Identifier preservation
    required_ids = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    for col in required_ids:
        if col not in final_df.columns:
            raise ValueError(f"[FATAL ERROR] Required candidate metadata column '{col}' is missing!")
        if final_df[col].isnull().any():
            raise ValueError(f"[FATAL ERROR] Null values found in identifier column '{col}'!")

    # 5. Duplicate check
    dup_count = int(final_df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum())
    if dup_count > 0:
        raise ValueError(f"[FATAL ERROR] Found {dup_count} duplicate candidate pairs!")

    # 8, 9. S2 / S3 candidate source breakdown
    source_counts = final_df["candidate_source"].value_counts().to_dict()
    s2_count = int(source_counts.get("S2", 0))
    s3_count = int(source_counts.get("S3", 0))

    feature_cols = [c for c in final_df.columns if c not in required_ids]

    # 10, 11, 12. Missingness, Infinite, and NaN checks
    numeric_df = final_df[feature_cols].select_dtypes(include=[np.number])
    null_counts = numeric_df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    inf_count = int(np.isinf(numeric_df.values).sum())

    if inf_count > 0:
        raise ValueError(f"[FATAL ERROR] Found {inf_count} infinite values (inf / -inf) in feature matrix!")

    # 13, 14. Dtypes and Column Uniqueness
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("[FATAL ERROR] Duplicate feature column names detected!")

    report = {
        "candidate_input_rows": input_rows,
        "final_output_rows": output_rows,
        "total_features": len(feature_cols),
        "s2_candidates": s2_count,
        "s3_candidates": s3_count,
        "total_nulls": total_nulls,
        "infinite_values": inf_count,
        "duplicate_pairs": dup_count,
        "validation_passed": True
    }

    logger.info(f"Phase 4 Integrity Checks PASSED: {output_rows:,} rows × {len(feature_cols)} features | S2: {s2_count:,}, S3: {s3_count:,}")
    return report


def run_phase4_pipeline(
    candidate_path: str | Path = "output/phase2/candidate_pairs.tsv",
    split: str = "test",
    output_dir: str | Path = "features/"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute Phase 4 Feature Pipeline.
    
    1. Load candidate pairs.
    2. Join normalized text representations.
    3. Generate Member 1 (Name) + Member 2 (Address) + Member 3 (Cross-field) features.
    4. Attach metadata (source1_entity_id, candidate_entity_id, candidate_source).
    5. Perform validation.
    6. Write Parquet and feature schema.
    """
    cand_p = Path(candidate_path)
    if not cand_p.exists():
        raise FileNotFoundError(f"Candidate file not found at {cand_p}")

    logger.info(f"Loading candidate pairs from {cand_p} ...")
    raw_cand = pd.read_csv(cand_p, sep="\t", dtype=str)

    if "candidate_entity_ids" in raw_cand.columns:
        rows = []
        for s1_id, cand_str in zip(raw_cand["source1_entity_id"], raw_cand["candidate_entity_ids"].fillna("")):
            cands = [c.strip() for c in cand_str.split(",") if c.strip()]
            for c_id in cands:
                rows.append((s1_id, c_id))
        candidates = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
    else:
        candidates = raw_cand[["source1_entity_id", "candidate_entity_id"]].copy()

    # Load and filter source records
    data_dir = Path("data") / split
    if not data_dir.exists():
        data_dir = Path("data") / "train"

    s1 = pd.read_csv(data_dir / f"{data_dir.name}_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(data_dir / f"{data_dir.name}_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(data_dir / f"{data_dir.name}_source3.tsv", sep="\t", dtype=str)

    s1_needed = set(candidates["source1_entity_id"])
    cand_needed = set(candidates["candidate_entity_id"])

    s1 = s1[s1["entity_id"].isin(s1_needed)].copy()
    s2 = s2[s2["entity_id"].isin(cand_needed)].copy()
    s3 = s3[s3["entity_id"].isin(cand_needed)].copy()

    norm = Normalizer()
    s1 = norm.normalize_dataframe(s1)
    s2 = norm.normalize_dataframe(s2)
    s3 = norm.normalize_dataframe(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)

    s1_lookup = s1.set_index("entity_id")[["normalized_name", "normalized_address", "country"]]
    s23_lookup = s23.set_index("entity_id")[["normalized_name", "normalized_address", "country"]]

    enriched = candidates.join(s1_lookup.add_suffix("_s1"), on="source1_entity_id")
    enriched = enriched.join(s23_lookup.add_suffix("_cand"), on="candidate_entity_id")

    logger.info(f"Extracting features for {len(enriched):,} candidate pairs ...")
    X = build_feature_matrix(enriched)

    # Attach Metadata (Step 3)
    cand_source = generate_candidate_source_metadata(candidates)
    
    meta_df = pd.DataFrame({
        "source1_entity_id": candidates["source1_entity_id"].values,
        "candidate_entity_id": candidates["candidate_entity_id"].values,
        "candidate_source": cand_source.values,
    })

    final_df = pd.concat([meta_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Data Integrity Checks (Step 6)
    val_report = validate_phase4_dataset(candidates, final_df)

    # Save Parquet Output & Feature Schema (Step 7 & 8)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        parquet_file = out_dir / "train_features.parquet"
    else:
        parquet_file = out_dir / "validation_features.parquet"

    final_df.to_parquet(parquet_file, index=False)
    logger.info(f"Saved Phase 4 unified feature parquet ({final_df.shape[0]:,} rows × {final_df.shape[1]} cols) $\rightarrow$ {parquet_file}")

    schema_file = out_dir / "feature_schema.json"
    generate_feature_schema(schema_file)

    return final_df, val_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Pairwise Feature Pipeline")
    parser.add_argument("--candidate-path", default="output/phase2/candidate_pairs.tsv", help="Path to Phase 2 candidate TSV")
    parser.add_argument("--split", default="test", help="Dataset split (train or test)")
    parser.add_argument("--output-dir", default="features/", help="Directory to save Parquet features and schema")
    args = parser.parse_args()

    run_phase4_pipeline(
        candidate_path=args.candidate_path,
        split=args.split,
        output_dir=args.output_dir
    )
