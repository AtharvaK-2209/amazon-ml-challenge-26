"""
build_features.py — Unified Phase 4 Feature Pipeline & Train/Validation Exporter.

Integrates Member 1 (Name) + Member 2 (Address) + Member 3 (Cross-field),
preserves candidate metadata (s1_entity_id, candidate_entity_id, candidate_source),
executes group-based entity train/validation splitting (to prevent entity data leakage),
validates data integrity, and outputs Parquet datasets + feature statistics + schema.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.features.pair_features import (
    PairFeatureExtractor,
    build_feature_matrix,
    generate_candidate_source_metadata,
    generate_feature_schema,
)
from src.features.validate_features import validate_feature_matrix
from src.preprocessing.normalize import Normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_features")


def generate_feature_statistics(df: pd.DataFrame, output_path: Optional[str | Path] = "features/feature_statistics.json") -> Dict[str, Any]:
    """
    Calculate and record statistical summaries for all numeric features (Part 9).
    """
    metadata_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    numeric_cols = [c for c in df.columns if c not in metadata_cols and np.issubdtype(df[c].dtype, np.number)]

    stats: Dict[str, Any] = {
        "summary": {
            "total_rows": int(len(df)),
            "total_features": int(len(numeric_cols)),
            "duplicate_candidate_pairs": int(df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum()),
            "s2_candidate_count": int((df["candidate_source"] == "S2").sum()),
            "s3_candidate_count": int((df["candidate_source"] == "S3").sum()),
        },
        "feature_metrics": {}
    }

    for col in numeric_cols:
        series = df[col]
        null_cnt = int(series.isnull().sum())
        stats["feature_metrics"][col] = {
            "min": float(series.min()) if not series.empty else 0.0,
            "max": float(series.max()) if not series.empty else 0.0,
            "mean": float(series.mean()) if not series.empty else 0.0,
            "median": float(series.median()) if not series.empty else 0.0,
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "missing_count": null_cnt,
            "missing_percentage": float(null_cnt / len(series) * 100.0) if len(series) > 0 else 0.0
        }

    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved feature statistics $\rightarrow$ {out_p}")

    return stats


def build_unified_feature_dataset(
    candidate_path: str | Path = "output/phase2/candidate_pairs.tsv",
    split: str = "test",
    val_split_ratio: float = 0.20,
    random_seed: int = 42,
    output_dir: str | Path = "features/"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Build unified Phase 4 feature table and perform leak-free GroupShuffleSplit on S1 entities.
    """
    cand_p = Path(candidate_path)
    if not cand_p.exists():
        raise FileNotFoundError(f"Candidate pair file does not exist at {cand_p}")

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

    # Load and filter raw source datasets
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

    # Attach Metadata (Step 4)
    cand_source = generate_candidate_source_metadata(candidates)
    meta_df = pd.DataFrame({
        "source1_entity_id": candidates["source1_entity_id"].values,
        "candidate_entity_id": candidates["candidate_entity_id"].values,
        "candidate_source": cand_source.values,
    })

    full_feature_df = pd.concat([meta_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Run Feature Validation (Step 7)
    val_report = validate_feature_matrix(candidates, full_feature_df)

    # Group-based Train/Validation Split by source1_entity_id (Step 6)
    # Prevents data leakage by ensuring all candidate pairs for an S1 entity stay together.
    groups = full_feature_df["source1_entity_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=random_seed)
    train_idx, val_idx = next(gss.split(full_feature_df, groups=groups))

    train_df = full_feature_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_feature_df.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Entity GroupShuffleSplit complete (80/20 by S1 ID): Train = {len(train_df):,} rows, Val = {len(val_df):,} rows")

    # Export Local Outputs & Schemas (Step 8 & Step 9)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "train_features.parquet"
    val_file = out_dir / "validation_features.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    logger.info(f"Saved local parquets $\rightarrow$ {train_file} & {val_file}")

    schema_file = out_dir / "feature_schema.json"
    generate_feature_schema(schema_file)

    stats_file = out_dir / "feature_statistics.json"
    generate_feature_statistics(full_feature_df, stats_file)

    return train_df, val_df, val_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Phase 4 Features")
    parser.add_argument("--candidate-path", default="output/phase2/candidate_pairs.tsv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="features/")
    args = parser.parse_args()

    build_unified_feature_dataset(
        candidate_path=args.candidate_path,
        split=args.split,
        val_split_ratio=args.val_ratio,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
