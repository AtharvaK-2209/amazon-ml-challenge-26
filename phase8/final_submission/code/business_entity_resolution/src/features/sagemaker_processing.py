#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — SageMaker Processing Entry Point for Phase 4.

This script executes the unified Phase 4 feature-engineering pipeline inside a
SageMaker Processing container (or locally).

It loads candidate pairs and source entity files, constructs name, address, and
cross-field features, validates the dataset, performs group-based train/val splitting,
and saves Parquet datasets, schema, and statistics reports to the designated output folder.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure src module is importable regardless of execution context
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.build_features import build_unified_feature_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sagemaker_processing")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 SageMaker Feature Processing Entry Point"
    )
    
    # Defaults align with standard SageMaker Processing paths (/opt/ml/processing/...)
    # or fallback to local project paths if running outside SageMaker.
    default_input_dir = os.environ.get("SM_INPUT_DIR", "/opt/ml/processing/input")
    default_output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")
    
    parser.add_argument(
        "--candidate-path",
        type=str,
        default=None,
        help="Path to candidate pairs TSV/Parquet file."
    )
    parser.add_argument(
        "--source1-path",
        type=str,
        default=None,
        help="Path to S1 entity CSV/Parquet file."
    )
    parser.add_argument(
        "--source2-path",
        type=str,
        default=None,
        help="Path to S2 entity CSV/Parquet file."
    )
    parser.add_argument(
        "--source3-path",
        type=str,
        default=None,
        help="Path to S3 entity CSV/Parquet file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated datasets and reports."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split dataset type ('train' or 'test')."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)."
    )
    
    return parser.parse_args()


def resolve_path(candidate_path: str, default_relative: str, sm_subpath: str) -> Path:
    """Helper to resolve paths dynamically between SageMaker and Local modes."""
    if candidate_path and Path(candidate_path).exists():
        return Path(candidate_path)
    
    # Check SageMaker input location
    sm_path = Path("/opt/ml/processing/input") / sm_subpath
    if sm_path.exists():
        return sm_path
    
    # Fallback to local workspace relative path
    local_path = project_root / default_relative
    return local_path


def main():
    args = parse_args()
    logger.INFO if hasattr(logger, 'INFO') else None
    logger.info("Starting Phase 4 SageMaker Feature Processing Execution...")
    
    # Resolve input paths
    candidate_file = resolve_path(args.candidate_path, "output/phase2/candidate_pairs.tsv", "candidates/candidate_pairs.tsv")
    source1_file = resolve_path(args.source1_path, "data/raw/dataset/source_1.csv", "data/source_1.csv")
    source2_file = resolve_path(args.source2_path, "data/raw/dataset/source_2.csv", "data/source_2.csv")
    source3_file = resolve_path(args.source3_path, "data/raw/dataset/source_3.csv", "data/source_3.csv")
    
    # Resolve output path
    output_dir = Path(args.output_dir) if args.output_dir else Path("/opt/ml/processing/output/features")
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Candidate Input: {candidate_file}")
    logger.info(f"Source 1 Input:  {source1_file}")
    logger.info(f"Source 2 Input:  {source2_file}")
    logger.info(f"Source 3 Input:  {source3_file}")
    logger.info(f"Output Directory: {output_dir}")
    
    if not candidate_file.exists():
        raise FileNotFoundError(f"Candidate file not found at: {candidate_file}")
    
    # Execute unified feature processing pipeline
    train_df, val_df, schema, stats = build_unified_feature_dataset(
        candidate_path=candidate_file,
        source1_path=source1_file,
        source2_path=source2_file,
        source3_path=source3_file,
        output_dir=output_dir,
        split=args.split,
        val_ratio=args.val_ratio
    )
    
    logger.info("SageMaker Feature Processing completed successfully!")
    logger.info(f"Artifacts generated in: {output_dir}")
    logger.info(f"  - train_features.parquet      ({len(train_df)} rows)")
    logger.info(f"  - validation_features.parquet ({len(val_df)} rows)")
    logger.info(f"  - feature_schema.json         ({len(schema)} features)")
    logger.info(f"  - feature_statistics.json")


if __name__ == "__main__":
    main()
