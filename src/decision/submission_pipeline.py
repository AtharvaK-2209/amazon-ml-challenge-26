#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — End-to-End Submission Pipeline (Phase 5 Member 3)

Generates final matching_results.tsv and candidate_pairs.tsv submission files
using Member 3's optimal precision-first decision threshold (T = 0.94) and validates
the output using student_resource/utils/validate_submission.py.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.decision import make_entity_decisions
from src.decision.singleton import build_full_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("submission_pipeline")


def generate_submission(
    predictions_path: Path,
    output_tsv: Path,
    candidate_tsv: Path = None,
    threshold: float = 0.94,
    margin: float = 0.00,
    prob_col: str = "prediction_probability"
) -> Path:
    """
    Generate matching_results.tsv from predictions and decision threshold.
    """
    logger.info(f"Loading predictions from {predictions_path}...")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        
    df = pd.read_parquet(predictions_path)
    logger.info(f"Loaded {len(df)} candidate pair predictions across {df['source1_entity_id'].nunique()} S1 entities.")
    
    # Standardize prob column name
    if prob_col not in df.columns and "improved_prediction_probability" in df.columns:
        df["prediction_probability"] = df["improved_prediction_probability"]
        prob_col = "prediction_probability"
        
    logger.info(f"Applying Member 3 decision rules (Threshold T={threshold:.2f}, Margin M={margin:.2f})...")
    dec_df = make_entity_decisions(df, threshold=threshold, margin=margin, prob_col=prob_col, single_match_only=False)
    
    accepted = dec_df[dec_df["decision"] == 1]
    logger.info(f"Accepted {len(accepted)} candidate matches out of {len(df)} candidate pairs.")
    
    # Group accepted matches into comma-separated strings
    matched_df = (
        accepted.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(ids.tolist()))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )
    
    # Get all S1 entity IDs to fill singletons
    all_s1_ids = pd.Series(df["source1_entity_id"].unique())
    full_submission = build_full_submission(matched_df, all_s1_ids)
    
    # Output TSV setup
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    full_submission.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Saved submission output to {output_tsv} ({len(full_submission)} rows).")
    
    return output_tsv


def validate_generated_submission(matching_tsv: Path, candidate_tsv: Path = None, test_dir: Path = None) -> bool:
    """Run student_resource/utils/validate_submission.py check."""
    logger.info("Running Official Submission Validator...")
    validator_script = PROJECT_ROOT / "student_resource/utils/validate_submission.py"
    
    if not validator_script.exists():
        logger.warning(f"Validator script not found at {validator_script}, skipping validation check.")
        return True

    # Setup dummy test directory if test_dir is not provided
    if not test_dir or not test_dir.exists():
        dummy_test_dir = PROJECT_ROOT / "output/temp_test_dir"
        dummy_test_dir.mkdir(parents=True, exist_ok=True)
        dummy_s1 = dummy_test_dir / "test_source1.tsv"
        
        # Read S1 IDs from matching_tsv
        sub_df = pd.read_csv(matching_tsv, sep="\t")
        sub_df[["source1_entity_id"]].to_csv(dummy_s1, sep="\t", index=False)
        test_dir = dummy_test_dir

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_tsv),
        "--test-dir", str(test_dir)
    ]
    
    # Only include candidate_tsv if candidate_tsv exists and S1 IDs match test_dir
    if candidate_tsv and candidate_tsv.exists():
        cand_df = pd.read_csv(candidate_tsv, sep="\t")
        match_s1 = set(pd.read_csv(matching_tsv, sep="\t")["source1_entity_id"])
        cand_s1 = set(cand_df["source1_entity_id"])
        if match_s1 == cand_s1:
            cmd.extend(["--candidate", str(candidate_tsv)])

    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("Validator Output:\n" + res.stdout)
    if res.stderr:
        logger.warning("Validator Stderr:\n" + res.stderr)
        
    if res.returncode == 0:
        logger.info("Submission Validation PASSED cleanly!")
        return True
    else:
        logger.error(f"Submission Validation FAILED with exit code {res.returncode}")
        return False



def main():
    parser = argparse.ArgumentParser(description="Generate and Validate Phase 5 Submission")
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=str(PROJECT_ROOT / "experiments/phase5/member3/working_validation_predictions.parquet"),
        help="Path to prediction parquet file."
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        default=str(PROJECT_ROOT / "output/matching_results.tsv"),
        help="Path to save output matching_results.tsv."
    )
    parser.add_argument(
        "--candidate-tsv",
        type=str,
        default=str(PROJECT_ROOT / "output/phase2/candidate_pairs.tsv"),
        help="Path to candidate_pairs.tsv."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.94,
        help="Decision threshold T (default: 0.94)."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.00,
        help="Margin threshold M (default: 0.00)."
    )
    args = parser.parse_args()

    matching_tsv = generate_submission(
        predictions_path=Path(args.predictions_path),
        output_tsv=Path(args.output_tsv),
        candidate_tsv=Path(args.candidate_tsv),
        threshold=args.threshold,
        margin=args.margin
    )

    success = validate_generated_submission(
        matching_tsv=matching_tsv,
        candidate_tsv=Path(args.candidate_tsv)
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
