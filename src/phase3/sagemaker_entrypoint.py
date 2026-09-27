"""
sagemaker_entrypoint.py — Portable SageMaker Processing Entrypoint for Phase 3.

This script executes the exact same Phase 3 pipeline inside AWS SageMaker Processing instances.
No separate feature generation logic is written for SageMaker.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to python path for SageMaker environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.phase3.pipeline import Phase3Pipeline
from src.phase3.config import Phase3Config, load_phase3_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sagemaker_entrypoint")


def run_sagemaker():
    """
    SageMaker Processing container execution handler.
    Reads input candidate pairs from /opt/ml/processing/input/
    and outputs pair_features.parquet to /opt/ml/processing/output/
    """
    parser = argparse.ArgumentParser(description="Phase 3 SageMaker Entrypoint")
    parser.add_argument("--config", type=str, default="configs/phase3.yaml", help="Path to phase3.yaml")
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input", help="SageMaker input directory")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output", help="SageMaker output directory")
    args = parser.parse_args()

    logger.info("Starting SageMaker Phase 3 Processing Job ...")

    # Override config paths for SageMaker container directory layout if in container
    cfg_path = Path(args.config)
    pipeline = Phase3Pipeline(config_path=cfg_path if cfg_path.exists() else "configs/phase3.yaml")

    # In SageMaker Processing, input artifacts are mounted to /opt/ml/processing/input/
    sm_input_cand = Path(args.input_dir) / "candidate_pairs.tsv"
    if sm_input_cand.exists():
        pipeline.cfg.input.candidate_pairs = str(sm_input_cand)
        logger.info(f"Overriding candidate input path to SageMaker mount: {sm_input_cand}")

    # Output paths mounted to /opt/ml/processing/output/
    sm_output_dir = Path(args.output_dir)
    if sm_output_dir.exists():
        pipeline.cfg.output.local_dir = str(sm_output_dir)
        pipeline.cfg.output.parquet_file = str(sm_output_dir / "pair_features.parquet")
        pipeline.cfg.output.metadata_file = str(sm_output_dir / "feature_metadata.json")
        logger.info(f"Overriding output directory to SageMaker mount: {sm_output_dir}")

    # Run same core Phase 3 pipeline
    feature_df, metadata = pipeline.run()

    logger.info(f"SageMaker Processing Job completed successfully. Output rows: {len(feature_df):,}")


if __name__ == "__main__":
    run_sagemaker()
