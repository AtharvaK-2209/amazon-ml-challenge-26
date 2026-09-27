#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — SageMaker Processing Entry Point for Phase 6.

This script executes the complete Phase 6 Integration Pipeline (Inference, Calibration,
Decision Engine, Validation) inside an AWS SageMaker Processing container (or locally).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.phase6.inference import run_phase6_inference
from src.phase6.calibration import apply_calibration
from src.decision.decision_engine import EntityDecisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("phase6_sagemaker_processing")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6 SageMaker Processing Entry Point")
    
    parser.add_argument("--features-path", type=str, default=None, help="Path to input features parquet")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model artifact")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output decisions & statistics")
    parser.add_argument("--threshold", type=float, default=0.94, help="Decision threshold T (default: 0.94)")
    parser.add_argument("--margin", type=float, default=0.00, help="Margin threshold M (default: 0.00)")
    parser.add_argument("--calibration-method", type=str, default="raw", help="Calibration method (default: raw)")
    
    return parser.parse_args()


def resolve_path(cli_path: str, local_relative: str, sm_subpath: str) -> Path:
    if cli_path and Path(cli_path).exists():
        return Path(cli_path)
    sm_path = Path("/opt/ml/processing/input") / sm_subpath
    if sm_path.exists():
        return sm_path
    local_path = project_root / local_relative
    return local_path


def main():
    args = parse_args()
    logger.info("Starting Phase 6 SageMaker Processing Execution...")
    
    features_file = resolve_path(args.features_path, "features/validation_features.parquet", "features/validation_features.parquet")
    model_file = resolve_path(args.model_path, "models/xgb_baseline.json", "model/xgb_baseline.json")
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("/opt/ml/processing/output/phase6")
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Features Input: {features_file}")
    logger.info(f"Model Input:    {model_file}")
    logger.info(f"Output Path:    {output_dir}")
    logger.info(f"Parameters:     T={args.threshold:.2f}, M={args.margin:.2f}, Calib={args.calibration_method}")
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found at {features_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
        
    # 1. Load features
    import pandas as pd
    features_df = pd.read_parquet(features_file)
    logger.info(f"Loaded {len(features_df)} feature rows across {features_df['source1_entity_id'].nunique()} S1 entities.")
    
    # 2. Run Model Inference
    preds_df = run_phase6_inference(features_df, model_path=model_file)
    
    # 3. Apply Calibration
    preds_calib = apply_calibration(preds_df, method=args.calibration_method)
    
    # 4. Decision Engine Execution
    engine = EntityDecisionEngine(threshold=args.threshold, margin=args.margin, prob_col="prediction_probability")
    entity_matches, stats = engine.process_decisions(preds_calib)
    
    # 5. Export Outputs
    entity_matches.to_parquet(output_dir / "entity_matches.parquet", index=False)
    
    import json
    with open(output_dir / "decision_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    logger.info("SageMaker Phase 6 Processing Job COMPLETED successfully!")
    logger.info(f"Matched entities: {stats['matched_entities']}, Unmatched: {stats['unmatched_entities']}")


if __name__ == "__main__":
    main()
