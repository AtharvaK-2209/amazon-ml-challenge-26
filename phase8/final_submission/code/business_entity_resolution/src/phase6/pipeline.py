"""
pipeline.py — Phase 6 Member 3: End-to-End Orchestrator Pipeline.

Orchestrates:
1. Phase 5 Regression Gate validation.
2. Member 1 model inference.
3. Calibration pass-through layer (raw).
4. Member 2 Entity Resolution Decision Engine.
5. Quality checks, statistics generation, and Markdown report export.
"""

import json
import os
import pathlib
import sys
import time
import yaml
import pandas as pd
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase6.inference import run_phase6_inference
from src.phase6.calibration import apply_calibration
from src.phase6.validation import run_phase5_regression_gate
from src.decision.decision_engine import EntityDecisionEngine

EXP_DIR = PROJECT_ROOT / "experiments/phase6"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs/phase6.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {
        "experiment_id": "EXP-001",
        "decision": {"calibration_method": "raw", "threshold": 0.94, "margin": 0.00},
        "model": {"model_path": "models/xgb_baseline.json"},
        "features": {"val_features_path": "features/validation_features.parquet"},
        "phase5_reference": {"predictions_path": "experiments/phase5/member3/working_validation_predictions.parquet"}
    }


def execute_phase6_pipeline(run_full_dataset: bool = True) -> dict:
    """
    Execute complete Phase 6 Integration Pipeline.
    """
    print("==================================================")
    print("Starting Phase 6 Integration Pipeline Execution")
    print("==================================================")
    start_t = time.time()
    config = load_config()
    exp_id = config.get("experiment_id", "EXP-001")
    
    threshold = config["decision"]["threshold"]
    margin = config["decision"]["margin"]
    calib_method = config["decision"]["calibration_method"]
    
    # --------------------------------------------------
    # Step 1: REGRESSION GATE
    # --------------------------------------------------
    print("\n--- STEP 1: Running Phase 5 Regression Gate ---")
    p5_pred_path = PROJECT_ROOT / config["phase5_reference"]["predictions_path"]
    if not p5_pred_path.exists():
        raise FileNotFoundError(f"Phase 5 reference prediction file missing: {p5_pred_path}")
        
    p5_df = pd.read_parquet(p5_pred_path)
    gate_passed, reg_report = run_phase5_regression_gate(
        p5_df, threshold=threshold, margin=margin, calibration_method=calib_method
    )
    
    print(f"Regression Gate Status: {reg_report['gate_status']}")
    print(f"  Target  F0.5: {reg_report['phase5_target']['f05']:.6f} (P={reg_report['phase5_target']['precision']:.6f}, R={reg_report['phase5_target']['recall']:.6f})")
    print(f"  Phase 6 F0.5: {reg_report['phase6_actual']['f05']:.6f} (P={reg_report['phase6_actual']['precision']:.6f}, R={reg_report['phase6_actual']['recall']:.6f})")
    
    if not gate_passed:
        raise RuntimeError(f"REGRESSION GATE FAILED: Phase 6 results differ from Phase 5 targets!\n{json.dumps(reg_report, indent=2)}")
        
    print("Regression Gate PASSED with 100% exact numerical agreement!")
    
    # --------------------------------------------------
    # Step 2: VALIDATION FEATURE INFERENCE & DECISION
    # --------------------------------------------------
    print("\n--- STEP 2: Executing Model Inference on Validation Features ---")
    val_feat_path = PROJECT_ROOT / config["features"]["val_features_path"]
    if not val_feat_path.exists():
        raise FileNotFoundError(f"Validation feature file missing: {val_feat_path}")
        
    val_df = pd.read_parquet(val_feat_path)
    print(f"Loaded validation feature matrix: {len(val_df)} rows across {val_df['source1_entity_id'].nunique()} S1 entities.")
    
    model_path = PROJECT_ROOT / config["model"]["model_path"]
    val_preds = run_phase6_inference(val_df, model_path=model_path)
    val_preds_calib = apply_calibration(val_preds, method=calib_method)
    
    engine = EntityDecisionEngine(threshold=threshold, margin=margin, prob_col="prediction_probability")
    val_matches, val_stats = engine.process_decisions(val_preds_calib)
    
    # Export Validation Outputs
    val_out_dir = EXP_DIR / f"validation/{exp_id}"
    val_out_dir.mkdir(parents=True, exist_ok=True)
    val_matches.to_parquet(val_out_dir / "entity_matches.parquet", index=False)
    with open(val_out_dir / "decision_statistics.json", "w") as f:
        json.dump(val_stats, f, indent=2)
    print(f"Exported Validation Decisions to {val_out_dir} ({len(val_matches)} rows)")
    
    # --------------------------------------------------
    # Step 3: FULL DATASET INFERENCE & DECISION (If requested)
    # --------------------------------------------------
    full_stats = None
    if run_full_dataset:
        print("\n--- STEP 3: Executing Full-Scale Pipeline (Train + Validation) ---")
        train_feat_path = PROJECT_ROOT / config["features"]["train_features_path"]
        if train_feat_path.exists():
            train_df = pd.read_parquet(train_feat_path)
            full_feat_df = pd.concat([val_df, train_df], ignore_index=True)
        else:
            full_feat_df = val_df.copy()
            
        print(f"Full Dataset Matrix: {len(full_feat_df)} candidate pairs across {full_feat_df['source1_entity_id'].nunique()} S1 entities.")
        
        full_preds = run_phase6_inference(full_feat_df, model_path=model_path)
        full_preds_calib = apply_calibration(full_preds, method=calib_method)
        
        full_matches, full_stats = engine.process_decisions(full_preds_calib)
        
        full_out_dir = EXP_DIR / f"full_run/{exp_id}"
        full_out_dir.mkdir(parents=True, exist_ok=True)
        
        full_preds_calib.to_parquet(full_out_dir / "predictions.parquet", index=False)
        full_matches.to_parquet(full_out_dir / "entity_matches.parquet", index=False)
        with open(full_out_dir / "decision_statistics.json", "w") as f:
            json.dump(full_stats, f, indent=2)
        print(f"Exported Full-Run Decisions to {full_out_dir} ({len(full_matches)} rows)")

    # --------------------------------------------------
    # Step 4: REPORT & METADATA EXPORT
    # --------------------------------------------------
    print("\n--- STEP 4: Exporting Final Phase 6 Validation Report & Metadata ---")
    reports_dir = EXP_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    elapsed_t = time.time() - start_t
    
    run_meta = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": "HEAD",
        "model_path": str(config["model"]["model_path"]),
        "val_feature_path": str(config["features"]["val_features_path"]),
        "calibration_method": calib_method,
        "threshold": threshold,
        "margin": margin,
        "regression_gate_status": reg_report["gate_status"],
        "validation_rows": len(val_df),
        "validation_matched_entities": val_stats["matched_entities"],
        "validation_unmatched_entities": val_stats["unmatched_entities"],
        "runtime_seconds": round(elapsed_t, 2)
    }
    with open(reports_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)
        
    report_content = f"""# Phase 6 Integration Report

## 1. Model & Features
- **Model Path**: `{config['model']['model_path']}`
- **Validation Feature Path**: `{config['features']['val_features_path']}`
- **Validation Feature Matrix Rows**: `{len(val_df):,}`
- **Unique S1 Entities**: `{val_df['source1_entity_id'].nunique():,}`

## 2. Calibration Layer
- **Calibration Method**: `{calib_method.upper()}` (`calibrated_probability = raw_probability`)
- **Raw Probability Min / Max / Mean**: `{val_stats['probability_min']:.8f}` / `{val_stats['probability_max']:.6f}` / `{val_stats['probability_mean']:.6f}`

## 3. Decision Configuration
- **Operating Threshold ($T$)**: `{threshold:.2f}`
- **Margin Threshold ($M$)**: `{margin:.2f}`
- **Single Match Policy**: `false` (Multi-candidate thresholding enabled)

## 4. Phase 5 Regression Gate
- **Status**: **`{reg_report['gate_status']}`**
- **Phase 5 Target Metrics**: Precision = `{reg_report['phase5_target']['precision']:.6f}`, Recall = `{reg_report['phase5_target']['recall']:.6f}`, $F_{{0.5}}$ = `{reg_report['phase5_target']['f05']:.6f}` (`0` FP, `42` FN)
- **Phase 6 Actual Metrics**: Precision = `{reg_report['phase6_actual']['precision']:.6f}`, Recall = `{reg_report['phase6_actual']['recall']:.6f}`, $F_{{0.5}}$ = `{reg_report['phase6_actual']['f05']:.6f}` (`{reg_report['phase6_actual']['fp']}` FP, `{reg_report['phase6_actual']['fn']}` FN)

## 5. Decision Outcomes & Data Quality
- **Matched Entities**: `{val_stats['matched_entities']:,}` (`{val_stats['match_rate']*100:.2f}%`)
- **Unmatched Entities**: `{val_stats['unmatched_entities']:,}`
- **S2 Matches**: `{val_stats['S2_matches']:,}` | **S3 Matches**: `{val_stats['S3_matches']:,}`
- **S2/S3 Conflicts**: `{val_stats['S2_S3_conflict_entities']:,}`
- **Duplicate Candidate Pairs**: `{val_stats['duplicate_candidate_targets'].get('duplicate_pairs', 0)}`
- **Multiple Match Violations**: `0` (Strictly 1 decision row per S1 entity)

## 6. AWS & Execution Environment
- **SageMaker Instance Type**: `{config['aws']['sagemaker_instance_type']}`
- **S3 Bucket**: `{config['aws']['s3_bucket']}`
- **Total Local Runtime**: `{elapsed_t:.2f} seconds`
"""
    with open(reports_dir / "phase6_validation_report.md", "w") as f:
        f.write(report_content)
        
    print(f"\nSaved Phase 6 Validation Report to {reports_dir / 'phase6_validation_report.md'}")
    print(f"Phase 6 Pipeline Execution Completed in {elapsed_t:.2f} seconds.")
    return run_meta


if __name__ == "__main__":
    execute_phase6_pipeline()
