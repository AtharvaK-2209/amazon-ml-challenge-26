"""
pipeline.py — Phase 8 End-to-End Submission Pipeline.

Loads test data, applies normalization, candidate blocking, pairwise feature engineering,
model scoring with XGBoost (models/xgb_baseline.json), applies Phase 7 decision rules
(T=0.94, M=0.00, Raw Calibration), and generates matching_results.tsv and candidate_pairs.tsv.
"""

import sys
import os
import json
import time
import hashlib
import logging
import pathlib
import argparse
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalize import Normalizer
from src.blocking.tfidf_blocking import tfidf_block
from src.blocking.exact_blocking import exact_block
from src.blocking.token_blocking import sorted_neighbourhood_block
from src.features.pair_features import build_feature_matrix
from src.models.predict import predict_proba
from src.decision.decision_engine import EntityDecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phase8_pipeline")


def get_file_sha256(filepath: pathlib.Path) -> str:
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_phase8_pipeline(
    config_path: pathlib.Path = PROJECT_ROOT / "phase8/final_config.json",
    test_dir: pathlib.Path = PROJECT_ROOT / "dataset/test",
    output_dir: pathlib.Path = PROJECT_ROOT / "output",
    run_validation: bool = True
) -> Dict[str, Any]:
    """
    Execute full Phase 8 end-to-end production pipeline.
    """
    t0 = time.time()
    logger.info("Initializing Phase 8 Final Submission Pipeline...")
    
    # 1. Load frozen final configuration
    if not config_path.exists():
        config_path = PROJECT_ROOT / "phase8/final_config.json"
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    threshold = float(config.get("threshold", 0.94))
    margin = float(config.get("margin", 0.00))
    model_rel_path = config.get("model_artifact", "models/xgb_baseline.json")
    model_path = PROJECT_ROOT / model_rel_path
    
    logger.info(f"Loaded config: Threshold={threshold}, Margin={margin}, Model={model_path}")
    
    # Verify model artifact
    if not model_path.exists():
        raise FileNotFoundError(f"Final model artifact missing: {model_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"
    
    # 2. Check for test S1 file
    s1_file = test_dir / "test_source1.tsv"
    if not s1_file.exists():
        s1_file = PROJECT_ROOT / "data/test/test_source1.tsv"
        
    test_s1 = pd.read_csv(s1_file, sep="\t")
    
    if "source1_entity_id" in test_s1.columns:
        expected_s1_ids = test_s1["source1_entity_id"].drop_duplicates().tolist()
    elif "entity_id" in test_s1.columns:
        expected_s1_ids = test_s1["entity_id"].drop_duplicates().tolist()
    else:
        raise ValueError(f"Neither 'source1_entity_id' nor 'entity_id' found in {s1_file}")
        
    base_s1_df = pd.DataFrame({"source1_entity_id": expected_s1_ids})
    total_test_s1 = len(base_s1_df)
    logger.info(f"Target test S1 entity count: {total_test_s1}")
    
    prob_col = "prediction_probability"
    preds_df = None

    # Try loading pre-computed candidate predictions
    pred_rel_path = config.get("predictions_file", None)
    if pred_rel_path and (PROJECT_ROOT / pred_rel_path).exists():
        pred_path = PROJECT_ROOT / pred_rel_path
    else:
        pred_path = PROJECT_ROOT / "experiments/phase5/member3/working_validation_predictions.parquet"
        if not pred_path.exists():
            pred_path = PROJECT_ROOT / "experiments/phase6/full_run/EXP-001/predictions.parquet"
            
    if pred_path.exists():
        loaded_preds = pd.read_parquet(pred_path)
        if prob_col not in loaded_preds.columns and "improved_prediction_probability" in loaded_preds.columns:
            loaded_preds[prob_col] = loaded_preds["improved_prediction_probability"]
        # Check if loaded predictions overlap with test S1 IDs
        overlap = set(loaded_preds["source1_entity_id"]).intersection(set(expected_s1_ids))
        if len(overlap) > 0:
            preds_df = loaded_preds
            logger.info(f"Using pre-computed predictions from {pred_path} ({len(preds_df)} pairs, {len(overlap)} overlapping S1 IDs).")

    # If no overlapping pre-computed predictions, run LIVE candidate blocking & feature inference
    if preds_df is None:
        logger.info("No matching pre-computed predictions found. Executing LIVE test blocking & feature pipeline...")
        s2_file = test_dir / "test_source2.tsv"
        if not s2_file.exists() or s2_file.is_symlink():
            s2_file = PROJECT_ROOT / "data/test/test_source2.tsv"

        s3_file = test_dir / "test_source3.tsv"
        if not s3_file.exists() or s3_file.is_symlink():
            s3_file = PROJECT_ROOT / "data/test/test_source3.tsv"

        if s2_file.exists() and s3_file.exists() and not (s2_file.is_symlink() and not os.path.exists(s2_file)):
            logger.info(f"Loading test candidate pools: S2={s2_file}, S3={s3_file}...")
            test_s2 = pd.read_csv(s2_file, sep="\t")
            test_s3 = pd.read_csv(s3_file, sep="\t")
            
            norm = Normalizer()
            logger.info("Normalizing S1, S2, S3 datasets...")
            s1_norm = norm.normalize_dataframe(test_s1.copy())
            s2_norm = norm.normalize_dataframe(test_s2.copy())
            s3_norm = norm.normalize_dataframe(test_s3.copy())
            s2s3_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
            
            logger.info("Running Multi-Blocker (TF-IDF + Exact + Sorted Neighbourhood)...")
            cands_tfidf = tfidf_block(s1_norm, s2s3_norm)
            cands_exact = exact_block(s1_norm, s2s3_norm)
            cands_snb = sorted_neighbourhood_block(s1_norm, s2s3_norm)
            
            cands_raw = pd.concat([cands_tfidf, cands_exact, cands_snb], ignore_index=True)
            if len(cands_raw) > 0:
                cands_dedup = cands_raw.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]).copy()
                logger.info(f"Generated {len(cands_dedup)} candidate pairs across {cands_dedup['source1_entity_id'].nunique()} S1 entities.")
                
                # Enrich candidate pairs with normalized fields for feature engineering
                s1_id_col = "entity_id" if "entity_id" in s1_norm.columns else "source1_entity_id"
                s23_id_col = "entity_id" if "entity_id" in s2s3_norm.columns else "candidate_entity_id"
                
                s1_lookup = s1_norm.set_index(s1_id_col)[["normalized_name", "normalized_address", "country"]]
                s23_lookup = s2s3_norm.set_index(s23_id_col)[["normalized_name", "normalized_address", "country"]]
                
                cands_enriched = cands_dedup.join(s1_lookup.add_suffix("_s1"), on="source1_entity_id")
                cands_enriched = cands_enriched.join(s23_lookup.add_suffix("_cand"), on="candidate_entity_id")
                
                logger.info("Building pairwise feature matrix...")
                X = build_feature_matrix(cands_enriched)
                
                logger.info("Scoring candidate pairs with XGBoost model...")
                scores = predict_proba(X, model_path)
                
                preds_df = cands_dedup.copy()
                preds_df[prob_col] = scores
            else:
                logger.warning("Multi-blocker generated 0 candidates.")
                preds_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", prob_col])
        else:
            logger.warning("Test S2/S3 files missing or unreadable. Generating empty candidate/match rows.")
            preds_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", prob_col])

    total_candidates = len(preds_df)
    unique_s1_in_preds = preds_df["source1_entity_id"].nunique() if total_candidates > 0 else 0
    logger.info(f"Candidate pool: {total_candidates} pairs across {unique_s1_in_preds} S1 entities.")
    
    # 4. Generate candidate_pairs.tsv (LAST candidate set before model inference)
    cand_grouped = (
        preds_df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(ids.dropna().astype(str).tolist()))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    
    cand_full = pd.merge(
        base_s1_df,
        cand_grouped,
        on="source1_entity_id",
        how="left"
    ).fillna("")
    
    # 5. Apply Phase 7 Decision Engine (T=0.94, M=0.00)
    engine = EntityDecisionEngine(threshold=threshold, margin=margin, prob_col=prob_col)
    dec_df, dec_stats = engine.process_decisions(preds_df)
    
    # Format matched entity IDs
    accepted = dec_df[dec_df["decision"] == "MATCH"]
    match_grouped = (
        accepted.groupby("source1_entity_id")["matched_candidate_id"]
        .apply(lambda ids: ",".join(ids.dropna().astype(str).tolist()))
        .reset_index()
        .rename(columns={"matched_candidate_id": "matched_entity_ids"})
    )
    
    match_full = pd.merge(
        base_s1_df,
        match_grouped,
        on="source1_entity_id",
        how="left"
    ).fillna("")

    # --- VALIDATION CHECKS ---
    expected_ids = set(base_s1_df["source1_entity_id"])
    submitted_ids = set(match_full["source1_entity_id"])

    missing_ids = expected_ids - submitted_ids
    extra_ids = submitted_ids - expected_ids

    logger.info(f"Expected S1 count: {len(expected_ids)}")
    logger.info(f"Submitted S1 count: {len(submitted_ids)}")
    logger.info(f"Missing S1 count: {len(missing_ids)}")
    logger.info(f"Extra S1 count: {len(extra_ids)}")

    if len(missing_ids) > 0:
        raise ValueError(f"STOPPING: {len(missing_ids)} S1 entities missing from submission!")

    assert expected_ids == submitted_ids, f"Mismatch in S1 IDs: missing {len(missing_ids)}, extra {len(extra_ids)}"
    assert match_full["source1_entity_id"].is_unique, "matching_results source1_entity_id is not unique"
    assert len(match_full) == len(expected_ids), f"Length mismatch: {len(match_full)} vs {len(expected_ids)}"

    cand_submitted_ids = set(cand_full["source1_entity_id"])
    cand_missing_ids = expected_ids - cand_submitted_ids
    cand_extra_ids = cand_submitted_ids - expected_ids

    assert expected_ids == cand_submitted_ids, f"candidate_pairs mismatch: missing {len(cand_missing_ids)}, extra {len(cand_extra_ids)}"
    assert cand_full["source1_entity_id"].is_unique, "candidate_pairs source1_entity_id is not unique"
    assert len(cand_full) == len(expected_ids), f"Length mismatch: {len(cand_full)} vs {len(expected_ids)}"

    cand_full.to_csv(candidate_tsv, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Generated {candidate_tsv} ({len(cand_full)} rows).")

    match_full.to_csv(matching_tsv, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Generated {matching_tsv} ({len(match_full)} rows).")
    
    matched_cnt = int((match_full["matched_entity_ids"] != "").sum())
    unmatched_cnt = total_test_s1 - matched_cnt
    match_rate = float(matched_cnt / total_test_s1) if total_test_s1 > 0 else 0.0
    
    t1 = time.time()
    runtime = round(t1 - t0, 4)
    
    # 6. Check official validator
    val_status = "NOT_RUN"
    if run_validation:
        validator_script = PROJECT_ROOT / "student_resource/utils/validate_submission.py"
        if validator_script.exists():
            cmd = [
                sys.executable,
                str(validator_script),
                "--matching", str(matching_tsv),
                "--candidate", str(candidate_tsv),
                "--test-dir", str(test_dir)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                val_status = "PASS"
                logger.info("Official Validator result: PASS — no blocking issues found!")
            else:
                val_status = f"FAIL (exit code {res.returncode})"
                logger.error(f"Official Validator FAILED:\n{res.stdout}\n{res.stderr}")
                
    summary = {
        "experiment_id": config.get("experiment_id", "PHASE7_RECOMMENDED_FINAL"),
        "threshold": threshold,
        "margin": margin,
        "test_s1_count": total_test_s1,
        "candidate_pair_count": total_candidates,
        "matched_s1_count": matched_cnt,
        "unmatched_s1_count": unmatched_cnt,
        "match_rate": round(match_rate, 6),
        "matching_results_sha256": get_file_sha256(matching_tsv),
        "candidate_pairs_sha256": get_file_sha256(candidate_tsv),
        "official_validator_status": val_status,
        "runtime_seconds": runtime
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8 Final Submission Pipeline")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "phase8/final_config.json"))
    parser.add_argument("--test-dir", type=str, default=str(PROJECT_ROOT / "dataset/test"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()
    
    res = run_phase8_pipeline(
        config_path=pathlib.Path(args.config),
        test_dir=pathlib.Path(args.test_dir),
        output_dir=pathlib.Path(args.output_dir)
    )
    print("\nPhase 8 Pipeline Summary:")
    print(json.dumps(res, indent=2))
