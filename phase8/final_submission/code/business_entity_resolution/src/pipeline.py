"""
pipeline.py — End-to-end orchestrator
Runs all phases in sequence for a given split (train or test).

Usage:
    python -m src.pipeline --split test --threshold 0.70
"""

import argparse
import pandas as pd
from pathlib import Path

from src import config as CFG
from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    OUTPUT_DIR, FEATURES_DIR,
    OUTPUT_MATCHING, OUTPUT_CANDIDATES,
    DECISION,
)
from src.preprocessing.normalize     import Normalizer
from src.preprocessing.address_parser import parse_address
from src.blocking.tfidf_blocking      import tfidf_block
from src.blocking.exact_blocking      import exact_block
from src.blocking.token_blocking      import sorted_neighbourhood_block
from src.features.pair_features       import build_feature_matrix
from src.models.predict               import predict_matches
from src.decision.singleton           import build_full_submission
from src.evaluation.evaluate_f05      import evaluate_f05


def run_pipeline(split: str = "test",
                 threshold: float | None = None,
                 save_features: bool = True):
    """
    Full pipeline:
    Phase 1: load + normalise
    Phase 2: blocking (TF-IDF ∪ exact ∪ sorted-neighbourhood)
    Phase 3: feature engineering
    Phase 4: ML scoring + thresholding
    Phase 5: singleton filling + output writing
    """
    t = threshold or DECISION["default_threshold"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Phase 1: Load & Normalise ──────────────────────────
    print("\n[Phase 1] Loading & Normalising …")
    if split == "test":
        s1 = pd.read_csv(TEST_S1, sep="\t")
        s2 = pd.read_csv(TEST_S2, sep="\t")
        s3 = pd.read_csv(TEST_S3, sep="\t")
    else:
        s1 = pd.read_csv(TRAIN_S1, sep="\t")
        s2 = pd.read_csv(TRAIN_S2, sep="\t")
        s3 = pd.read_csv(TRAIN_S3, sep="\t")

    norm = Normalizer()
    s1   = norm.normalize_dataframe(s1)
    s2   = norm.normalize_dataframe(s2)
    s3   = norm.normalize_dataframe(s3)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1: {len(s1):,}  S2+S3: {len(s2s3):,}")

    # ── Phase 2: Blocking ──────────────────────────────────
    print("\n[Phase 2] Blocking …")
    cands_tfidf = tfidf_block(s1, s2s3)
    cands_exact = exact_block(s1, s2s3)
    cands_snb   = sorted_neighbourhood_block(s1, s2s3)

    candidates = (
        pd.concat([cands_tfidf, cands_exact, cands_snb], ignore_index=True)
        .drop_duplicates(subset=["source1_entity_id","candidate_entity_id"])
    )
    print(f"  Total candidate pairs: {len(candidates):,}")

    # Save candidate_pairs.tsv
    cand_out = candidates.groupby("source1_entity_id")["candidate_entity_id"] \
                         .apply(lambda ids: ",".join(ids.tolist())) \
                         .reset_index() \
                         .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    full_cands = build_full_submission(
        cand_out.rename(columns={"candidate_entity_ids":"matched_entity_ids"}),
        s1["entity_id"]
    ).rename(columns={"matched_entity_ids":"candidate_entity_ids"})
    full_cands.to_csv(OUTPUT_CANDIDATES, sep="\t", index=False)
    print(f"  candidate_pairs.tsv → {OUTPUT_CANDIDATES}")

    # ── Phase 3: Feature Engineering ──────────────────────
    print("\n[Phase 3] Building feature matrix …")
    # Enrich candidates with normalised fields
    s1_lookup  = s1.set_index("entity_id")[["normalized_name","normalized_address","country"]]
    s23_lookup = s2s3.set_index("entity_id")[["normalized_name","normalized_address","country"]]

    candidates = candidates.join(s1_lookup.add_suffix("_s1"),  on="source1_entity_id")
    candidates = candidates.join(s23_lookup.add_suffix("_cand"), on="candidate_entity_id")

    X = build_feature_matrix(candidates)
    if save_features:
        FEATURES_DIR.mkdir(exist_ok=True)
        feat_path = FEATURES_DIR / f"{split}_features.parquet"
        pd.concat([candidates[["source1_entity_id","candidate_entity_id"]], X], axis=1) \
          .to_parquet(feat_path)
        print(f"  Features saved → {feat_path}")

    # ── Phase 4: Predict & Threshold ──────────────────────
    print(f"\n[Phase 4] Scoring pairs (threshold={t}) …")
    matched = predict_matches(candidates, X, threshold=t)

    # ── Phase 5: Singleton filling + output ───────────────
    print("\n[Phase 5] Building submission …")
    submission = build_full_submission(matched, s1["entity_id"])
    submission.to_csv(OUTPUT_MATCHING, sep="\t", index=False)
    print(f"  matching_results.tsv → {OUTPUT_MATCHING}")
    print(f"  Matched entities: {(submission['matched_entity_ids'] != '').sum():,}")
    print(f"  Singletons:       {(submission['matched_entity_ids'] == '').sum():,}")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",     default="test",  choices=["train","test"])
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    run_pipeline(split=args.split, threshold=args.threshold)
