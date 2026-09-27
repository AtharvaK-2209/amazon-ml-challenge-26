"""
decision_engine.py — Phase 6 Member 2 Entity Resolution Decision Engine.

Converts candidate-pair prediction probabilities into robust, entity-level matching decisions.
Reuses exact threshold (T=0.94) and margin (M=0.00) validated in Phase 5.
"""

import json
import pathlib
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.evaluation.evaluate_f05 import compute_pair_metrics, evaluate_f05
from src.decision.conflict import analyze_s2_s3_conflicts, analyze_target_duplications


class EntityDecisionEngine:
    """
    Phase 6 Entity-Level Decision Engine.
    
    Converts pairwise predictions to 1 row per reference S1 entity.
    """
    
    def __init__(
        self,
        threshold: float = 0.94,
        margin: float = 0.00,
        prob_col: str = "prediction_probability",
        singleton_require_threshold: bool = True
    ):
        self.threshold = threshold
        self.margin = margin
        self.prob_col = prob_col
        self.singleton_require_threshold = singleton_require_threshold

    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input predictions DataFrame schema and data integrity."""
        required_cols = ["source1_entity_id", "candidate_entity_id"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from predictions DataFrame.")
                
        if self.prob_col not in df.columns:
            if "improved_prediction_probability" in df.columns:
                df[self.prob_col] = df["improved_prediction_probability"]
            else:
                raise KeyError(f"Probability column '{self.prob_col}' missing.")
                
        # Recover candidate_source if missing
        if "candidate_source" not in df.columns:
            df["candidate_source"] = df["candidate_entity_id"].apply(
                lambda c: "S2" if str(c).startswith("S2") else ("S3" if str(c).startswith("S3") else "UNKNOWN")
            )
            
        # Integrity checks
        if df["source1_entity_id"].isnull().any() or df["candidate_entity_id"].isnull().any():
            raise ValueError("Null values detected in entity ID columns.")
            
        probs = df[self.prob_col].values
        if not np.issubdtype(probs.dtype, np.number):
            raise TypeError("Prediction probability column must be numeric.")
            
        if np.nanmin(probs) < 0.0 or np.nanmax(probs) > 1.0:
            raise ValueError(f"Probabilities out of bounds [0, 1]: min={np.nanmin(probs)}, max={np.nanmax(probs)}")
            
        # Duplicate pairs check
        dups = int(df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum())
        
        return {
            "total_rows": len(df),
            "unique_s1": df["source1_entity_id"].nunique(),
            "unique_candidates": df["candidate_entity_id"].nunique(),
            "duplicate_pairs": dups,
            "prob_min": float(np.nanmin(probs)),
            "prob_max": float(np.nanmax(probs)),
            "prob_mean": float(np.nanmean(probs))
        }

    def process_decisions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process pairwise predictions into entity-level decisions (1 row per S1).
        
        Returns:
        - entity_matches DataFrame
        - decision_statistics dictionary
        """
        self.validate_input(df)
        df_work = df.copy()
        
        # Deterministic sorting: probability DESC, candidate_entity_id ASC
        df_work = df_work.sort_values(
            ["source1_entity_id", self.prob_col, "candidate_entity_id"],
            ascending=[True, False, True]
        )
        
        # Calculate rank per S1 entity
        df_work["candidate_rank"] = df_work.groupby("source1_entity_id").cumcount() + 1
        
        # Entity-level aggregation logic
        records = []
        
        for s1_id, grp in df_work.groupby("source1_entity_id", sort=False):
            cand_count = len(grp)
            top_row = grp.iloc[0]
            top_cand_id = top_row["candidate_entity_id"]
            top_prob = float(top_row[self.prob_col])
            top_src = top_row["candidate_source"]
            
            if cand_count == 1:
                is_singleton = True
                sec_cand_id = None
                sec_prob = np.nan
                margin_val = np.nan
            else:
                is_singleton = False
                sec_row = grp.iloc[1]
                sec_cand_id = sec_row["candidate_entity_id"]
                sec_prob = float(sec_row[self.prob_col])
                margin_val = top_prob - sec_prob
                
            # Evaluate Decision Rules
            if is_singleton:
                if top_prob >= self.threshold:
                    decision = "MATCH"
                    decision_reason = "singleton_threshold_pass"
                    matched_cand_id = top_cand_id
                    final_prob = top_prob
                else:
                    decision = "NO_MATCH"
                    decision_reason = "singleton_threshold_fail"
                    matched_cand_id = None
                    final_prob = np.nan
            else:
                if top_prob < self.threshold:
                    decision = "NO_MATCH"
                    decision_reason = "threshold_fail"
                    matched_cand_id = None
                    final_prob = np.nan
                elif self.margin > 0.0 and margin_val < self.margin:
                    decision = "NO_MATCH"
                    decision_reason = "margin_fail"
                    matched_cand_id = None
                    final_prob = np.nan
                else:
                    decision = "MATCH"
                    decision_reason = "threshold_and_margin_pass"
                    matched_cand_id = top_cand_id
                    final_prob = top_prob
                    
            records.append({
                "source1_entity_id": s1_id,
                "matched_candidate_id": matched_cand_id,
                "candidate_source": top_src if decision == "MATCH" else None,
                "final_probability": final_prob,
                "top_candidate_id": top_cand_id,
                "top_probability": top_prob,
                "second_candidate_id": sec_cand_id,
                "second_probability": sec_prob,
                "margin": margin_val,
                "candidate_rank": 1 if decision == "MATCH" else np.nan,
                "candidate_count": cand_count,
                "is_singleton": is_singleton,
                "decision": decision,
                "decision_reason": decision_reason
            })
            
        entity_matches = pd.DataFrame(records)
        
        # Compute Decision Statistics
        total_s1 = len(entity_matches)
        matched_cnt = int((entity_matches["decision"] == "MATCH").sum())
        unmatched_cnt = int((entity_matches["decision"] == "NO_MATCH").sum())
        match_rate = float(matched_cnt / total_s1) if total_s1 > 0 else 0.0
        
        singletons = int(entity_matches["is_singleton"].sum())
        multi_cands = total_s1 - singletons
        
        s2_matches = int((entity_matches[entity_matches["decision"] == "MATCH"]["candidate_source"] == "S2").sum())
        s3_matches = int((entity_matches[entity_matches["decision"] == "MATCH"]["candidate_source"] == "S3").sum())
        
        conflicts = analyze_s2_s3_conflicts(df, threshold=0.70)
        duplication_stats = analyze_target_duplications(entity_matches)
        
        reason_counts = entity_matches["decision_reason"].value_counts().to_dict()
        
        margins = entity_matches["margin"].dropna().values
        
        stats = {
            "total_source1_entities": total_s1,
            "matched_entities": matched_cnt,
            "unmatched_entities": unmatched_cnt,
            "match_rate": round(match_rate, 6),
            "singleton_entities": singletons,
            "multi_candidate_entities": multi_cands,
            "S2_matches": s2_matches,
            "S3_matches": s3_matches,
            "S2_S3_conflict_entities": conflicts["conflict_entity_count"],
            "threshold": self.threshold,
            "margin": self.margin,
            "multiple_match_violations": 0,  # Exactly 1 row per S1 guaranteed
            "duplicate_candidate_targets": duplication_stats,
            "probability_min": float(entity_matches["top_probability"].min()),
            "probability_max": float(entity_matches["top_probability"].max()),
            "probability_mean": float(entity_matches["top_probability"].mean()),
            "margin_min": float(np.nanmin(margins)) if len(margins) > 0 else np.nan,
            "margin_max": float(np.nanmax(margins)) if len(margins) > 0 else np.nan,
            "margin_mean": float(np.nanmean(margins)) if len(margins) > 0 else np.nan,
            "decision_reason_counts": reason_counts
        }
        
        return entity_matches, stats
