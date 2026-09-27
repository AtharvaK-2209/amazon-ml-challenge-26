"""
conflict.py — Phase 6 Member 2: S2/S3 Conflict Analysis & Target Duplication Analytics.

Analyzes candidate source conflicts (entities with strong candidates in both S2 and S3)
and duplicate target matches (candidate entity IDs matched to multiple S1 reference entities).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_s2_s3_conflicts(df: pd.DataFrame, threshold: float = 0.70) -> Dict[str, Any]:
    """
    Analyze entities that have candidates from BOTH S2 and S3 with probability >= threshold.
    
    Returns metrics dict.
    """
    strong_cands = df[df["prediction_probability"] >= threshold]
    grp_sources = strong_cands.groupby("source1_entity_id")["candidate_source"].apply(lambda s: set(s))
    
    conflict_s1_ids = grp_sources[grp_sources.apply(lambda s: "S2" in s and "S3" in s)].index.tolist()
    
    # Conflict details
    conflict_df = df[df["source1_entity_id"].isin(conflict_s1_ids)].copy()
    
    return {
        "conflict_entity_count": len(conflict_s1_ids),
        "conflict_s1_ids": conflict_s1_ids,
        "conflict_pair_count": len(conflict_df)
    }


def analyze_target_duplications(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze whether the same candidate_entity_id is matched to multiple source1_entity_id values.
    
    matches_df: DataFrame containing entity-level decisions where decision == 'MATCH'.
    Returns analysis dictionary.
    """
    matched_only = matches_df[matches_df["decision"] == "MATCH"]
    if matched_only.empty:
        return {
            "unique_candidates_matched_to_one_s1": 0,
            "candidates_matched_to_multiple_s1": 0,
            "max_s1_sharing_one_candidate": 0
        }
        
    cand_counts = matched_only["matched_candidate_id"].value_counts()
    
    single_matched = int((cand_counts == 1).sum())
    multi_matched = int((cand_counts > 1).sum())
    max_sharing = int(cand_counts.max()) if not cand_counts.empty else 0
    
    return {
        "unique_candidates_matched_to_one_s1": single_matched,
        "candidates_matched_to_multiple_s1": multi_matched,
        "max_s1_sharing_one_candidate": max_sharing
    }
