"""
ranking.py — Candidate ranking and margin calculation per reference entity.

Phase 5 Member 3 Component.
"""

import pandas as pd
import numpy as np


def rank_candidates(df: pd.DataFrame, prob_col: str = "prediction_probability") -> pd.DataFrame:
    """
    Rank candidate pairs per source1_entity_id by prediction probability descending.
    
    Adds columns:
    - candidate_rank: 1 for highest probability, 2 for second highest, etc.
    - top_probability: highest probability for that S1 entity
    - second_probability: second highest probability (or 0.0 for singletons)
    - margin: top_probability - second_probability (or top_probability for singletons)
    """
    df = df.copy()
    
    # Ensure source1_entity_id is a column
    if "source1_entity_id" not in df.columns:
        if df.index.name == "source1_entity_id":
            df = df.reset_index()
            
    # Sort by S1 entity ID and probability descending
    df = df.sort_values(["source1_entity_id", prob_col], ascending=[True, False])
    
    # Calculate candidate rank per S1 entity
    df["candidate_rank"] = df.groupby("source1_entity_id").cumcount() + 1
    
    # Vectorized / efficient group statistics
    stats = df.groupby("source1_entity_id")[prob_col].agg(
        top_probability=lambda s: s.iloc[0],
        second_probability=lambda s: s.iloc[1] if len(s) > 1 else 0.0
    ).reset_index()
    
    stats["margin"] = stats["top_probability"] - stats["second_probability"]
    
    # Merge stats back cleanly
    ranked_df = df.merge(stats, on="source1_entity_id", how="left")
    return ranked_df
