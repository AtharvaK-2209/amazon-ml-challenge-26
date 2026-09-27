"""
decision.py — Unified Phase 5 decision engine for entity matching.

Combines candidate ranking, probability thresholding, multi-candidate selection, and margin requirements.
"""

import pandas as pd
import numpy as np
from src.decision.ranking import rank_candidates


def make_entity_decisions(
    df: pd.DataFrame,
    threshold: float = 0.85,
    margin: float = 0.00,
    prob_col: str = "prediction_probability",
    single_match_only: bool = False
) -> pd.DataFrame:
    """
    Apply entity-level decision logic to candidate predictions.
    
    Parameters:
    - df: DataFrame containing predictions
    - threshold: minimum prediction probability required for match
    - margin: minimum required gap between top candidate probability and second candidate probability
    - prob_col: probability column name
    - single_match_only: if True, restrict decisions to top-1 candidate only.
    
    Returns DataFrame with 'decision' (0 or 1) and 'decision_reason'.
    """
    ranked_df = rank_candidates(df, prob_col=prob_col)
    
    if single_match_only:
        is_top1 = ranked_df["candidate_rank"] == 1
        pass_thresh = ranked_df[prob_col] >= threshold
        pass_margin = ranked_df["margin"] >= margin
        
        decision = (is_top1 & pass_thresh & pass_margin).astype(int)
        
        reasons = np.where(
            ~is_top1,
            "REJECT_NOT_TOP_RANK",
            np.where(
                ~pass_thresh,
                "REJECT_PROB_BELOW_THRESHOLD",
                np.where(
                    ~pass_margin,
                    "REJECT_MARGIN_BELOW_THRESHOLD",
                    "ACCEPT_TOP1_MATCH"
                )
            )
        )
    else:
        pass_thresh = ranked_df[prob_col] >= threshold
        if margin > 0.0:
            pass_margin = ranked_df["margin"] >= margin
        else:
            pass_margin = pd.Series(True, index=ranked_df.index)
            
        decision = (pass_thresh & pass_margin).astype(int)
        
        reasons = np.where(
            ~pass_thresh,
            "REJECT_PROB_BELOW_THRESHOLD",
            np.where(
                ~pass_margin,
                "REJECT_MARGIN_GAP_BELOW_THRESHOLD",
                "ACCEPT_MATCH"
            )
        )
    
    ranked_df["decision"] = decision
    ranked_df["decision_reason"] = reasons
    return ranked_df
