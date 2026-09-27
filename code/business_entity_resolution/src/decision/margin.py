"""margin.py — Phase 6: Margin-based decision (best_score - second_best)."""
import pandas as pd


def apply_margin_filter(candidates_df: pd.DataFrame,
                        min_margin: float = 0.1) -> pd.DataFrame:
    """
    For each S1 entity, only keep a candidate match if the top score leads
    the second-best score by at least min_margin.
    Prevents accepting weak winners in crowded candidate sets.
    """
    def _filter_group(grp):
        grp = grp.sort_values("match_score", ascending=False)
        if len(grp) < 2:
            return grp
        top    = grp.iloc[0]["match_score"]
        second = grp.iloc[1]["match_score"]
        if top - second < min_margin:
            return grp.iloc[:0]   # reject all — no clear winner
        return grp[grp["match_score"] == top]

    return (
        candidates_df
        .groupby("source1_entity_id", group_keys=False)
        .apply(_filter_group)
        .reset_index(drop=True)
    )
