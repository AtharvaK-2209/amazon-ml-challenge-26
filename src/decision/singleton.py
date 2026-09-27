"""singleton.py — Phase 6: Singleton identification (no-match entities)."""
import pandas as pd


def build_full_submission(matched_df: pd.DataFrame,
                          all_s1_ids: pd.Series) -> pd.DataFrame:
    """
    Ensure every Source-1 entity appears in the output.
    Entities with no matches get an empty matched_entity_ids string.

    matched_df:   [source1_entity_id, matched_entity_ids]
    all_s1_ids:   Series of all S1 entity_ids in the test set
    Returns:      [source1_entity_id, matched_entity_ids] — one row per S1
    """
    full = pd.DataFrame({"source1_entity_id": all_s1_ids})
    full = full.merge(matched_df, on="source1_entity_id", how="left")
    full["matched_entity_ids"] = full["matched_entity_ids"].fillna("")
    return full[["source1_entity_id","matched_entity_ids"]]
