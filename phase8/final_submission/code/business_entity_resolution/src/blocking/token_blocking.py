"""token_blocking.py — B02: Sorted Neighbourhood blocking on normalised names."""
import pandas as pd
from src.config import BLOCKING


def sorted_neighbourhood_block(s1: pd.DataFrame, s2s3: pd.DataFrame,
                                name_col: str = "normalized_name",
                                window: int | None = None) -> pd.DataFrame:
    """
    Sort all records by normalised name; slide a window of size W.
    Returns candidate pairs DataFrame: [source1_entity_id, candidate_entity_id]
    """
    W = window or BLOCKING["sorted_neighbourhood_window"]

    all_records = pd.concat([
        s1[["entity_id", name_col, "country"]].assign(_is_s1=True),
        s2s3[["entity_id", name_col, "country"]].assign(_is_s1=False),
    ], ignore_index=True).sort_values(name_col)

    pairs = []
    records = all_records.to_dict("records")
    for i, rec in enumerate(records):
        if not rec["_is_s1"]:
            continue
        window_slice = records[max(0, i - W): i + W + 1]
        for other in window_slice:
            if other["entity_id"] == rec["entity_id"]:
                continue
            if other["_is_s1"]:
                continue
            if other["country"] != rec["country"]:
                continue
            pairs.append((rec["entity_id"], other["entity_id"]))

    return pd.DataFrame(pairs, columns=["source1_entity_id","candidate_entity_id"])
