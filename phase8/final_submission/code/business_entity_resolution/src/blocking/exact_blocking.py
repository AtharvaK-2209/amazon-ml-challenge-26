"""exact_blocking.py — B05: First-token exact match blocking."""
import pandas as pd
from src.config import BLOCKING


def exact_block(s1: pd.DataFrame, s2s3: pd.DataFrame,
                name_col: str = "normalized_name") -> pd.DataFrame:
    """
    Group records by their first normalised name token.
    Returns candidate pairs DataFrame: [source1_entity_id, candidate_entity_id]
    """
    min_len = BLOCKING.get("address_min_token_len", 4)

    def first_token(name):
        toks = str(name).split()
        return toks[0] if toks and len(toks[0]) >= min_len else None

    s1 = s1.copy()
    s2s3 = s2s3.copy()
    s1["_key"]   = s1[name_col].apply(first_token)
    s2s3["_key"] = s2s3[name_col].apply(first_token)

    s1_keyed   = s1.dropna(subset=["_key"])
    s2s3_keyed = s2s3.dropna(subset=["_key"])

    merged = s1_keyed[["entity_id","_key","country"]].merge(
        s2s3_keyed[["entity_id","_key","country"]],
        on=["_key", "country"], suffixes=("_s1","_cand")
    )
    return merged[["entity_id_s1","entity_id_cand"]].rename(
        columns={"entity_id_s1":"source1_entity_id","entity_id_cand":"candidate_entity_id"}
    )
