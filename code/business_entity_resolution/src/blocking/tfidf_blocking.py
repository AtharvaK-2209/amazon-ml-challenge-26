"""tfidf_blocking.py — B01: TF-IDF char-ngram blocking (primary strategy)."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import BLOCKING


def tfidf_block(s1: pd.DataFrame, s2s3: pd.DataFrame,
                name_col: str = "normalized_name",
                top_k: int | None = None,
                min_score: float | None = None) -> pd.DataFrame:
    """
    Build TF-IDF char-3gram index over S2+S3 normalised names.
    For each S1 record, retrieve top-K candidates by cosine similarity.
    Partitioned by country (B03).

    Returns candidate pairs DataFrame: [source1_entity_id, candidate_entity_id, tfidf_score]
    """
    K     = top_k    or BLOCKING["tfidf_top_k"]
    SCORE = min_score or BLOCKING["tfidf_min_score"]
    ngram = BLOCKING["tfidf_ngram_range"]
    analyzer = BLOCKING["tfidf_analyzer"]

    all_pairs = []
    countries = s1["country"].unique()

    for country in countries:
        s1_c    = s1[s1["country"] == country].reset_index(drop=True)
        s2s3_c  = s2s3[s2s3["country"] == country].reset_index(drop=True)

        if s1_c.empty or s2s3_c.empty:
            continue

        corpus_cand = s2s3_c[name_col].fillna("").tolist()
        corpus_s1   = s1_c[name_col].fillna("").tolist()

        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram)
        mat_cand = vec.fit_transform(corpus_cand)
        mat_s1   = vec.transform(corpus_s1)

        # Process in batches to avoid memory spikes
        BATCH = 500
        for i in range(0, len(s1_c), BATCH):
            batch_mat  = mat_s1[i:i+BATCH]
            sims       = cosine_similarity(batch_mat, mat_cand)  # (batch, |cand|)
            batch_rows = s1_c.iloc[i:i+BATCH]

            for j, (_, s1_row) in enumerate(batch_rows.iterrows()):
                scores = sims[j]
                top_idx = np.argsort(scores)[::-1][:K]
                for idx in top_idx:
                    if scores[idx] >= SCORE:
                        all_pairs.append((
                            s1_row["entity_id"],
                            s2s3_c.iloc[idx]["entity_id"],
                            round(float(scores[idx]), 4),
                        ))

    return pd.DataFrame(all_pairs,
                        columns=["source1_entity_id","candidate_entity_id","tfidf_score"])
