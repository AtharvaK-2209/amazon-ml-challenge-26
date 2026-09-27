"""
faiss_blocking.py — B01 (FAISS variant): Dense ANN blocking for large-scale candidate generation.
Uses sentence-transformers embeddings + FAISS index.
Intended for SageMaker / EC2 CPU runs where TF-IDF memory limits are hit.

Install: pip install faiss-cpu sentence-transformers
"""
import pandas as pd
import numpy as np
from src.config import BLOCKING


def faiss_block(s1: pd.DataFrame, s2s3: pd.DataFrame,
                name_col: str = "normalized_name",
                model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                top_k: int | None = None) -> pd.DataFrame:
    """
    Embed normalised names with a multilingual sentence-transformer.
    Build a FAISS flat-L2 index over S2+S3 embeddings.
    Query with each S1 embedding to get top-K candidates.

    Returns candidate pairs DataFrame: [source1_entity_id, candidate_entity_id, faiss_score]
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "faiss-cpu and sentence-transformers are required for faiss_blocking. "
            "Run: pip install faiss-cpu sentence-transformers"
        ) from e

    K = top_k or BLOCKING["tfidf_top_k"]
    model = SentenceTransformer(model_name)

    all_pairs = []
    for country in s1["country"].unique():
        s1_c   = s1[s1["country"] == country].reset_index(drop=True)
        cand_c = s2s3[s2s3["country"] == country].reset_index(drop=True)
        if s1_c.empty or cand_c.empty:
            continue

        emb_cand = model.encode(cand_c[name_col].fillna("").tolist(),
                                batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True).astype(np.float32)
        emb_s1   = model.encode(s1_c[name_col].fillna("").tolist(),
                                batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True).astype(np.float32)

        dim   = emb_cand.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner-product = cosine for normalised vecs
        index.add(emb_cand)

        distances, indices = index.search(emb_s1, K)

        for i, s1_row in s1_c.iterrows():
            for rank, (idx, score) in enumerate(zip(indices[i], distances[i])):
                if idx >= 0:
                    all_pairs.append((
                        s1_row["entity_id"],
                        cand_c.iloc[idx]["entity_id"],
                        round(float(score), 4),
                    ))

    return pd.DataFrame(all_pairs,
                        columns=["source1_entity_id","candidate_entity_id","faiss_score"])
