"""
name_features.py — Phase 4: Business name pairwise features.
Implements the 11 exact requested business name similarity features.
"""

from typing import Dict, Any, Optional
import math
from rapidfuzz import fuzz, distance
import numpy as np

def _clean_str(s: Any) -> str:
    """Safely cast to string and clean."""
    if s is None:
        return ""
    try:
        if math.isnan(s):
            return ""
    except TypeError:
        pass
    
    val = str(s).strip()
    if val.lower() in ("nan", "none", "<na>"):
        return ""
    return val

def _jaccard(tokens1: list, tokens2: list) -> float:
    """Token-level Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def generate_name_features(
    s1_name: str,
    candidate_name: str,
    tfidf_vectorizer=None,
    char_vectorizer=None
) -> Dict[str, float]:
    """
    Compute pairwise similarity features for business names.
    
    Args:
        s1_name: Phase 2 normalized name for S1.
        candidate_name: Phase 2 normalized name for candidate (S2/S3).
        tfidf_vectorizer: Pre-fit sklearn TfidfVectorizer (word-level).
        char_vectorizer: Pre-fit sklearn TfidfVectorizer (char-level).
        
    Returns:
        Dictionary of 11 required feature names to deterministic float values.
    """
    n1 = _clean_str(s1_name)
    n2 = _clean_str(candidate_name)
    
    features = {}
    
    # 1. name_levenshtein
    features["name_levenshtein"] = float(distance.Levenshtein.distance(n1, n2))
    
    if not n1 and not n2:
        features["name_fuzz_ratio"] = 100.0
        features["name_wratio"] = 100.0
        features["name_token_sort_ratio"] = 100.0
        features["name_token_set_ratio"] = 100.0
        features["name_jaccard"] = 1.0
        features["name_tfidf_cosine"] = 1.0
        features["name_char_cosine"] = 1.0
        features["name_length_diff"] = 0.0
        features["name_length_ratio"] = 1.0
        features["name_token_count_diff"] = 0.0
        return features
        
    if not n1 or not n2:
        features["name_fuzz_ratio"] = 0.0
        features["name_wratio"] = 0.0
        features["name_token_sort_ratio"] = 0.0
        features["name_token_set_ratio"] = 0.0
        features["name_jaccard"] = 0.0
        features["name_tfidf_cosine"] = 0.0
        features["name_char_cosine"] = 0.0
        features["name_length_diff"] = float(abs(len(n1) - len(n2)))
        features["name_length_ratio"] = 0.0
        
        t1, t2 = n1.split(), n2.split()
        features["name_token_count_diff"] = float(abs(len(t1) - len(t2)))
        return features

    # RapidFuzz returns [0, 100]
    features["name_fuzz_ratio"] = fuzz.ratio(n1, n2)
    features["name_wratio"] = fuzz.WRatio(n1, n2)
    features["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2)
    features["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2)
    
    # Jaccard
    t1, t2 = n1.split(), n2.split()
    features["name_jaccard"] = _jaccard(t1, t2)
    
    # Length features
    len1, len2 = len(n1), len(n2)
    features["name_length_diff"] = float(abs(len1 - len2))
    features["name_length_ratio"] = float(min(len1, len2) / max(len1, len2))
    features["name_token_count_diff"] = float(abs(len(t1) - len(t2)))
    
    # TF-IDF Cosine (Word)
    features["name_tfidf_cosine"] = 0.0
    if tfidf_vectorizer is not None:
        try:
            vecs = tfidf_vectorizer.transform([n1, n2])
            # v1 * v2^T since vectors are L2 normalized by default
            cos = (vecs[0] @ vecs[1].T).toarray()[0][0]
            features["name_tfidf_cosine"] = float(cos)
        except Exception:
            features["name_tfidf_cosine"] = 0.0

    # TF-IDF Cosine (Char)
    features["name_char_cosine"] = 0.0
    if char_vectorizer is not None:
        try:
            vecs = char_vectorizer.transform([n1, n2])
            cos = (vecs[0] @ vecs[1].T).toarray()[0][0]
            features["name_char_cosine"] = float(cos)
        except Exception:
            features["name_char_cosine"] = 0.0
            
    return features
