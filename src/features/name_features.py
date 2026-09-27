"""
name_features.py — Phase 4: Business name pairwise features.
Extracts a small, purposeful set of character and token similarity features
between two business names. Handles missing values deterministically.
"""

from typing import Dict, Any, Optional
import math
from rapidfuzz import fuzz

def _clean_str(s: Any) -> str:
    """Safely cast to string and clean."""
    if s is None:
        return ""
    # Treat pd.NA or np.nan as empty string
    try:
        if math.isnan(s):
            return ""
    except TypeError:
        pass
    
    val = str(s).strip()
    # Treat "nan" or "none" strings (which can happen in pandas) as empty
    if val.lower() in ("nan", "none", "<na>"):
        return ""
    return val

def _jaccard(tokens1: list, tokens2: list) -> float:
    """Jaccard similarity between two token lists."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def _norm_edit_distance(name1: str, name2: str) -> float:
    """Normalized edit distance: 1.0 - (edit_distance / max_len)."""
    if not name1 and not name2:
        return 1.0
    if not name1 or not name2:
        return 0.0
    max_len = max(len(name1), len(name2))
    distance = fuzz.distance(name1, name2)
    return max(0.0, 1.0 - (distance / max_len))

def extract_name_features(
    name1: str,
    name2: str,
    suffix1: Optional[str] = None,
    suffix2: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute pairwise similarity features for business names.
    
    Args:
        name1: S1 normalized business name
        name2: S2/S3 normalized business name
        suffix1: Extracted legal suffix for S1 (optional)
        suffix2: Extracted legal suffix for S2/S3 (optional)
        
    Returns:
        Dictionary of feature names to deterministic float values.
    """
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    
    features = {}
    
    # Missing value flags
    features["name_s1_missing"] = 1.0 if not n1 else 0.0
    features["name_s2_missing"] = 1.0 if not n2 else 0.0
    
    if not n1 or not n2:
        # Fallback values when one or both are missing
        features["name_ratio"] = 1.0 if (not n1 and not n2) else 0.0
        features["name_token_sort"] = features["name_ratio"]
        features["name_jaccard"] = features["name_ratio"]
        features["name_norm_edit"] = features["name_ratio"]
        features["name_len_ratio"] = 1.0 if (not n1 and not n2) else 0.0
        features["same_legal_suffix"] = 0.0
        return features

    # Character similarities (using RapidFuzz for speed)
    # fuzz.ratio is standard Levenshtein-based similarity
    features["name_ratio"] = fuzz.ratio(n1, n2) / 100.0
    
    # fuzz.token_sort_ratio handles token reordering (e.g. "ABC Corp" vs "Corp ABC")
    features["name_token_sort"] = fuzz.token_sort_ratio(n1, n2) / 100.0
    
    features["name_norm_edit"] = _norm_edit_distance(n1, n2)
    
    # Token features
    t1 = n1.split()
    t2 = n2.split()
    features["name_jaccard"] = _jaccard(t1, t2)
    
    # Structural/length features
    # Min length / Max length. Prevents division by zero.
    len1, len2 = len(n1), len(n2)
    features["name_len_ratio"] = min(len1, len2) / max(len1, len2)
    
    # Suffix features
    if suffix1 is not None and suffix2 is not None:
        s1 = _clean_str(suffix1)
        s2 = _clean_str(suffix2)
        if s1 and s2:
            features["same_legal_suffix"] = 1.0 if s1 == s2 else 0.0
        else:
            features["same_legal_suffix"] = 0.0
    else:
        features["same_legal_suffix"] = 0.0
        
    return features

def extract_name_features_batch(
    name_pairs: list,
    suffix_pairs: Optional[list] = None
) -> list:
    """
    Extract features for multiple name pairs.
    
    Args:
        name_pairs: List of (name1, name2) tuples
        suffix_pairs: Optional list of (suffix1, suffix2) tuples
        
    Returns:
        List of feature dictionaries
    """
    if suffix_pairs:
        return [
            extract_name_features(n1, n2, s1, s2)
            for (n1, n2), (s1, s2) in zip(name_pairs, suffix_pairs)
        ]
    return [extract_name_features(n1, n2) for n1, n2 in name_pairs]
