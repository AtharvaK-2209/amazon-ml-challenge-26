"""
name_features.py — Phase 4: Business name pairwise features.
Extracts character, token, structural, and acronym similarity features
between two business names. Handles missing values deterministically.
"""

from typing import Dict, Any, List, Optional
import math
import logging
import re
from rapidfuzz import fuzz, distance

logger = logging.getLogger(__name__)


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
    dist = distance.Levenshtein.distance(name1, name2)
    return max(0.0, 1.0 - (dist / max_len))


def jaccard_similarity(name1: str, name2: str) -> float:
    """Compute Jaccard similarity between token sets of two business names."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    if not n1 or not n2:
        return 0.0
    set1 = set(n1.lower().split())
    set2 = set(n2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return float(len(intersection) / len(union))


def levenshtein_distance(name1: str, name2: str) -> int:
    """Compute Levenshtein edit distance between two business names."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    if not n1 or not n2:
        return max(len(n1), len(n2))
    return int(distance.Levenshtein.distance(n1.lower(), n2.lower()))


def token_sort_ratio(name1: str, name2: str) -> float:
    """Compute token sort ratio similarity between two business names."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    if not n1 or not n2:
        return 0.0
    return float(fuzz.token_sort_ratio(n1, n2) / 100.0)


def partial_ratio(name1: str, name2: str) -> float:
    """Compute partial ratio similarity between two business names."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    if not n1 or not n2:
        return 0.0
    return float(fuzz.partial_ratio(n1, n2) / 100.0)


def acronym_match(name1: str, name2: str) -> bool:
    """Check if either name is an acronym of the other or if acronyms match."""
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    if not n1 or not n2:
        return False
    
    t1 = [w for w in re.split(r'\s+', n1) if w]
    t2 = [w for w in re.split(r'\s+', n2) if w]
    
    if not t1 or not t2:
        return False

    acronym1 = "".join(w[0] for w in t1).lower()
    acronym2 = "".join(w[0] for w in t2).lower()
    
    s1_clean = "".join(t1).lower()
    s2_clean = "".join(t2).lower()
    
    if len(s1_clean) >= 2 and s1_clean == acronym2:
        return True
    if len(s2_clean) >= 2 and s2_clean == acronym1:
        return True
    if len(acronym1) >= 2 and len(acronym2) >= 2 and acronym1 == acronym2:
        return True
        
    return False


def extract_name_features(
    name1: str,
    name2: str,
    suffix1: Optional[str] = None,
    suffix2: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute pairwise similarity features for business names (functional API).
    """
    n1 = _clean_str(name1)
    n2 = _clean_str(name2)
    
    features = {}
    
    features["name_s1_missing"] = 1.0 if not n1 else 0.0
    features["name_s2_missing"] = 1.0 if not n2 else 0.0
    
    if not n1 or not n2:
        features["name_ratio"] = 1.0 if (not n1 and not n2) else 0.0
        features["name_token_sort"] = features["name_ratio"]
        features["name_jaccard"] = features["name_ratio"]
        features["name_norm_edit"] = features["name_ratio"]
        features["name_len_ratio"] = 1.0 if (not n1 and not n2) else 0.0
        features["same_legal_suffix"] = 0.0
        features["name_wratio"] = features["name_ratio"]
        features["name_levenshtein"] = float(max(len(n1), len(n2)))
        features["name_token_set"] = features["name_ratio"]
        features["name_partial_ratio"] = features["name_ratio"]
        features["name_acronym_match"] = 0.0
        features["name_similarity"] = features["name_ratio"]
        return features

    features["name_ratio"] = float(fuzz.ratio(n1, n2) / 100.0)
    features["name_wratio"] = float(fuzz.WRatio(n1, n2) / 100.0)
    features["name_token_sort"] = float(fuzz.token_sort_ratio(n1, n2) / 100.0)
    features["name_token_set"] = float(fuzz.token_set_ratio(n1, n2) / 100.0)
    features["name_partial_ratio"] = float(fuzz.partial_ratio(n1, n2) / 100.0)
    features["name_norm_edit"] = _norm_edit_distance(n1, n2)
    features["name_levenshtein"] = float(distance.Levenshtein.distance(n1.lower(), n2.lower()))
    
    t1 = n1.split()
    t2 = n2.split()
    features["name_jaccard"] = _jaccard(t1, t2)
    features["name_acronym_match"] = float(acronym_match(n1, n2))
    
    len1, len2 = len(n1), len(n2)
    features["name_len_ratio"] = min(len1, len2) / max(len1, len2)
    
    if suffix1 is not None and suffix2 is not None:
        s1 = _clean_str(suffix1)
        s2 = _clean_str(suffix2)
        features["same_legal_suffix"] = 1.0 if (s1 and s2 and s1 == s2) else 0.0
    else:
        features["same_legal_suffix"] = 0.0

    features["name_similarity"] = features.get("name_wratio", features.get("name_token_sort", 0.0))
    return features


def extract_name_features_batch(
    name_pairs: list,
    suffix_pairs: Optional[list] = None
) -> list:
    """Extract features for multiple name pairs."""
    if suffix_pairs:
        return [
            extract_name_features(n1, n2, s1, s2)
            for (n1, n2), (s1, s2) in zip(name_pairs, suffix_pairs)
        ]
    return [extract_name_features(n1, n2) for n1, n2 in name_pairs]


class NameFeatureExtractor:
    """
    Feature extractor for business name pairs.
    Computes a comprehensive set of similarity features between business names.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.features = self.config.get('name_features', [
            'ratio',
            'wratio',
            'jaccard_similarity',
            'levenshtein_distance',
            'token_sort_ratio',
            'token_set_ratio',
            'partial_ratio',
            'acronym_match'
        ])
        logger.info(f"Initializing NameFeatureExtractor with {len(self.features)} features")
    
    def extract_features(self, name1: str, name2: str) -> Dict[str, float]:
        """Extract name similarity features for a name pair."""
        return extract_name_features(name1, name2)
    
    def extract_features_batch(self, name_pairs: list) -> list:
        """Extract features for multiple name pairs."""
        return extract_name_features_batch(name_pairs)
