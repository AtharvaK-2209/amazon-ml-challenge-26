"""
address_features.py — Phase 4 Part 2: Address Feature Engineering

Computes a comprehensive set of pairwise address similarity and comparison features:
- RapidFuzz similarity measures (address_fuzz_ratio, address_wratio, address_token_sort_ratio, address_token_set_ratio)
- Jaccard token similarity (address_jaccard)
- Word-level TF-IDF cosine similarity (address_tfidf_cosine)
- Character n-gram TF-IDF cosine similarity (address_char_cosine)
- Numeric token overlap (address_numeric_overlap)
- Component comparison features (house_number_match, postal_match, city_match, state_match) using convention:
    1  = exact match
    0  = mismatch
   -1  = missing / unknown

Designed for scalable performance across training, validation, and test candidate pairs.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import logging
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.address_parser import parse_address
from src.config import NORM

logger = logging.getLogger(__name__)

# Standard street type keywords for comparison
STREET_TYPES: Set[str] = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway",
    "way", "place", "court", "circle", "plaza", "square", "parkway", "alley",
    "rd", "st", "ave", "blvd", "dr", "ln", "hwy", "way", "pl", "ct", "cir"
}

STREET_TYPE_EXPANSIONS = NORM.get("address_abbreviation_expansions", {})


def jaccard_similarity(address1: str, address2: str) -> float:
    """
    Compute Jaccard token similarity between two address strings.
    
    Args:
        address1: First address string
        address2: Second address string
        
    Returns:
        Jaccard similarity score in [0.0, 1.0]
    """
    if not address1 or not address2:
        return 0.0
    s1 = set(address1.lower().split())
    s2 = set(address2.lower().split())
    if not s1 or not s2:
        return 0.0
    intersection = s1 & s2
    union = s1 | s2
    return float(len(intersection) / len(union))


def levenshtein_distance(address1: str, address2: str) -> int:
    """
    Compute Levenshtein edit distance between two addresses.
    """
    if not address1 or not address2:
        return max(len(address1 or ""), len(address2 or ""))
    return int(distance.Levenshtein.distance(address1.lower(), address2.lower()))


def _extract_numbers(address: str) -> Set[str]:
    """Extract all distinct numeric tokens from address string."""
    if not address:
        return set()
    return set(re.findall(r"\d+", str(address)))


def numeric_match(address1: str, address2: str) -> bool:
    """
    Check if addresses share at least one numeric token.
    """
    nums1 = _extract_numbers(address1)
    nums2 = _extract_numbers(address2)
    if not nums1 or not nums2:
        return False
    return len(nums1 & nums2) > 0


def numeric_token_overlap(address1: str, address2: str) -> float:
    """
    Compute Jaccard overlap of numeric tokens in two addresses.
    
    Args:
        address1: First address
        address2: Second address
        
    Returns:
        Numeric token Jaccard overlap in [0.0, 1.0]
    """
    nums1 = _extract_numbers(address1)
    nums2 = _extract_numbers(address2)
    if not nums1 or not nums2:
        return 0.0
    intersection = nums1 & nums2
    union = nums1 | nums2
    return float(len(intersection) / len(union))


def _extract_street_types(address: str) -> Set[str]:
    """Extract and expand street type keywords from address."""
    if not address:
        return set()
    tokens = address.lower().replace(",", " ").split()
    st_types = set()
    for tok in tokens:
        tok_clean = tok.rstrip(".")
        if tok_clean in STREET_TYPES:
            canonical = STREET_TYPE_EXPANSIONS.get(tok_clean, tok_clean)
            st_types.add(canonical)
    return st_types


def street_type_match(address1: str, address2: str) -> bool:
    """Check if addresses have matching street types."""
    st1 = _extract_street_types(address1)
    st2 = _extract_street_types(address2)
    if not st1 or not st2:
        return False
    return len(st1 & st2) > 0


def _compare_component(val1: Optional[str], val2: Optional[str]) -> int:
    """
    Helper for component matching following project convention:
        1  = exact match (both non-empty & equal)
        0  = mismatch (both non-empty & different)
       -1  = missing / unknown (either or both missing)
    """
    if not val1 or not val2 or not str(val1).strip() or not str(val2).strip():
        return -1
    v1 = str(val1).lower().strip()
    v2 = str(val2).lower().strip()
    return 1 if v1 == v2 else 0


def city_match(address1: str, address2: str, country1: str = "", country2: str = "") -> int:
    """
    Check if parsed cities match between two addresses.
    Returns: 1 = match, 0 = mismatch, -1 = missing
    """
    p1 = parse_address(address1, country1)
    p2 = parse_address(address2, country2)
    return _compare_component(p1.get("city"), p2.get("city"))


def state_match(address1: str, address2: str, country1: str = "", country2: str = "") -> int:
    """
    Check if parsed states match between two addresses.
    Returns: 1 = match, 0 = mismatch, -1 = missing
    """
    p1 = parse_address(address1, country1)
    p2 = parse_address(address2, country2)
    return _compare_component(p1.get("state"), p2.get("state"))


def postal_match(address1: str, address2: str, country1: str = "", country2: str = "") -> int:
    """
    Check if parsed postal/PIN codes match between two addresses.
    Returns: 1 = match, 0 = mismatch, -1 = missing
    """
    p1 = parse_address(address1, country1)
    p2 = parse_address(address2, country2)
    return _compare_component(p1.get("postal_code"), p2.get("postal_code"))


def house_number_match(address1: str, address2: str, country1: str = "", country2: str = "") -> int:
    """
    Check if parsed house numbers match between two addresses.
    Returns: 1 = match, 0 = mismatch, -1 = missing
    """
    p1 = parse_address(address1, country1)
    p2 = parse_address(address2, country2)
    
    h1 = p1.get("house_number")
    h2 = p2.get("house_number")
    
    # Fallback to leading number if parse_address yielded None
    if not h1 and address1:
        m = re.match(r"^\s*(\d+[a-zA-Z]?)\b", address1)
        if m:
            h1 = m.group(1)
    if not h2 and address2:
        m = re.match(r"^\s*(\d+[a-zA-Z]?)\b", address2)
        if m:
            h2 = m.group(1)
            
    return _compare_component(h1, h2)


class AddressFeatureExtractor:
    """
    Phase 4 - Part 2 Address Feature Extractor class.
    Generates high-quality address features for candidate entity pairs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AddressFeatureExtractor with optional configuration and TF-IDF vectorizers.
        """
        self.config = config or {}
        
        # Word TF-IDF vectorizer configuration
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=25000,
            token_pattern=r"(?u)\b\w+\b"
        )
        
        # Character n-gram TF-IDF vectorizer configuration
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=1,
            max_features=25000
        )
        
        self.vectorizers_fitted = False
        logger.info("Initializing AddressFeatureExtractor (Phase 4 - Part 2)")

    def fit_vectorizers(self, corpus: List[str]) -> "AddressFeatureExtractor":
        """
        Fit word and char-level TF-IDF vectorizers on an address corpus.
        """
        clean_corpus = [str(a or "").strip() for a in corpus if str(a or "").strip()]
        if not clean_corpus:
            clean_corpus = ["dummy address"]
            
        self.word_vectorizer.fit(clean_corpus)
        self.char_vectorizer.fit(clean_corpus)
        self.vectorizers_fitted = True
        return self

    def extract_features(
        self,
        address1: str,
        address2: str,
        country1: str = "",
        country2: str = ""
    ) -> Dict[str, float]:
        """
        Extract address features for a single address pair.
        
        Args:
            address1: First address string
            address2: Second address string
            country1: Country for entity 1
            country2: Country for entity 2
            
        Returns:
            Dictionary mapping feature names to numerical values.
        """
        a1 = str(address1 or "").strip()
        a2 = str(address2 or "").strip()

        features: Dict[str, float] = {}

        # 1. RapidFuzz Similarities [0.0, 1.0]
        if not a1 or not a2:
            features['address_fuzz_ratio'] = 0.0
            features['address_wratio'] = 0.0
            features['address_token_sort_ratio'] = 0.0
            features['address_token_set_ratio'] = 0.0
            features['address_similarity'] = 0.0
            features['address_ratio'] = 0.0
            features['address_partial_ratio'] = 0.0
            features['address_jaccard'] = 0.0
            features['address_levenshtein'] = float(max(len(a1), len(a2)))
            features['address_tfidf_cosine'] = 0.0
            features['address_char_cosine'] = 0.0
        else:
            fuzz_ratio = float(fuzz.ratio(a1, a2) / 100.0)
            wratio = float(fuzz.WRatio(a1, a2) / 100.0)
            sort_ratio = float(fuzz.token_sort_ratio(a1, a2) / 100.0)
            set_ratio = float(fuzz.token_set_ratio(a1, a2) / 100.0)
            partial_ratio = float(fuzz.partial_ratio(a1, a2) / 100.0)

            features['address_fuzz_ratio'] = fuzz_ratio
            features['address_wratio'] = wratio
            features['address_token_sort_ratio'] = sort_ratio
            features['address_token_set_ratio'] = set_ratio
            
            # Backward-compatible aliases
            features['address_similarity'] = wratio
            features['address_ratio'] = fuzz_ratio
            features['address_partial_ratio'] = partial_ratio
            features['address_jaccard'] = jaccard_similarity(a1, a2)
            features['address_levenshtein'] = float(levenshtein_distance(a1, a2))

            # 2. TF-IDF & Character Cosine Similarities
            try:
                if self.vectorizers_fitted:
                    w_vecs = self.word_vectorizer.transform([a1, a2])
                    c_vecs = self.char_vectorizer.transform([a1, a2])
                    w_cos = float(w_vecs[0].dot(w_vecs[1].T).toarray()[0][0])
                    c_cos = float(c_vecs[0].dot(c_vecs[1].T).toarray()[0][0])
                else:
                    # Fit dynamically on pair
                    vec_w = TfidfVectorizer(ngram_range=(1, 2)).fit([a1, a2])
                    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit([a1, a2])
                    w_vecs = vec_w.transform([a1, a2])
                    c_vecs = vec_c.transform([a1, a2])
                    w_cos = float(w_vecs[0].dot(w_vecs[1].T).toarray()[0][0])
                    c_cos = float(c_vecs[0].dot(c_vecs[1].T).toarray()[0][0])
            except Exception:
                w_cos = features['address_jaccard']
                c_cos = features['address_token_sort_ratio']

            features['address_tfidf_cosine'] = float(np.clip(w_cos, 0.0, 1.0))
            features['address_char_cosine'] = float(np.clip(c_cos, 0.0, 1.0))

        # 3. Numeric Overlap
        features['address_numeric_overlap'] = numeric_token_overlap(a1, a2)
        features['address_numeric_match'] = 1.0 if numeric_match(a1, a2) else 0.0
        features['address_street_type_match'] = 1.0 if street_type_match(a1, a2) else 0.0

        # 4. Component Comparison Features (1 = match, 0 = mismatch, -1 = missing)
        p1 = parse_address(a1, country1)
        p2 = parse_address(a2, country2)

        h_match = house_number_match(a1, a2, country1, country2)
        p_match = postal_match(a1, a2, country1, country2)
        c_match = city_match(a1, a2, country1, country2)
        s_match = state_match(a1, a2, country1, country2)

        features['house_number_match'] = float(h_match)
        features['postal_match'] = float(p_match)
        features['city_match'] = float(c_match)
        features['state_match'] = float(s_match)

        # Boolean aliases for legacy consumers
        features['same_house_number'] = 1.0 if h_match == 1 else 0.0
        features['same_postal_code'] = 1.0 if p_match == 1 else 0.0
        features['same_city'] = 1.0 if c_match == 1 else 0.0
        features['same_state'] = 1.0 if s_match == 1 else 0.0
        features['address_city_match'] = features['same_city']

        return features

    def extract_features_batch(self, address_pairs: list) -> list:
        """
        Extract features for a list of address pairs.
        """
        # Collect all unique address strings to fit TF-IDF if not fitted
        if not self.vectorizers_fitted:
            all_addrs = []
            for pair in address_pairs:
                a1 = pair[0] if len(pair) > 0 else ""
                a2 = pair[1] if len(pair) > 1 else ""
                all_addrs.extend([str(a1 or ""), str(a2 or "")])
            self.fit_vectorizers(all_addrs)

        results = []
        for pair in address_pairs:
            a1 = pair[0] if len(pair) > 0 else ""
            a2 = pair[1] if len(pair) > 1 else ""
            c1 = pair[2] if len(pair) > 2 else ""
            c2 = pair[3] if len(pair) > 3 else ""
            results.append(self.extract_features(a1, a2, c1, c2))
        return results

    def extract_features_dataframe(
        self,
        df: pd.DataFrame,
        addr1_col: str = "normalized_address_s1",
        addr2_col: str = "normalized_address_cand",
        country1_col: str = "country_s1",
        country2_col: str = "country_cand"
    ) -> pd.DataFrame:
        """
        Scalable vector-accelerated feature extraction for a pandas DataFrame of candidate pairs.
        """
        if df.empty:
            return pd.DataFrame()

        # Fit vectorizers on unique addresses in df
        addrs1 = df[addr1_col].fillna("").astype(str).tolist() if addr1_col in df.columns else [""] * len(df)
        addrs2 = df[addr2_col].fillna("").astype(str).tolist() if addr2_col in df.columns else [""] * len(df)

        c1_list = df[country1_col].fillna("").astype(str).tolist() if country1_col in df.columns else [""] * len(df)
        c2_list = df[country2_col].fillna("").astype(str).tolist() if country2_col in df.columns else [""] * len(df)

        unique_addrs = list(set(addrs1 + addrs2))
        if not self.vectorizers_fitted:
            self.fit_vectorizers(unique_addrs)

        # Precompute TF-IDF matrices for unique addresses
        w_mat = self.word_vectorizer.transform(unique_addrs)
        c_mat = self.char_vectorizer.transform(unique_addrs)
        addr_to_idx = {addr: i for i, addr in enumerate(unique_addrs)}

        rows = []
        for a1, a2, c1, c2 in zip(addrs1, addrs2, c1_list, c2_list):
            feats = self.extract_features(a1, a2, c1, c2)
            
            # Fast matrix dot-product lookup for TF-IDF similarities
            idx1, idx2 = addr_to_idx.get(a1), addr_to_idx.get(a2)
            if idx1 is not None and idx2 is not None and a1 and a2:
                w_sim = float(w_mat[idx1].dot(w_mat[idx2].T).toarray()[0][0])
                c_sim = float(c_mat[idx1].dot(c_mat[idx2].T).toarray()[0][0])
                feats['address_tfidf_cosine'] = float(np.clip(w_sim, 0.0, 1.0))
                feats['address_char_cosine'] = float(np.clip(c_sim, 0.0, 1.0))

            rows.append(feats)

        res_df = pd.DataFrame(rows)
        return res_df.fillna(0.0)


# Standalone function helpers
def extract_address_features(
    address1: str,
    address2: str,
    country1: str = "",
    country2: str = ""
) -> Dict[str, float]:
    """
    Extract full set of pairwise address features for two addresses.
    """
    extractor = AddressFeatureExtractor()
    return extractor.extract_features(address1, address2, country1, country2)
