"""
address_features.py — Address feature engineering for entity resolution.

Computes pairwise address similarity features, structural/numeric features,
and component-level comparisons (house number, postal code, state, city)
for candidate entity pairs.
"""

from typing import Dict, Any, List, Optional, Set
import logging
import re
from functools import lru_cache
from rapidfuzz import fuzz, distance

from src.preprocessing.address_parser import parse_address
from src.config import NORM

logger = logging.getLogger(__name__)

# Standard street type keywords for comparison
STREET_TYPES: Set[str] = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway",
    "way", "place", "court", "circle", "plaza", "square", "parkway", "alley",
    "rd", "st", "ave", "blvd", "dr", "ln", "hwy", "way", "pl", "ct", "cir"
}

# Mapping of normalized street types for robust matching
STREET_TYPE_EXPANSIONS = NORM.get("address_abbreviation_expansions", {})


def jaccard_similarity(address1: str, address2: str) -> float:
    """
    Compute Jaccard similarity between token sets of two addresses.
    
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
    
    Args:
        address1: First address string
        address2: Second address string
        
    Returns:
        Edit distance (number of operations)
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
    Check if addresses have matching numeric components.
    
    Args:
        address1: First address
        address2: Second address
        
    Returns:
        True if numeric components exist and have non-empty overlap
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
    """
    Check if addresses have matching street types (e.g. both 'road' or 'street').
    
    Args:
        address1: First address
        address2: Second address
        
    Returns:
        True if street types match
    """
    st1 = _extract_street_types(address1)
    st2 = _extract_street_types(address2)
    if not st1 or not st2:
        return False
    return len(st1 & st2) > 0


def city_match(address1: str, address2: str, country1: str = "", country2: str = "") -> bool:
    """
    Check if parsed cities match between two addresses.
    
    Args:
        address1: First address
        address2: Second address
        country1: Country of first entity
        country2: Country of second entity
        
    Returns:
        True if parsed cities match and are non-empty
    """
    p1 = parse_address(address1, country1)
    p2 = parse_address(address2, country2)
    c1, c2 = p1.get("city"), p2.get("city")
    if c1 and c2:
        return c1.lower().strip() == c2.lower().strip()
    return False


def extract_address_features(
    address1: str,
    address2: str,
    country1: str = "",
    country2: str = ""
) -> Dict[str, float]:
    """
    Extract full set of pairwise address similarity and structural features.
    
    Args:
        address1: First address string (raw or normalized)
        address2: Second address string (raw or normalized)
        country1: Country code/name for entity 1
        country2: Country code/name for entity 2
        
    Returns:
        Dictionary of feature name -> float value
    """
    a1 = str(address1 or "").strip()
    a2 = str(address2 or "").strip()
    
    features: Dict[str, float] = {}

    # 1. String & Token Similarities (RapidFuzz scaled to [0.0, 1.0])
    if not a1 or not a2:
        features['address_similarity'] = 0.0
        features['address_ratio'] = 0.0
        features['address_wratio'] = 0.0
        features['address_partial_ratio'] = 0.0
        features['address_token_sort_ratio'] = 0.0
        features['address_token_set_ratio'] = 0.0
        features['address_jaccard'] = 0.0
        features['address_levenshtein'] = float(max(len(a1), len(a2)))
    else:
        ratio_val = float(fuzz.ratio(a1, a2) / 100.0)
        wratio_val = float(fuzz.WRatio(a1, a2) / 100.0)
        partial_val = float(fuzz.partial_ratio(a1, a2) / 100.0)
        sort_val = float(fuzz.token_sort_ratio(a1, a2) / 100.0)
        set_val = float(fuzz.token_set_ratio(a1, a2) / 100.0)
        
        features['address_ratio'] = ratio_val
        features['address_wratio'] = wratio_val
        features['address_partial_ratio'] = partial_val
        features['address_token_sort_ratio'] = sort_val
        features['address_token_set_ratio'] = set_val
        features['address_similarity'] = wratio_val
        features['address_jaccard'] = jaccard_similarity(a1, a2)
        features['address_levenshtein'] = float(levenshtein_distance(a1, a2))

    # 2. Structural & Numeric Features
    features['numeric_token_overlap'] = numeric_token_overlap(a1, a2)
    features['address_numeric_match'] = 1.0 if numeric_match(a1, a2) else 0.0
    features['address_street_type_match'] = 1.0 if street_type_match(a1, a2) else 0.0

    # 3. Component Extraction & Comparison (parse_address)
    p1 = parse_address(a1, country1)
    p2 = parse_address(a2, country2)
    
    h1, h2 = p1.get("house_number"), p2.get("house_number")
    pc1, pc2 = p1.get("postal_code"), p2.get("postal_code")
    st1, st2 = p1.get("state"), p2.get("state")
    c1, c2 = p1.get("city"), p2.get("city")

    features['same_house_number'] = 1.0 if (h1 and h2 and h1.lower().strip() == h2.lower().strip()) else 0.0
    features['same_postal_code'] = 1.0 if (pc1 and pc2 and pc1.lower().strip() == pc2.lower().strip()) else 0.0
    features['same_state'] = 1.0 if (st1 and st2 and st1.lower().strip() == st2.lower().strip()) else 0.0
    features['same_city'] = 1.0 if (c1 and c2 and c1.lower().strip() == c2.lower().strip()) else 0.0
    features['address_city_match'] = features['same_city']

    return features


class AddressFeatureExtractor:
    """
    Feature extractor class for pairwise address comparison.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AddressFeatureExtractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.features = self.config.get('address_features', [
            'address_similarity',
            'address_ratio',
            'address_wratio',
            'address_partial_ratio',
            'address_token_sort_ratio',
            'address_token_set_ratio',
            'address_jaccard',
            'address_levenshtein',
            'numeric_token_overlap',
            'address_numeric_match',
            'address_street_type_match',
            'same_house_number',
            'same_postal_code',
            'same_state',
            'same_city',
        ])
        logger.info(f"Initializing AddressFeatureExtractor with {len(self.features)} features")

    def extract_features(
        self,
        address1: str,
        address2: str,
        country1: str = "",
        country2: str = ""
    ) -> Dict[str, float]:
        """
        Extract address features for a pair of addresses.
        
        Args:
            address1: First address
            address2: Second address
            country1: Country for address 1
            country2: Country for address 2
            
        Returns:
            Dict mapping feature name to float value
        """
        all_feats = extract_address_features(address1, address2, country1, country2)
        # Filter according to self.features if specified, else return all
        if self.config and 'address_features' in self.config:
            return {k: v for k, v in all_feats.items() if k in self.features}
        return all_feats

    def extract_features_batch(self, address_pairs: list) -> list:
        """
        Extract features for multiple address pairs.
        
        Args:
            address_pairs: List of tuples (address1, address2) or (address1, address2, country1, country2)
            
        Returns:
            List of feature dictionaries
        """
        results = []
        for pair in address_pairs:
            if len(pair) == 4:
                a1, a2, c1, c2 = pair
            elif len(pair) == 2:
                a1, a2 = pair
                c1, c2 = "", ""
            else:
                a1, a2 = pair[0], pair[1]
                c1 = pair[2] if len(pair) > 2 else ""
                c2 = pair[3] if len(pair) > 3 else ""
            results.append(self.extract_features(a1, a2, c1, c2))
        return results
