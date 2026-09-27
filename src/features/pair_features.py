"""
pair_features.py — Pairwise & cross-field feature engineering for entity resolution (Member 3).

Combines name similarity features (Member 1) and address similarity features (Member 2)
with cross-field interactions (country_match, house_number_match, length ratios/diffs)
and candidate metadata preservation (s1_entity_id, candidate_entity_id, candidate_source).
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Union
import logging
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.features.name_features import NameFeatureExtractor
from src.features.address_features import AddressFeatureExtractor, parse_address

logger = logging.getLogger(__name__)


def extract_house_number(address: str) -> str:
    """
    Robust house number extraction from business address.
    Matches house/building numbers while avoiding 5-6 digit postal codes.
    
    Args:
        address: Raw or normalized business address string.
        
    Returns:
        Extracted house number string or empty string if not found.
    """
    if not address or not str(address).strip():
        return ""
    addr = str(address).strip()
    
    # 1. Match leading house digits/alphanumeric (e.g. "1795 Westchester Dr", "12-A Main St", "100B")
    match = re.search(r"^\b(\d{1,4}[A-Za-z]?|\d{1,4}-\d{1,4}[A-Za-z]?)\b", addr)
    if match:
        return match.group(1)

    # 2. Match unit/suite/house prefix (e.g. "No. 42", "Building 5", "Apt 3B")
    match = re.search(r"\b(?:no|building|bldg|ste|suite|apt|unit|house|plot|flat)\.?\s*(\d{1,4}[A-Za-z]?)\b", addr, re.IGNORECASE)
    if match:
        return match.group(1)

    # 3. Match any 1-4 digit standalone number (excluding 5-6 digit postal codes)
    matches = re.findall(r"\b(\d{1,4})\b", addr)
    if matches:
        return matches[0]

    return ""


def calculate_length_metrics(s1: str, s2: str) -> Tuple[float, float]:
    """
    Compute length ratio and length difference safely.
    
    Returns:
        Tuple of (length_ratio, length_diff)
    """
    len1 = len(str(s1 or "").strip())
    len2 = len(str(s2 or "").strip())
    
    if len1 == 0 and len2 == 0:
        ratio = 1.0
    else:
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        ratio = float(min_len / max_len) if max_len > 0 else 1.0
        
    diff = float(abs(len1 - len2))
    return ratio, diff


class PairFeatureExtractor:
    """
    Comprehensive feature extractor for entity pairs.
    Coordinates Member 1 (Name), Member 2 (Address), Member 3 (Cross-field & Metadata).
    """
    
    def __init__(
        self,
        name_extractor: Optional[NameFeatureExtractor] = None,
        address_extractor: Optional[AddressFeatureExtractor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.name_extractor = name_extractor or NameFeatureExtractor(self.config)
        self.address_extractor = address_extractor or AddressFeatureExtractor(self.config)
        logger.info("Initializing PairFeatureExtractor")
    
    def extract_pair_features(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract all pairwise features for an entity pair.
        
        Args:
            entity1: First entity dictionary (containing name, address, country)
            entity2: Second entity dictionary (containing name, address, country)
            
        Returns:
            Dictionary of feature name -> feature value
        """
        features: Dict[str, Any] = {}

        name1 = str(entity1.get('normalized_name') or entity1.get('business_name') or entity1.get('original_name') or "").strip()
        name2 = str(entity2.get('normalized_name') or entity2.get('business_name') or entity2.get('original_name') or "").strip()

        addr1 = str(entity1.get('normalized_address') or entity1.get('business_address') or entity1.get('original_address') or "").strip()
        addr2 = str(entity2.get('normalized_address') or entity2.get('business_address') or entity2.get('original_address') or "").strip()

        country1 = str(entity1.get('country') or "").strip()
        country2 = str(entity2.get('country') or "").strip()

        # 1. Missingness Interaction Features
        name1_empty = (len(name1) == 0)
        name2_empty = (len(name2) == 0)
        addr1_empty = (len(addr1) == 0)
        addr2_empty = (len(addr2) == 0)

        name_missing = 1.0 if (name1_empty or name2_empty) else 0.0
        address_missing = 1.0 if (addr1_empty or addr2_empty) else 0.0
        both_missing = 1.0 if (name_missing == 1.0 and address_missing == 1.0) else 0.0

        features['name_missing'] = name_missing
        features['address_missing'] = address_missing
        features['both_missing'] = both_missing

        # 2. Extract Name Features (Member 1)
        name_feats = self.name_extractor.extract_features(name1, name2)
        features.update(name_feats)

        # 3. Extract Address Features (Member 2)
        addr_feats = self.address_extractor.extract_features(addr1, addr2, country1, country2)
        features.update(addr_feats)

        # 4. Member 3 Cross-Field Features
        # A. Country Match
        c1_norm = country1.upper()
        c2_norm = country2.upper()
        country_match_val = 1.0 if (c1_norm and c2_norm and c1_norm == c2_norm) else 0.0
        features['country_match'] = country_match_val
        features['same_country'] = country_match_val

        # B. House Number Extraction & Match
        h1 = extract_house_number(addr1)
        h2 = extract_house_number(addr2)
        features['house_number_s1'] = h1
        features['house_number_candidate'] = h2
        features['house_number_match'] = 1.0 if (h1 and h2 and h1 == h2) else 0.0

        # C. Name Length Features
        name_ratio, name_diff = calculate_length_metrics(name1, name2)
        features['name_length_ratio'] = name_ratio
        features['name_len_ratio'] = name_ratio
        features['name_length_diff'] = name_diff
        features['name_length_difference'] = name_diff

        # D. Address Length Features
        addr_ratio, addr_diff = calculate_length_metrics(addr1, addr2)
        features['address_length_ratio'] = addr_ratio
        features['address_length_diff'] = addr_diff
        features['address_length_difference'] = addr_diff

        # E. Cross-Field Similarity Interactions
        name_sim = float(features.get('name_similarity', features.get('name_wratio', 0.0)))
        addr_sim = float(features.get('address_similarity', features.get('address_wratio', 0.0)))
        features['name_address_similarity_product'] = float(name_sim * addr_sim)
        features['name_address_similarity_sum'] = float(name_sim + addr_sim)

        return features

    def extract_features_for_pairs(
        self,
        entity_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        logger.info(f"Extracting features for {len(entity_pairs)} entity pairs")
        return [self.extract_pair_features(e1, e2) for e1, e2 in entity_pairs]


def build_feature_matrix(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unified Phase 4 pandas DataFrame feature matrix from candidate pairs.
    Preserves candidate pair metadata: source1_entity_id, candidate_entity_id, candidate_source.
    
    Args:
        candidates_df: Enriched candidate pairs DataFrame.
        
    Returns:
        DataFrame X containing candidate metadata + numerical feature columns.
    """
    if candidates_df.empty:
        return pd.DataFrame()

    extractor = PairFeatureExtractor()

    s1_name_col = next((c for c in ['normalized_name_s1', 'business_name_s1', 'original_name_s1', 'name_s1'] if c in candidates_df.columns), None)
    cand_name_col = next((c for c in ['normalized_name_cand', 'business_name_cand', 'original_name_cand', 'name_cand'] if c in candidates_df.columns), None)

    s1_addr_col = next((c for c in ['normalized_address_s1', 'business_address_s1', 'original_address_s1', 'address_s1'] if c in candidates_df.columns), None)
    cand_addr_col = next((c for c in ['normalized_address_cand', 'business_address_cand', 'original_address_cand', 'address_cand'] if c in candidates_df.columns), None)

    s1_country_col = next((c for c in ['country_s1', 'country_1'] if c in candidates_df.columns), None)
    cand_country_col = next((c for c in ['country_cand', 'country_2'] if c in candidates_df.columns), None)

    feature_rows = []
    
    for row in candidates_df.itertuples(index=False):
        row_dict = row._asdict()

        e1 = {
            'business_name': row_dict.get(s1_name_col, "") if s1_name_col else "",
            'business_address': row_dict.get(s1_addr_col, "") if s1_addr_col else "",
            'country': row_dict.get(s1_country_col, "") if s1_country_col else "",
        }
        e2 = {
            'business_name': row_dict.get(cand_name_col, "") if cand_name_col else "",
            'business_address': row_dict.get(cand_addr_col, "") if cand_addr_col else "",
            'country': row_dict.get(cand_country_col, "") if cand_country_col else "",
        }

        feats = extractor.extract_pair_features(e1, e2)
        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows)
    
    # Exclude internal string helper columns from numerical output if desired, or keep as numeric
    string_cols = ["house_number_s1", "house_number_candidate"]
    for col in string_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    X = X.fillna(0.0)
    return X


def generate_candidate_source_metadata(candidates_df: pd.DataFrame) -> pd.Series:
    """
    Generate candidate_source metadata column ('S2' or 'S3') from candidate_entity_id.
    """
    cand_ids = candidates_df["candidate_entity_id"].fillna("").astype(str)
    
    def parse_src(c_id: str) -> str:
        if c_id.startswith("S2") or c_id.startswith("s2"):
            return "S2"
        elif c_id.startswith("S3") or c_id.startswith("s3"):
            return "S3"
        return "UNKNOWN"
        
    return cand_ids.apply(parse_src)


def generate_feature_schema(output_path: Optional[Union[str, Path]] = "features/feature_schema.json") -> Dict[str, Any]:
    """
    Generate machine-readable JSON feature schema detailing categories, dtypes, and member assignments.
    """
    schema = {
        "metadata": {
            "s1_entity_id": {"category": "metadata", "dtype": "string", "description": "S1 reference entity ID", "source": "metadata"},
            "candidate_entity_id": {"category": "metadata", "dtype": "string", "description": "Candidate S2/S3 entity ID", "source": "metadata"},
            "candidate_source": {"category": "metadata", "dtype": "string", "description": "Candidate source origin ('S2' or 'S3')", "source": "member_3"}
        },
        "features": {
            # Member 1: Name Features
            "name_ratio": {"category": "name", "dtype": "float64", "description": "RapidFuzz ratio similarity between names [0..1]", "source": "member_1"},
            "name_wratio": {"category": "name", "dtype": "float64", "description": "RapidFuzz weighted ratio similarity [0..1]", "source": "member_1"},
            "name_token_sort": {"category": "name", "dtype": "float64", "description": "RapidFuzz token sort ratio [0..1]", "source": "member_1"},
            "name_token_set": {"category": "name", "dtype": "float64", "description": "RapidFuzz token set ratio [0..1]", "source": "member_1"},
            "name_partial_ratio": {"category": "name", "dtype": "float64", "description": "RapidFuzz partial ratio similarity [0..1]", "source": "member_1"},
            "name_norm_edit": {"category": "name", "dtype": "float64", "description": "Normalized edit distance [0..1]", "source": "member_1"},
            "name_levenshtein": {"category": "name", "dtype": "float64", "description": "Raw Levenshtein edit distance", "source": "member_1"},
            "name_jaccard": {"category": "name", "dtype": "float64", "description": "Jaccard similarity of name tokens [0..1]", "source": "member_1"},
            "name_acronym_match": {"category": "name", "dtype": "float64", "description": "Acronym match indicator (1.0 if match, 0.0 otherwise)", "source": "member_1"},
            "name_similarity": {"category": "name", "dtype": "float64", "description": "Primary name similarity summary score [0..1]", "source": "member_1"},
            "name_s1_missing": {"category": "missingness", "dtype": "float64", "description": "S1 name missing indicator", "source": "member_1"},
            "name_s2_missing": {"category": "missingness", "dtype": "float64", "description": "Candidate name missing indicator", "source": "member_1"},

            # Member 2: Address Features
            "address_ratio": {"category": "address", "dtype": "float64", "description": "RapidFuzz ratio similarity between addresses [0..1]", "source": "member_2"},
            "address_wratio": {"category": "address", "dtype": "float64", "description": "RapidFuzz weighted ratio similarity between addresses [0..1]", "source": "member_2"},
            "address_partial_ratio": {"category": "address", "dtype": "float64", "description": "RapidFuzz partial ratio similarity for addresses [0..1]", "source": "member_2"},
            "address_token_sort_ratio": {"category": "address", "dtype": "float64", "description": "RapidFuzz token sort ratio for addresses [0..1]", "source": "member_2"},
            "address_token_set_ratio": {"category": "address", "dtype": "float64", "description": "RapidFuzz token set ratio for addresses [0..1]", "source": "member_2"},
            "address_similarity": {"category": "address", "dtype": "float64", "description": "Primary address similarity summary score [0..1]", "source": "member_2"},
            "address_jaccard": {"category": "address", "dtype": "float64", "description": "Jaccard similarity of address tokens [0..1]", "source": "member_2"},
            "address_levenshtein": {"category": "address", "dtype": "float64", "description": "Raw Levenshtein distance between addresses", "source": "member_2"},
            "numeric_token_overlap": {"category": "address", "dtype": "float64", "description": "Jaccard overlap of numeric tokens in address [0..1]", "source": "member_2"},
            "address_numeric_match": {"category": "address", "dtype": "float64", "description": "Numeric components match indicator (1.0 if overlap > 0)", "source": "member_2"},
            "address_street_type_match": {"category": "address", "dtype": "float64", "description": "Street type keyword match indicator", "source": "member_2"},
            "same_house_number": {"category": "address", "dtype": "float64", "description": "House number component match indicator", "source": "member_2"},
            "same_postal_code": {"category": "address", "dtype": "float64", "description": "Postal code match indicator", "source": "member_2"},
            "same_state": {"category": "address", "dtype": "float64", "description": "State component match indicator", "source": "member_2"},
            "same_city": {"category": "address", "dtype": "float64", "description": "City component match indicator", "source": "member_2"},
            "address_city_match": {"category": "address", "dtype": "float64", "description": "City match indicator", "source": "member_2"},

            # Member 3: Cross-Field Features & Metadata
            "country_match": {"category": "cross_field", "dtype": "float64", "description": "Country equality indicator (1.0 if match, 0.0 otherwise)", "source": "member_3"},
            "same_country": {"category": "cross_field", "dtype": "float64", "description": "Alias for country_match", "source": "member_3"},
            "house_number_match": {"category": "cross_field", "dtype": "float64", "description": "Robust house number match indicator (1.0 if match)", "source": "member_3"},
            "name_length_ratio": {"category": "cross_field", "dtype": "float64", "description": "Min/Max ratio of name lengths [0..1]", "source": "member_3"},
            "name_len_ratio": {"category": "cross_field", "dtype": "float64", "description": "Alias for name_length_ratio", "source": "member_3"},
            "name_length_diff": {"category": "cross_field", "dtype": "float64", "description": "Absolute difference in name lengths", "source": "member_3"},
            "name_length_difference": {"category": "cross_field", "dtype": "float64", "description": "Alias for name_length_diff", "source": "member_3"},
            "address_length_ratio": {"category": "cross_field", "dtype": "float64", "description": "Min/Max ratio of address lengths [0..1]", "source": "member_3"},
            "address_length_diff": {"category": "cross_field", "dtype": "float64", "description": "Absolute difference in address lengths", "source": "member_3"},
            "address_length_difference": {"category": "cross_field", "dtype": "float64", "description": "Alias for address_length_diff", "source": "member_3"},
            "name_address_similarity_product": {"category": "cross_field", "dtype": "float64", "description": "Product of name_similarity and address_similarity", "source": "member_3"},
            "name_address_similarity_sum": {"category": "cross_field", "dtype": "float64", "description": "Sum of name_similarity and address_similarity", "source": "member_3"},
            "name_missing": {"category": "missingness", "dtype": "float64", "description": "Either name missing indicator", "source": "member_3"},
            "address_missing": {"category": "missingness", "dtype": "float64", "description": "Either address missing indicator", "source": "member_3"},
            "both_missing": {"category": "missingness", "dtype": "float64", "description": "Both name and address missing indicator", "source": "member_3"}
        }
    }
    
    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Saved machine-readable feature schema to {out_p}")
        
    return schema
