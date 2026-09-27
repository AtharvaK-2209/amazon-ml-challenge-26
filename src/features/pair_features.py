"""
pair_features.py — Pairwise & cross-field feature engineering for entity resolution.

Combines name similarity features and address similarity features with
cross-field interactions (e.g. name*address similarity, length differences,
same_country, same_city) and missingness indicators.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import pandas as pd
import numpy as np

from src.features.name_features import NameFeatureExtractor
from src.features.address_features import AddressFeatureExtractor, parse_address

logger = logging.getLogger(__name__)


class PairFeatureExtractor:
    """
    Comprehensive feature extractor for entity pairs.
    
    Coordinates extraction of name features, address features,
    cross-field features, and missingness interaction features.
    """
    
    def __init__(
        self,
        name_extractor: Optional[NameFeatureExtractor] = None,
        address_extractor: Optional[AddressFeatureExtractor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PairFeatureExtractor.
        
        Args:
            name_extractor: NameFeatureExtractor instance
            address_extractor: AddressFeatureExtractor instance
            config: Feature configuration parameters
        """
        self.config = config or {}
        self.name_extractor = name_extractor or NameFeatureExtractor(self.config)
        self.address_extractor = address_extractor or AddressFeatureExtractor(self.config)
        
        logger.info("Initializing PairFeatureExtractor")
    
    def extract_pair_features(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract all pairwise features for an entity pair.
        
        Args:
            entity1: First entity dictionary (containing name, address, country)
            entity2: Second entity dictionary (containing name, address, country)
            
        Returns:
            Dictionary of feature name -> float feature value
        """
        features: Dict[str, float] = {}

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

        # 2. Extract Name Features
        name_feats = self.name_extractor.extract_features(name1, name2)
        features.update(name_feats)

        # 3. Extract Address Features
        addr_feats = self.address_extractor.extract_features(addr1, addr2, country1, country2)
        features.update(addr_feats)

        # 4. Cross-Field Features
        name_sim = features.get('name_similarity', features.get('name_wratio', 0.0))
        addr_sim = features.get('address_similarity', features.get('address_wratio', 0.0))

        features['name_address_similarity_product'] = float(name_sim * addr_sim)
        features['name_address_similarity_sum'] = float(name_sim + addr_sim)

        # Country match
        features['same_country'] = 1.0 if (country1 and country2 and country1.upper() == country2.upper()) else 0.0

        # Length differences
        features['name_length_difference'] = float(abs(len(name1) - len(name2)))
        features['address_length_difference'] = float(abs(len(addr1) - len(addr2)))

        return features
    
    def extract_features_for_pairs(
        self,
        entity_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, float]]:
        """
        Extract features for multiple entity pairs.
        
        Args:
            entity_pairs: List of (entity1, entity2) tuples
            
        Returns:
            List of feature dictionaries
        """
        logger.info(f"Extracting features for {len(entity_pairs)} entity pairs")
        return [self.extract_pair_features(e1, e2) for e1, e2 in entity_pairs]
    
    def create_feature_matrix(
        self,
        candidate_pairs: List[Tuple[str, str, str]],
        s1_entities: Dict[str, Dict[str, Any]],
        s2_entities: Dict[str, Dict[str, Any]],
        s3_entities: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Create feature matrix for candidate pairs using entity lookup dictionaries.
        
        Args:
            candidate_pairs: List of (s1_id, s2_id, s3_id) or (s1_id, cand_id) tuples
            s1_entities: Mapping of S1 entity ID -> entity dict
            s2_entities: Mapping of S2 entity ID -> entity dict
            s3_entities: Optional mapping of S3 entity ID -> entity dict
            
        Returns:
            Tuple of (feature_names list, list of feature dicts)
        """
        logger.info(f"Creating feature matrix for {len(candidate_pairs)} candidates")
        
        s23_entities = dict(s2_entities)
        if s3_entities:
            s23_entities.update(s3_entities)

        feature_rows = []
        for pair in candidate_pairs:
            s1_id = pair[0]
            cand_id = pair[1] if len(pair) > 1 else pair[0]
            
            e1 = s1_entities.get(s1_id, {})
            e2 = s23_entities.get(cand_id, {})

            feats = self.extract_pair_features(e1, e2)
            feature_rows.append(feats)

        feature_names = list(feature_rows[0].keys()) if feature_rows else []
        return feature_names, feature_rows


def build_feature_matrix(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pandas DataFrame feature matrix from a candidate pairs DataFrame.
    
    This function is integrated into pipeline.py and processes candidate pairs
    enriched with s1 and candidate representations.
    
    Args:
        candidates_df: DataFrame containing candidate pairs (with _s1 and _cand suffix columns)
        
    Returns:
        DataFrame X containing numerical feature columns
    """
    if candidates_df.empty:
        return pd.DataFrame()

    extractor = PairFeatureExtractor()

    # Identify relevant columns in candidates_df
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
    X = X.fillna(0.0)
    return X
