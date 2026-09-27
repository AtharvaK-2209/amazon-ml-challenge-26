"""
Business name feature engineering for entity resolution.

This module computes similarity features between business name pairs
to support entity matching decisions.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def jaccard_similarity(
    name1: str,
    name2: str
) -> float:
    """
    Compute Jaccard similarity between two business names.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Jaccard similarity score in [0, 1]
        
    TODO: Implement in Phase 1
    """
    return 0.0


def levenshtein_distance(
    name1: str,
    name2: str
) -> int:
    """
    Compute Levenshtein edit distance between two business names.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Edit distance (number of operations)
        
    TODO: Implement in Phase 1
    """
    return 0


def token_sort_ratio(
    name1: str,
    name2: str
) -> float:
    """
    Compute token sort ratio similarity between two business names.
    
    Tokenizes both strings, sorts the tokens, and computes similarity.
    Handles word reordering issues.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Similarity ratio in [0, 100]
        
    TODO: Implement in Phase 1
    """
    return 0.0


def partial_ratio(
    name1: str,
    name2: str
) -> float:
    """
    Compute partial ratio similarity between two business names.
    
    Finds the best matching substring and computes similarity.
    Useful for matching substrings within longer names.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Similarity ratio in [0, 100]
        
    TODO: Implement in Phase 1
    """
    return 0.0


def acronym_match(
    name1: str,
    name2: str
) -> bool:
    """
    Check if either name is an acronym of the other.
    
    Example: "ABC Corp" matches "ABC"
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        True if acronym match found
        
    TODO: Implement in Phase 1
    """
    return False


class NameFeatureExtractor:
    """
    Feature extractor for business name pairs.
    
    Computes a comprehensive set of similarity features between
    business names for entity matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize name feature extractor.
        
        Args:
            config: Feature configuration parameters
        """
        self.config = config or {}
        self.features = self.config.get('name_features', [
            'jaccard_similarity',
            'levenshtein_distance',
            'token_sort_ratio',
            'partial_ratio',
            'acronym_match'
        ])
        logger.info(f"Initializing NameFeatureExtractor with {len(self.features)} features")
    
    def extract_features(
        self,
        name1: str,
        name2: str
    ) -> Dict[str, float]:
        """
        Extract name similarity features for a name pair.
        
        Args:
            name1: First business name
            name2: Second business name
            
        Returns:
            Dictionary of feature name -> feature value
            
        TODO: Implement in Phase 1
        """
        features = {}
        
        if 'jaccard_similarity' in self.features:
            features['name_jaccard'] = jaccard_similarity(name1, name2)
        
        if 'levenshtein_distance' in self.features:
            features['name_levenshtein'] = float(levenshtein_distance(name1, name2))
        
        if 'token_sort_ratio' in self.features:
            features['name_token_sort'] = token_sort_ratio(name1, name2)
        
        if 'partial_ratio' in self.features:
            features['name_partial_ratio'] = partial_ratio(name1, name2)
        
        if 'acronym_match' in self.features:
            features['name_acronym_match'] = float(acronym_match(name1, name2))
        
        return features
    
    def extract_features_batch(
        self,
        name_pairs: list
    ) -> list:
        """
        Extract features for multiple name pairs.
        
        Args:
            name_pairs: List of (name1, name2) tuples
            
        Returns:
            List of feature dictionaries
            
        TODO: Implement in Phase 1
        """
        return [self.extract_features(n1, n2) for n1, n2 in name_pairs]
