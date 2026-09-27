"""
Business name feature engineering for entity resolution.

This module computes similarity features between business name pairs
to support entity matching decisions.
"""

from typing import Dict, Any, List, Optional
import logging
import re
from rapidfuzz import fuzz, distance

logger = logging.getLogger(__name__)


def jaccard_similarity(name1: str, name2: str) -> float:
    """
    Compute Jaccard similarity between token sets of two business names.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Jaccard similarity score in [0, 1]
    """
    if not name1 or not name2:
        return 0.0
    set1 = set(name1.lower().split())
    set2 = set(name2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return float(len(intersection) / len(union))


def levenshtein_distance(name1: str, name2: str) -> int:
    """
    Compute Levenshtein edit distance between two business names.
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Edit distance (number of character operations)
    """
    if not name1 or not name2:
        return max(len(name1 or ""), len(name2 or ""))
    return int(distance.Levenshtein.distance(name1.lower(), name2.lower()))


def token_sort_ratio(name1: str, name2: str) -> float:
    """
    Compute token sort ratio similarity between two business names.
    
    Tokenizes both strings, sorts the tokens, and computes similarity.
    Handles word reordering issues. Returns scaled score in [0.0, 1.0].
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Similarity ratio in [0.0, 1.0]
    """
    if not name1 or not name2:
        return 0.0
    return float(fuzz.token_sort_ratio(name1, name2) / 100.0)


def partial_ratio(name1: str, name2: str) -> float:
    """
    Compute partial ratio similarity between two business names.
    
    Finds the best matching substring and computes similarity.
    Returns scaled score in [0.0, 1.0].
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        Similarity ratio in [0.0, 1.0]
    """
    if not name1 or not name2:
        return 0.0
    return float(fuzz.partial_ratio(name1, name2) / 100.0)


def acronym_match(name1: str, name2: str) -> bool:
    """
    Check if either name is an acronym of the other or if acronyms match.
    
    Example: "ABC Corp" matches "ABC"
    
    Args:
        name1: First business name
        name2: Second business name
        
    Returns:
        True if acronym match found
    """
    if not name1 or not name2:
        return False
    
    t1 = [w for w in re.split(r'\s+', name1.strip()) if w]
    t2 = [w for w in re.split(r'\s+', name2.strip()) if w]
    
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
        """
        Extract name similarity features for a name pair.
        
        Args:
            name1: First business name
            name2: Second business name
            
        Returns:
            Dictionary of feature name -> feature value (all float values in 0..1 or exact metric)
        """
        n1 = str(name1 or "")
        n2 = str(name2 or "")

        features: Dict[str, float] = {}
        
        if 'ratio' in self.features or 'name_ratio' in self.features:
            features['name_ratio'] = float(fuzz.ratio(n1, n2) / 100.0) if n1 and n2 else 0.0
            
        if 'wratio' in self.features or 'name_wratio' in self.features:
            features['name_wratio'] = float(fuzz.WRatio(n1, n2) / 100.0) if n1 and n2 else 0.0

        if 'jaccard_similarity' in self.features or 'name_jaccard' in self.features:
            features['name_jaccard'] = jaccard_similarity(n1, n2)
        
        if 'levenshtein_distance' in self.features or 'name_levenshtein' in self.features:
            features['name_levenshtein'] = float(levenshtein_distance(n1, n2))
        
        if 'token_sort_ratio' in self.features or 'name_token_sort' in self.features:
            features['name_token_sort'] = token_sort_ratio(n1, n2)
            
        if 'token_set_ratio' in self.features or 'name_token_set' in self.features:
            features['name_token_set'] = float(fuzz.token_set_ratio(n1, n2) / 100.0) if n1 and n2 else 0.0
        
        if 'partial_ratio' in self.features or 'name_partial_ratio' in self.features:
            features['name_partial_ratio'] = partial_ratio(n1, n2)
        
        if 'acronym_match' in self.features or 'name_acronym_match' in self.features:
            features['name_acronym_match'] = float(acronym_match(n1, n2))
        
        # Primary summary similarity score for name
        features['name_similarity'] = features.get('name_wratio', features.get('name_token_sort', 0.0))
        
        return features
    
    def extract_features_batch(self, name_pairs: list) -> list:
        """
        Extract features for multiple name pairs.
        
        Args:
            name_pairs: List of (name1, name2) tuples
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_features(n1, n2) for n1, n2 in name_pairs]
