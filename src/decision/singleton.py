"""
Singleton handling for entity resolution.

This module handles cases where Source 1 entities have no matches
in Source 2 or Source 3 (singletons).

Proper singleton handling is critical for entity-level F0.5 optimization.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SingletonHandler:
    """
    Singleton detection and handling for entity matching.
    
    Identifies Source 1 entities with no valid matches and handles
    them appropriately in the final output.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize singleton handler.
        
        Args:
            config: Decision configuration including:
                - singleton_detection: Whether to detect singletons
                - singleton_threshold: Probability below which = no match
        """
        self.config = config or {}
        self.singleton_detection = self.config.get('singleton_detection', True)
        self.singleton_threshold = self.config.get('singleton_threshold', 0.3)
        
        logger.info("Initializing SingletonHandler")
    
    def is_singleton(
        self,
        max_probability: Optional[float]
    ) -> bool:
        """
        Determine if an entity is a singleton (no match).
        
        Args:
            max_probability: Highest match probability for this entity
            
        Returns:
            True if entity should be treated as singleton
            
        TODO: Implement full logic in Phase 1
        """
        if not self.singleton_detection:
            return False
        
        if max_probability is None:
            return True
        
        return max_probability < self.singleton_threshold
    
    def identify_singletons(
        self,
        s1_entities: List[str],
        match_results: Dict[str, Tuple[str, str, float]]
    ) -> List[str]:
        """
        Identify all singleton entities in S1.
        
        Args:
            s1_entities: List of all Source 1 entity IDs
            match_results: Dictionary mapping s1_id to (s2_id, s3_id, probability)
            
        Returns:
            List of singleton entity IDs
            
        TODO: Implement in Phase 1
        """
        logger.info("Identifying singleton entities")
        
        singletons = []
        
        for s1_id in s1_entities:
            if s1_id not in match_results:
                singletons.append(s1_id)
            else:
                _, _, probability = match_results[s1_id]
                if self.is_singleton(probability):
                    singletons.append(s1_id)
        
        logger.info(f"Found {len(singletons)} singleton entities")
        return singletons
    
    def handle_singletons(
        self,
        match_results: Dict[str, Tuple[str, str, float]],
        singletons: List[str]
    ) -> Dict[str, Tuple[Optional[str], Optional[str], float]]:
        """
        Update match results to properly represent singletons.
        
        Args:
            match_results: Original match results
            singletons: List of singleton entity IDs
            
        Returns:
            Updated match results with singletons marked as (None, None, 0.0)
            
        TODO: Implement in Phase 1
        """
        logger.info(f"Handling {len(singletons)} singletons")
        
        updated_results = match_results.copy()
        
        for s1_id in singletons:
            updated_results[s1_id] = (None, None, 0.0)
        
        return updated_results
    
    def validate_coverage(
        self,
        s1_entities: List[str],
        match_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that every S1 entity appears exactly once in results.
        
        Competition requirement: Every S1 entity must appear exactly once.
        
        Args:
            s1_entities: List of all Source 1 entity IDs
            match_results: Match results dictionary
            
        Returns:
            Validation statistics
            
        TODO: Implement in Phase 1
        """
        logger.info("Validating S1 entity coverage")
        
        stats = {
            'total_s1_entities': len(s1_entities),
            'entities_with_matches': 0,
            'singleton_entities': 0,
            'missing_entities': 0,
            'duplicate_entities': 0,
            'is_valid': True
        }
        
        seen = set()
        
        for s1_id in s1_entities:
            if s1_id not in match_results:
                stats['missing_entities'] += 1
                stats['is_valid'] = False
            elif s1_id in seen:
                stats['duplicate_entities'] += 1
                stats['is_valid'] = False
            else:
                seen.add(s1_id)
                match = match_results[s1_id]
                if match[0] is None or match[1] is None:
                    stats['singleton_entities'] += 1
                else:
                    stats['entities_with_matches'] += 1
        
        logger.info(f"Validation: {stats}")
        return stats
