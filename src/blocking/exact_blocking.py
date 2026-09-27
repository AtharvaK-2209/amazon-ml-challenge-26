"""
Exact match blocking for entity resolution.

This module implements blocking strategies based on exact field matches
to generate candidate pairs efficiently.

Blocking generates candidate pairs only - it does NOT make final match decisions.
"""

from typing import Dict, Any, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)


class ExactBlocker:
    """
    Exact match blocking strategy.
    
    Generates candidate pairs by matching entities on exact field values.
    This is the most restrictive blocking strategy with high precision
    but may miss true matches with minor variations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize exact blocker with configuration.
        
        Args:
            config: Blocking configuration parameters
        """
        self.config = config or {}
        self.fields = self.config.get('fields', ['business_name', 'business_address'])
        logger.info(f"Initializing ExactBlocker on fields: {self.fields}")
    
    def build_blocks(
        self,
        entities: List[Dict[str, Any]],
        field: str
    ) -> Dict[str, List[str]]:
        """
        Build blocks based on exact field matches.
        
        Args:
            entities: List of entity dictionaries
            field: Field name to block on
            
        Returns:
            Dictionary mapping field values to lists of entity IDs
            
        TODO: Implement in Phase 1
        """
        logger.info(f"Building blocks on field: {field}")
        
        blocks: Dict[str, List[str]] = {}
        
        for entity in entities:
            entity_id = entity.get('entity_id')
            field_value = entity.get(field, '')
            
            if field_value not in blocks:
                blocks[field_value] = []
            blocks[field_value].append(entity_id)
        
        logger.info(f"Created {len(blocks)} blocks for field {field}")
        return blocks
    
    def generate_candidate_pairs(
        self,
        s1_entities: List[Dict[str, Any]],
        s2_entities: List[Dict[str, Any]],
        s3_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Generate candidate pairs using exact match blocking.
        
        Args:
            s1_entities: List of Source 1 entities
            s2_entities: List of Source 2 entities
            s3_entities: Optional list of Source 3 entities
            
        Returns:
            List of candidate tuples: (s1_entity_id, s2_entity_id, s3_entity_id)
            where s3_entity_id may be None
            
        TODO: Implement in Phase 1
        """
        logger.info("Generating candidate pairs using exact blocking")
        
        candidates: List[Tuple[str, str, str]] = []
        
        # TODO: Implement multi-source blocking logic
        
        logger.info(f"Generated {len(candidates)} candidate pairs")
        return candidates
    
    def get_block_key(
        self,
        entity: Dict[str, Any]
    ) -> str:
        """
        Generate block key for an entity.
        
        Args:
            entity: Entity dictionary
            
        Returns:
            Block key string
            
        TODO: Implement in Phase 1
        """
        key_parts = []
        for field in self.fields:
            key_parts.append(str(entity.get(field, '')))
        return '|'.join(key_parts)
