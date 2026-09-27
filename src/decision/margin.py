"""
Margin-based decision logic for entity resolution.

This module implements margin logic to handle ambiguous cases
where multiple candidates have similar match probabilities.

Margin logic helps prevent false merges by requiring clear
separation between the best candidate and alternatives.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarginDecider:
    """
    Margin-based decision engine for entity matching.
    
    Ensures that matches have sufficient margin over alternative
    candidates to reduce false merge rate.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize margin decider.
        
        Args:
            config: Decision configuration including:
                - margin_threshold: Minimum gap between best and second-best
        """
        self.config = config or {}
        self.margin_threshold = self.config.get('margin_threshold', 0.15)
        
        logger.info(f"Initializing MarginDecider with threshold={self.margin_threshold}")
    
    def compute_margin(
        self,
        best_probability: float,
        second_best_probability: float
    ) -> float:
        """
        Compute margin between best and second-best candidate.
        
        Args:
            best_probability: Probability of best candidate
            second_best_probability: Probability of second-best candidate
            
        Returns:
            Margin (absolute difference)
        """
        return best_probability - second_best_probability
    
    def is_sufficient_margin(
        self,
        best_probability: float,
        second_best_probability: Optional[float]
    ) -> bool:
        """
        Check if margin is sufficient for confident match.
        
        Args:
            best_probability: Probability of best candidate
            second_best_probability: Probability of second-best candidate
            
        Returns:
            True if margin is sufficient (or no second candidate)
            
        TODO: Implement full logic in Phase 1
        """
        # If no second candidate, margin is sufficient
        if second_best_probability is None:
            return True
        
        margin = self.compute_margin(best_probability, second_best_probability)
        
        return margin >= self.margin_threshold
    
    def analyze_candidates(
        self,
        candidates: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze margin between candidates.
        
        Args:
            candidates: List of (entity_id, probability) tuples
            
        Returns:
            Dictionary with:
            - best_candidate: entity_id
            - best_probability: float
            - second_best: entity_id or None
            - second_best_probability: float or None
            - margin: float
            - is_sufficient: bool
            
        TODO: Implement in Phase 1
        """
        if not candidates:
            return {
                'best_candidate': None,
                'best_probability': None,
                'second_best': None,
                'second_best_probability': None,
                'margin': None,
                'is_sufficient': False
            }
        
        # Sort by probability
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )
        
        best_id, best_prob = sorted_candidates[0]
        
        if len(sorted_candidates) > 1:
            second_id, second_prob = sorted_candidates[1]
            margin = self.compute_margin(best_prob, second_prob)
            is_sufficient = margin >= self.margin_threshold
        else:
            second_id = None
            second_prob = None
            margin = None
            is_sufficient = True
        
        return {
            'best_candidate': best_id,
            'best_probability': best_prob,
            'second_best': second_id,
            'second_best_probability': second_prob,
            'margin': margin,
            'is_sufficient': is_sufficient
        }
    
    def resolve_ambiguous_cases(
        self,
        s1_entity_id: str,
        candidate_matches: List[Tuple[str, str, float]]
    ) -> Tuple[Optional[Tuple[str, str]], Dict[str, Any]]:
        """
        Resolve cases with multiple high-probability candidates.
        
        Args:
            s1_entity_id: Source 1 entity ID
            candidate_matches: List of (s2_id, s3_id, probability) tuples
            
        Returns:
            Tuple of (best_match, analysis)
            best_match is (s2_id, s3_id) or None if no confident match
            
        TODO: Implement in Phase 1
        """
        analysis = self.analyze_candidates(
            [(f"{s2}|{s3}", prob) for s2, s3, prob in candidate_matches]
        )
        
        if not analysis['is_sufficient']:
            logger.debug(
                f"Insufficient margin for {s1_entity_id}: {analysis['margin']:.3f}"
            )
            return None, analysis
        
        if analysis['best_candidate'] is None:
            return None, analysis
        
        best_s2, best_s3 = analysis['best_candidate'].split('|')
        return (best_s2, best_s3), analysis
