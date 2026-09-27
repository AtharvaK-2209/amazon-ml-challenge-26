"""
Evaluation module for entity resolution.

This module implements F0.5 evaluation aligned with competition metrics.
Evaluation is performed at the Source-1 entity level.

F0.5 weights precision more heavily than recall, making false merges costly.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_precision(
    true_positives: int,
    false_positives: int
) -> float:
    """
    Compute precision score.
    
    Args:
        true_positives: Number of true positive matches
        false_positives: Number of false positive matches
        
    Returns:
        Precision score in [0, 1]
    """
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)


def compute_recall(
    true_positives: int,
    false_negatives: int
) -> float:
    """
    Compute recall score.
    
    Args:
        true_positives: Number of true positive matches
        false_negatives: Number of false negative matches
        
    Returns:
        Recall score in [0, 1]
    """
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)


def compute_fbeta(
    precision: float,
    recall: float,
    beta: float = 0.5
) -> float:
    """
    Compute F-beta score.
    
    F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    For beta=0.5, precision is weighted more heavily than recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        beta: Weight parameter (0.5 for competition metric)
        
    Returns:
        F-beta score in [0, 1]
    """
    if precision + recall == 0:
        return 0.0
    
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / \
            (beta_squared * precision + recall)
    
    return fbeta


class EntityEvaluator:
    """
    Entity-level evaluator for entity resolution.
    
    Evaluates matching quality at the Source-1 entity level using
    macro-averaged F0.5 score as required by the competition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration including:
                - beta: F-beta parameter (default: 0.5)
                - log_confusion_matrix: Whether to log confusion matrix
                - log_per_entity_results: Whether to log per-entity results
        """
        self.config = config or {}
        self.beta = self.config.get('beta', 0.5)
        self.log_confusion_matrix = self.config.get('log_confusion_matrix', True)
        self.log_per_entity_results = self.config.get('log_per_entity_results', False)
        
        logger.info(f"Initializing EntityEvaluator with F{self.beta} metric")
    
    def evaluate(
        self,
        predictions: Dict[str, Tuple[str, str]],
        ground_truth: Dict[str, Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate entity matching predictions against ground truth.
        
        Args:
            predictions: Dictionary mapping s1_id to (s2_id, s3_id) predictions
            ground_truth: Dictionary mapping s1_id to (s2_id, s3_id) true matches
            
        Returns:
            Dictionary with evaluation metrics:
            - precision
            - recall
            - f0.5
            - true_positives
            - false_positives
            - false_negatives
            
        TODO: Implement full evaluation logic in Phase 1
        """
        logger.info(f"Evaluating {len(predictions)} predictions against {len(ground_truth)} ground truth")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # TODO: Implement entity-level evaluation
        # For each S1 entity:
        # - Check if prediction matches ground truth
        # - Count TP, FP, FN
        
        precision = compute_precision(true_positives, false_positives)
        recall = compute_recall(true_positives, false_negatives)
        fbeta = compute_fbeta(precision, recall, self.beta)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f0.5': fbeta,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def compute_macro_f05(
        self,
        predictions: Dict[str, Tuple[str, str]],
        ground_truth: Dict[str, Tuple[str, str]]
    ) -> float:
        """
        Compute macro-averaged F0.5 score at entity level.
        
        Macro-average computes F0.5 for each S1 entity and then averages,
        giving equal weight to each entity regardless of match status.
        
        Args:
            predictions: Dictionary mapping s1_id to (s2_id, s3_id) predictions
            ground_truth: Dictionary mapping s1_id to (s2_id, s3_id) true matches
            
        Returns:
            Macro-averaged F0.5 score
            
        TODO: Implement in Phase 1
        """
        logger.info("Computing macro-averaged F0.5")
        
        # TODO: Implement macro-averaged F0.5
        
        return 0.0
    
    def generate_confusion_matrix(
        self,
        predictions: Dict[str, Tuple[str, str]],
        ground_truth: Dict[str, Tuple[str, str]]
    ) -> Dict[str, int]:
        """
        Generate confusion matrix for entity matching.
        
        Args:
            predictions: Dictionary mapping s1_id to (s2_id, s3_id) predictions
            ground_truth: Dictionary mapping s1_id to (s2_id, s3_id) true matches
            
        Returns:
            Dictionary with confusion matrix counts
            
        TODO: Implement in Phase 1
        """
        return {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
    
    def log_per_entity_results(
        self,
        predictions: Dict[str, Tuple[str, str]],
        ground_truth: Dict[str, Tuple[str, str]],
        output_path: str
    ) -> None:
        """
        Log detailed per-entity results for analysis.
        
        Args:
            predictions: Dictionary mapping s1_id to (s2_id, s3_id) predictions
            ground_truth: Dictionary mapping s1_id to (s2_id, s3_id) true matches
            output_path: Path to save results
            
        TODO: Implement in Phase 1
        """
        logger.info(f"Logging per-entity results to {output_path}")
        # TODO: Implement detailed logging
