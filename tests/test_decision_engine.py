"""
test_decision_engine.py — Comprehensive unit tests for Phase 6 Member 2 Decision Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.decision.decision_engine import EntityDecisionEngine


def test_ranking_and_deterministic_ties():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-B", "prediction_probability": 0.90},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.90},
    ])
    engine = EntityDecisionEngine(threshold=0.80, margin=0.00)
    matches, stats = engine.process_decisions(df)
    
    # S2-A should win due to tie-breaker ASC sorting on candidate_entity_id
    assert matches.iloc[0]["matched_candidate_id"] == "S2-A"


def test_threshold_pass():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "prediction_probability": 0.95},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.00)
    matches, stats = engine.process_decisions(df)
    
    assert matches.iloc[0]["decision"] == "MATCH"
    assert matches.iloc[0]["matched_candidate_id"] == "S2-001"
    assert matches.iloc[0]["decision_reason"] == "singleton_threshold_pass"


def test_threshold_fail():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "prediction_probability": 0.90},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.00)
    matches, stats = engine.process_decisions(df)
    
    assert matches.iloc[0]["decision"] == "NO_MATCH"
    assert pd.isna(matches.iloc[0]["matched_candidate_id"])
    assert matches.iloc[0]["decision_reason"] == "singleton_threshold_fail"


def test_margin_pass():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.97},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-B", "prediction_probability": 0.91},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.05)
    matches, stats = engine.process_decisions(df)
    
    # top_prob = 0.97, sec_prob = 0.91, margin = 0.06 >= 0.05
    assert matches.iloc[0]["decision"] == "MATCH"
    assert matches.iloc[0]["matched_candidate_id"] == "S2-A"
    assert matches.iloc[0]["decision_reason"] == "threshold_and_margin_pass"


def test_margin_fail():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.97},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-B", "prediction_probability": 0.96},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.05)
    matches, stats = engine.process_decisions(df)
    
    # margin = 0.01 < 0.05 -> margin_fail
    assert matches.iloc[0]["decision"] == "NO_MATCH"
    assert pd.isna(matches.iloc[0]["matched_candidate_id"])
    assert matches.iloc[0]["decision_reason"] == "margin_fail"


def test_singleton_candidate_handling():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.96},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.10)
    matches, stats = engine.process_decisions(df)
    
    assert matches.iloc[0]["is_singleton"] == True
    assert pd.isna(matches.iloc[0]["second_probability"])
    assert pd.isna(matches.iloc[0]["margin"])
    assert matches.iloc[0]["decision"] == "MATCH"


def test_multiple_candidates():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.98},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-B", "prediction_probability": 0.85},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-C", "prediction_probability": 0.60},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.10)
    matches, stats = engine.process_decisions(df)
    
    assert matches.iloc[0]["candidate_count"] == 3
    assert matches.iloc[0]["decision"] == "MATCH"
    assert matches.iloc[0]["matched_candidate_id"] == "S2-A"


def test_s2_s3_conflict():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.98},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-B", "prediction_probability": 0.97},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.00)
    matches, stats = engine.process_decisions(df)
    
    # Highest ranked candidate is selected without artificial bias
    assert matches.iloc[0]["matched_candidate_id"] == "S2-A"
    assert matches.iloc[0]["candidate_source"] == "S2"


def test_duplicate_candidate_pairs_validation():
    df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.95},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-A", "prediction_probability": 0.95},
    ])
    engine = EntityDecisionEngine(threshold=0.94, margin=0.00)
    val_info = engine.validate_input(df)
    assert val_info["duplicate_pairs"] == 1
