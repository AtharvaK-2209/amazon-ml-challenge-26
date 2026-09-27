"""
test_phase8_member2.py — Verification unit tests for Phase 8 Member 2 Auditor.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("phase8/member2")


def test_deliverable_files_exist():
    """Verify all Phase 8 Member 2 deliverable files exist."""
    required_files = [
        OUT_DIR / "data_audit.md",
        OUT_DIR / "output_audit.md",
        OUT_DIR / "validation_log.txt",
        OUT_DIR / "phase8_validation_report.md",
        OUT_DIR / "audit_results.json",
        OUT_DIR / "candidate_distribution.csv",
        OUT_DIR / "invalid_ids.csv",
        OUT_DIR / "duplicate_rows.csv"
    ]
    for filepath in required_files:
        assert filepath.exists(), f"Missing deliverable: {filepath}"


def test_audit_results_json():
    """Verify audit_results.json structure and numbers."""
    with open(OUT_DIR / "audit_results.json") as f:
        res = json.load(f)
        
    assert "data_info" in res
    assert "candidate_info" in res
    assert "output_info" in res
    assert "official_validator_passed" in res
    
    assert res["official_validator_passed"] is True
    assert res["data_info"]["expected_unique_s1_entities"] == 1732544
    assert res["output_info"]["output_unique_s1"] == 1732544
    assert res["output_info"]["missing_s1_count"] == 0
    assert res["output_info"]["duplicate_s1_rows"] == 0
    assert res["output_info"]["invalid_match_ids_count"] == 0


def test_phase8_report_checklist():
    """Verify phase8_validation_report.md checklist items."""
    content = (OUT_DIR / "phase8_validation_report.md").read_text()
    
    assert "TEST DATA: PASS" in content
    assert "S1 COVERAGE: PASS" in content
    assert "DUPLICATES: PASS" in content
    assert "MATCH IDS: PASS" in content
    assert "OUTPUT SCHEMA: PASS" in content
    assert "OFFICIAL VALIDATOR: PASS" in content


def test_validation_log_pass():
    """Verify validation_log.txt captured official validator output."""
    log_content = (OUT_DIR / "validation_log.txt").read_text()
    assert "required S1 entities: 1732544" in log_content
    assert "PASS — no blocking issues found. Safe to submit." in log_content
