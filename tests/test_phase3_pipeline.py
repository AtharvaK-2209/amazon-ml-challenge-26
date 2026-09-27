"""
test_phase3_pipeline.py — Unit tests for Phase 3 infrastructure and validation suite.
Does not require live AWS credentials (uses mocks/pure functions).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.phase3.config import Phase3Config, load_phase3_config
from src.phase3.io import FeatureIO, S3ArtifactManager
from src.phase3.experiment import ExperimentTracker
from src.phase3.pipeline import validate_feature_matrix


def test_config_loading(tmp_path):
    """Test loading configuration from YAML file."""
    yaml_content = """
project:
  phase: 3
  random_seed: 99
input:
  candidate_pairs: "output/test_cands.tsv"
features:
  name: true
  address: false
output:
  parquet_file: "output/test_features.parquet"
aws:
  bucket: "test-bucket"
experiments:
  experiment_id: "EXP-TEST"
"""
    cfg_file = tmp_path / "test_phase3.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_phase3_config(cfg_file)
    assert cfg.project.random_seed == 99
    assert cfg.input.candidate_pairs == "output/test_cands.tsv"
    assert cfg.features.name is True
    assert cfg.features.address is False
    assert cfg.aws.bucket == "test-bucket"
    assert cfg.experiments.experiment_id == "EXP-TEST"


def test_s3_path_generation():
    """Test S3 upload and download command generation."""
    mgr = S3ArtifactManager(bucket="my-bucket", profile="my-profile", region="us-east-1")
    upload_cmd = mgr.build_upload_command("output/pair_features.parquet", "features/phase3/EXP-001/pair_features.parquet")
    download_cmd = mgr.build_download_command("features/phase3/EXP-001/pair_features.parquet", "local.parquet")

    assert "aws s3 cp output/pair_features.parquet s3://my-bucket/features/phase3/EXP-001/pair_features.parquet" in upload_cmd
    assert "--profile my-profile" in upload_cmd
    assert "--region us-east-1" in upload_cmd
    assert "aws s3 cp s3://my-bucket/features/phase3/EXP-001/pair_features.parquet local.parquet" in download_cmd


def test_candidate_pair_loading_aggregated(tmp_path):
    """Test loading aggregated candidate pairs [source1_entity_id, candidate_entity_ids]."""
    tsv_content = "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-10,S3-20\nS1-2\tS2-30\n"
    tsv_file = tmp_path / "aggregated_candidates.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    flat_df = FeatureIO.load_candidate_pairs(tsv_file)
    assert len(flat_df) == 3
    assert list(flat_df.columns) == ["source1_entity_id", "candidate_entity_id"]
    assert set(flat_df["source1_entity_id"]) == {"S1-1", "S1-2"}
    assert set(flat_df["candidate_entity_id"]) == {"S2-10", "S3-20", "S2-30"}


def test_validation_row_preservation_success():
    """Test that feature matrix validation passes when row counts match perfectly."""
    candidates_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10"},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-20"},
    ])
    feature_matrix = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "name_ratio": 0.9, "addr_sim": 0.8},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-20", "name_ratio": 0.1, "addr_sim": 0.2},
    ])

    cfg = Phase3Config()
    report = validate_feature_matrix(candidates_df, feature_matrix, cfg)
    assert report["validation_passed"] is True
    assert report["input_rows"] == 2
    assert report["output_rows"] == 2
    assert report["total_features"] == 2


def test_validation_row_mismatch_failure():
    """Test that validation fails loudly with ValueError if output row count does not match input."""
    candidates_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10"},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-20"},
    ])
    # Unexpected extra row (103,500 style row explosion)
    feature_matrix = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "feat": 1.0},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-20", "feat": 2.0},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-30", "feat": 3.0},
    ])

    cfg = Phase3Config()
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_feature_matrix(candidates_df, feature_matrix, cfg)


def test_validation_duplicate_candidate_pairs_failure():
    """Test that validation fails if duplicate candidate pairs exist."""
    candidates_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10"},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10"},
    ])
    feature_matrix = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "feat": 1.0},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "feat": 1.0},
    ])

    cfg = Phase3Config()
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_matrix(candidates_df, feature_matrix, cfg)


def test_validation_infinite_value_failure():
    """Test that validation fails if infinite values exist."""
    candidates_df = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10"}])
    feature_matrix = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "feat": np.inf}])

    cfg = Phase3Config()
    with pytest.raises(ValueError, match="infinite"):
        validate_feature_matrix(candidates_df, feature_matrix, cfg)


def test_experiment_logger(tmp_path):
    """Test experiment logging to CSV."""
    log_csv = tmp_path / "experiments.csv"
    tracker = ExperimentTracker(log_csv)
    res = tracker.log_experiment(
        experiment_id="EXP-999",
        config_path="configs/phase3.yaml",
        input_artifact="output/phase2/candidate_pairs.tsv",
        feature_set="name + address + cross",
        row_count=500,
        feature_count=38,
        status="SUCCESS",
        notes="Unit test experiment log"
    )

    assert res["experiment_id"] == "EXP-999"
    assert log_csv.exists()

    df = pd.read_csv(log_csv)
    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == "EXP-999"
    assert df.iloc[0]["row_count"] == 500
