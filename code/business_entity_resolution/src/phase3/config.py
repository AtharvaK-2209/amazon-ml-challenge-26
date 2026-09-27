"""
config.py — Configuration loader and data structures for Phase 3 pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class ProjectConfig:
    name: str = "Amazon ML Challenge 2026"
    phase: int = 3
    random_seed: int = 42


@dataclass
class InputConfig:
    candidate_pairs: str = "output/phase2/candidate_pairs.tsv"
    data_dir: str = "data/"
    split: str = "test"


@dataclass
class FeaturesConfig:
    name: bool = True
    address: bool = True
    cross_field: bool = True


@dataclass
class ValidationConfig:
    fail_on_row_mismatch: bool = True
    check_infinite: bool = True
    check_missingness: bool = True
    check_cardinality: bool = True


@dataclass
class OutputConfig:
    local_dir: str = "output/phase3/"
    parquet_file: str = "output/phase3/pair_features.parquet"
    metadata_file: str = "output/phase3/feature_metadata.json"
    format: str = "parquet"


@dataclass
class AWSConfig:
    bucket: str = "amazon-ml-challenge-2026-atharva"
    region: str = "ap-south-1"
    profile: str = "amazon-ml"
    s3_candidates_path: str = "candidates/phase2/candidate_pairs.tsv"
    s3_features_prefix: str = "features/phase3/"


@dataclass
class ExperimentsConfig:
    experiment_id: str = "EXP-001"
    tracking_file: str = "experiments/phase3/experiments.csv"
    s3_prefix: str = "experiments/phase3/"


@dataclass
class Phase3Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input: InputConfig = field(default_factory=InputConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)


def _parse_yaml_fallback(filepath: Path) -> Dict[str, Any]:
    """Basic line parser fallback for simple YAML configs if PyYAML is missing."""
    import yaml
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_phase3_config(config_path: str | Path = "configs/phase3.yaml") -> Phase3Config:
    """
    Load Phase 3 configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Phase3Config instance with typed attributes.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        # Return default configuration if config file is not found
        return Phase3Config()

    data = _parse_yaml_fallback(path) or {}

    proj = data.get("project", {})
    inp = data.get("input", {})
    feats = data.get("features", {})
    val = data.get("validation", {})
    out = data.get("output", {})
    aws = data.get("aws", {})
    exp = data.get("experiments", {})

    cfg = Phase3Config(
        project=ProjectConfig(
            name=proj.get("name", "Amazon ML Challenge 2026"),
            phase=int(proj.get("phase", 3)),
            random_seed=int(proj.get("random_seed", 42)),
        ),
        input=InputConfig(
            candidate_pairs=str(inp.get("candidate_pairs", "output/phase2/candidate_pairs.tsv")),
            data_dir=str(inp.get("data_dir", "data/")),
            split=str(inp.get("split", "test")),
        ),
        features=FeaturesConfig(
            name=bool(feats.get("name", True)),
            address=bool(feats.get("address", True)),
            cross_field=bool(feats.get("cross_field", True)),
        ),
        validation=ValidationConfig(
            fail_on_row_mismatch=bool(val.get("fail_on_row_mismatch", True)),
            check_infinite=bool(val.get("check_infinite", True)),
            check_missingness=bool(val.get("check_missingness", True)),
            check_cardinality=bool(val.get("check_cardinality", True)),
        ),
        output=OutputConfig(
            local_dir=str(out.get("local_dir", "output/phase3/")),
            parquet_file=str(out.get("parquet_file", "output/phase3/pair_features.parquet")),
            metadata_file=str(out.get("metadata_file", "output/phase3/feature_metadata.json")),
            format=str(out.get("format", "parquet")),
        ),
        aws=AWSConfig(
            bucket=str(aws.get("bucket", "amazon-ml-challenge-2026-atharva")),
            region=str(aws.get("region", "ap-south-1")),
            profile=str(aws.get("profile", "amazon-ml")),
            s3_candidates_path=str(aws.get("s3_candidates_path", "candidates/phase2/candidate_pairs.tsv")),
            s3_features_prefix=str(aws.get("s3_features_prefix", "features/phase3/")),
        ),
        experiments=ExperimentsConfig(
            experiment_id=str(exp.get("experiment_id", "EXP-001")),
            tracking_file=str(exp.get("tracking_file", "experiments/phase3/experiments.csv")),
            s3_prefix=str(exp.get("s3_prefix", "experiments/phase3/")),
        ),
        raw_dict=data,
    )
    return cfg
