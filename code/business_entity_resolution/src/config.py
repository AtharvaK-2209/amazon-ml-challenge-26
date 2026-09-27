"""
config.py — Central configuration for the Amazon ML Challenge pipeline.
All paths, constants and hyperparameters live here.
Import with: from src.config import CFG
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root

# ── Data paths ─────────────────────────────────────────────
DATA_DIR   = ROOT if (ROOT / "train").exists() else ROOT / "student_resource" / "dataset"
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
EDA_DIR    = ROOT / "eda_output"
OUTPUT_DIR   = ROOT / "output"
FEATURES_DIR = ROOT / "features"
CONFIGS_DIR  = ROOT / "configs"
MODELS_DIR   = ROOT / "src" / "models" / "saved"

TRAIN_S1  = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2  = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3  = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT  = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1   = TEST_DIR / "test_source1.tsv"
TEST_S2   = TEST_DIR / "test_source2.tsv"
TEST_S3   = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING   = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

NOISE_SPEC = EDA_DIR / "noise_spec.json"

# ── Load noise spec (if available) ─────────────────────────
def load_noise_spec() -> dict:
    if NOISE_SPEC.exists():
        with open(NOISE_SPEC) as f:
            return json.load(f)
    return {}

# ── Normalisation config ────────────────────────────────────
NORM = {
    "legal_suffix_expansions": {
        "pvt": "private", "ltd": "limited", "llc": "limited liability company",
        "corp": "corporation", "inc": "incorporated", "co": "company",
        "llp": "limited liability partnership", "plc": "public limited company",
        "huf": "hindu undivided family", "sarl": "societe a responsabilite limitee",
        "sas": "societe par actions simplifiee", "sa": "societe anonyme",
    },
    "address_abbreviation_expansions": {
        "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
        "dr": "drive", "ln": "lane", "hwy": "highway",
        "apt": "apartment", "ste": "suite", "fl": "floor",
        "tq": "taluka", "dist": "district", "opp": "opposite",
        "extn": "extension",
    },
}

# ── Blocking config ─────────────────────────────────────────
BLOCKING = {
    "tfidf_ngram_range": (3, 3),
    "tfidf_analyzer": "char_wb",
    "tfidf_top_k": 50,
    "tfidf_min_score": 0.10,
    "sorted_neighbourhood_window": 10,
    "address_idf_threshold": 3.0,
    "address_min_token_len": 4,
    "partition_key": "country",          # hard country partition
}

# ── Feature config ──────────────────────────────────────────
FEATURES = {
    "rapidfuzz_scorers": ["ratio", "WRatio", "token_sort_ratio", "token_set_ratio"],
    "name_tfidf_max_features": 50000,
    "addr_tfidf_max_features": 50000,
}

# ── Model config ────────────────────────────────────────────
MODEL = {
    "xgb_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    },
    "validation_split": 0.20,   # 80/20 split by S1 entity
    "random_seed": 42,
}

# ── Decision engine config ──────────────────────────────────
DECISION = {
    "threshold_sweep": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "default_threshold": 0.70,
    "singleton_strategy": "no_match_if_max_score_below_threshold",
}

# ── Evaluation config ───────────────────────────────────────
EVAL = {
    "beta": 0.5,    # F_0.5
    "macro_average": True,
}

# ── Config classes & compatibility helpers for tests ──────────
import os
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    output_dir: Path = OUTPUT_DIR

@dataclass
class BlockingConfig:
    passes: list = field(default_factory=lambda: ["tfidf", "exact", "snb"])

@dataclass
class DecisionConfig:
    match_threshold: float = DECISION["default_threshold"]
    margin_threshold: float = 0.05

@dataclass
class Config:
    random_seed: int = MODEL["random_seed"]
    data: DataConfig = field(default_factory=DataConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

_global_config: Config | None = None

def load_config(config_path: Path | str | None = None) -> Config:
    global _global_config
    _global_config = Config()
    return _global_config

def get_config() -> Config:
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config

def _expand_env_vars(val: dict | list | str | None) -> dict | list | str | None:
    if isinstance(val, dict):
        return {k: _expand_env_vars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_expand_env_vars(v) for v in val]
    elif isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        var_name = val[2:-1]
        return os.environ.get(var_name, val)
    return val

