"""
Configuration management for Amazon ML Challenge 2026 Entity Resolution Pipeline.

This module provides centralized configuration loading and management,
supporting both local development and AWS deployment scenarios.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data path configuration."""
    train_dir: Path
    test_dir: Path
    output_dir: Path
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None


@dataclass
class BlockingConfig:
    """Blocking strategy configuration."""
    passes: List[Dict[str, Any]] = field(default_factory=list)
    tfidf: Dict[str, Any] = field(default_factory=dict)
    faiss: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    name_features: List[str] = field(default_factory=list)
    address_features: List[str] = field(default_factory=list)
    pair_features: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model training and prediction configuration."""
    xgboost: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionConfig:
    """Decision engine configuration."""
    match_threshold: float = 0.85
    margin_threshold: float = 0.15
    singleton_detection: bool = True
    singleton_threshold: float = 0.3


@dataclass
class Config:
    """Main configuration container."""
    random_seed: int = 42
    data: DataConfig = None
    blocking: BlockingConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    decision: DecisionConfig = None
    evaluation: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    aws_region: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.data is None:
            self.data = DataConfig(
                train_dir=Path("data/train"),
                test_dir=Path("data/test"),
                output_dir=Path("output")
            )


def _expand_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in configuration values.
    
    Supports ${ENV_VAR} syntax in YAML configuration.
    
    Args:
        config_dict: Configuration dictionary to process
        
    Returns:
        Configuration dictionary with environment variables expanded
    """
    if isinstance(config_dict, dict):
        return {k: _expand_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_expand_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        if config_dict.startswith("${") and config_dict.endswith("}"):
            env_var = config_dict[2:-1]
            return os.getenv(env_var, "")
        return config_dict
    else:
        return config_dict


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        yaml.YAMLError: If configuration file is invalid
    """
    if config_path is None:
        # Default configuration path
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found at {config_path}, using defaults")
        return Config()
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)
    
    # Parse configuration sections
    data_config = DataConfig(
        train_dir=Path(config_dict.get("data", {}).get("train_dir", "data/train")),
        test_dir=Path(config_dict.get("data", {}).get("test_dir", "data/test")),
        output_dir=Path(config_dict.get("data", {}).get("output_dir", "output")),
        use_s3=config_dict.get("data", {}).get("use_s3", False),
        s3_bucket=config_dict.get("data", {}).get("s3_bucket"),
        s3_prefix=config_dict.get("data", {}).get("s3_prefix")
    )
    
    blocking_dict = config_dict.get("blocking", {})
    blocking_config = BlockingConfig(
        passes=blocking_dict.get("passes", []),
        tfidf=blocking_dict.get("tfidf", {}),
        faiss=blocking_dict.get("faiss", {})
    )
    
    features_dict = config_dict.get("features", {})
    feature_config = FeatureConfig(
        name_features=features_dict.get("name_features", []),
        address_features=features_dict.get("address_features", []),
        pair_features=features_dict.get("pair_features", [])
    )
    
    model_config = ModelConfig(
        xgboost=config_dict.get("model", {}).get("xgboost", {}),
        calibration=config_dict.get("model", {}).get("calibration", {})
    )
    
    decision_dict = config_dict.get("decision", {})
    decision_config = DecisionConfig(
        match_threshold=decision_dict.get("match_threshold", 0.85),
        margin_threshold=decision_dict.get("margin_threshold", 0.15),
        singleton_detection=decision_dict.get("singleton_detection", True),
        singleton_threshold=decision_dict.get("singleton_threshold", 0.3)
    )
    
    config = Config(
        random_seed=config_dict.get("random_seed", 42),
        data=data_config,
        blocking=blocking_config,
        features=feature_config,
        model=model_config,
        decision=decision_config,
        evaluation=config_dict.get("evaluation", {}),
        logging=config_dict.get("logging", {}),
        aws_region=config_dict.get("aws_region")
    )
    
    logger.info("Configuration loaded successfully")
    return config


def setup_logging(config: Config) -> None:
    """
    Setup logging configuration based on config settings.
    
    Args:
        config: Configuration object with logging settings
    """
    log_config = config.logging
    
    level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    handlers = []
    
    if log_config.get("log_to_console", True):
        handlers.append(logging.StreamHandler())
    
    if log_config.get("log_to_file", True):
        log_file = Path(log_config.get("log_file", "logs/pipeline.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    logger.info("Logging configured successfully")


# Global configuration instance
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance.
    
    Args:
        reload: Force reload configuration from file
        
    Returns:
        Global Config instance
    """
    global _config
    
    if _config is None or reload:
        _config = load_config()
    
    return _config
