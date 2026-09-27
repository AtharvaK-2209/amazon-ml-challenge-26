"""
Test configuration loading and validation.

This test verifies that the configuration system works correctly
and that no secrets are hard-coded.
"""

import pytest
from pathlib import Path
import os

from src.config import Config, load_config, get_config


class TestConfig:
    """Test configuration functionality."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert config is not None
        assert isinstance(config, Config)
    
    def test_config_has_random_seed(self):
        """Test that configuration has random seed."""
        config = load_config()
        assert hasattr(config, 'random_seed')
        assert isinstance(config.random_seed, int)
    
    def test_config_has_data_paths(self):
        """Test that configuration has data paths."""
        config = load_config()
        assert hasattr(config, 'data')
        assert hasattr(config.data, 'train_dir')
        assert hasattr(config.data, 'test_dir')
        assert hasattr(config.data, 'output_dir')
    
    def test_config_paths_are_pathlib(self):
        """Test that paths are pathlib.Path objects."""
        config = load_config()
        assert isinstance(config.data.train_dir, Path)
        assert isinstance(config.data.test_dir, Path)
        assert isinstance(config.data.output_dir, Path)
    
    def test_config_has_blocking_settings(self):
        """Test that configuration has blocking settings."""
        config = load_config()
        assert hasattr(config, 'blocking')
        assert hasattr(config.blocking, 'passes')
    
    def test_config_has_decision_thresholds(self):
        """Test that configuration has decision thresholds."""
        config = load_config()
        assert hasattr(config, 'decision')
        assert hasattr(config.decision, 'match_threshold')
        assert hasattr(config.decision, 'margin_threshold')
        assert 0.0 <= config.decision.match_threshold <= 1.0
    
    def test_get_config_returns_global_instance(self):
        """Test that get_config returns global instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_config_yaml_loads(self):
        """Test that config.yaml loads successfully."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        assert config_path.exists(), "config.yaml not found"
        
        config = load_config(config_path)
        assert config is not None


class TestNoSecrets:
    """Test that no secrets are hard-coded in the project."""
    
    def test_no_aws_keys_in_code(self):
        """Test that AWS keys are not hard-coded."""
        # Read all Python files
        src_dir = Path(__file__).parent.parent / "src"
        
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            
            # Check for common secret patterns
            assert "AWS_ACCESS_KEY_ID=" not in content, \
                f"Found AWS_ACCESS_KEY_ID in {py_file}"
            assert "AWS_SECRET_ACCESS_KEY=" not in content, \
                f"Found AWS_SECRET_ACCESS_KEY in {py_file}"
            assert "AKIA" not in content, \
                f"Found potential AWS key ID pattern in {py_file}"
    
    def test_no_secrets_in_config_yaml(self):
        """Test that config.yaml doesn't contain secrets."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        content = config_path.read_text()
        
        # Should use environment variable placeholders
        assert "AKIA" not in content, "Found potential AWS key in config.yaml"
        assert "aws_access_key" not in content.lower(), \
            "Found aws_access_key in config.yaml"
    
    def test_env_example_has_placeholders(self):
        """Test that .env.example has only placeholders."""
        env_example_path = Path(__file__).parent.parent / ".env.example"
        
        if env_example_path.exists():
            content = env_example_path.read_text()
            
            # Should not have actual values
            assert "AWS_ACCESS_KEY_ID=\n" in content or "AWS_ACCESS_KEY_ID= " in content, \
                ".env.example should have empty AWS_ACCESS_KEY_ID"
    
    def test_gitignore_excludes_env(self):
        """Test that .gitignore excludes .env files."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        content = gitignore_path.read_text()
        
        assert ".env" in content, ".gitignore should exclude .env files"


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_env_var_expansion(self):
        """Test that environment variables are expanded in config."""
        # Set a test environment variable
        os.environ["TEST_VAR"] = "test_value_123"
        
        # This would be used by config loader
        from src.config import _expand_env_vars
        
        test_dict = {
            "key1": "${TEST_VAR}",
            "key2": "static_value",
            "nested": {
                "key3": "${TEST_VAR}"
            }
        }
        
        expanded = _expand_env_vars(test_dict)
        
        assert expanded["key1"] == "test_value_123"
        assert expanded["key2"] == "static_value"
        assert expanded["nested"]["key3"] == "test_value_123"
        
        # Clean up
        del os.environ["TEST_VAR"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
