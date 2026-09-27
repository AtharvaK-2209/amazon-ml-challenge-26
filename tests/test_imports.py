"""
Test module imports for verification.

This test verifies that all modules can be imported successfully
and that the project structure is correct.
"""

import pytest
from pathlib import Path


class TestImports:
    """Test that all modules import successfully."""
    
    def test_import_config(self):
        """Test config module imports."""
        import src.config as config
        assert hasattr(config, "MODEL")
        assert hasattr(config, "BLOCKING")
        assert hasattr(config, "NORM")
        assert hasattr(config, "FEATURES")
    
    def test_import_preprocessing(self):
        """Test preprocessing modules import."""
        from src.preprocessing.normalize import Normalizer
        from src.preprocessing.address_parser import parse_address
        assert Normalizer is not None
        assert parse_address is not None
    
    def test_import_blocking(self):
        """Test blocking modules import."""
        from src.blocking.exact_blocking import exact_block
        from src.blocking.token_blocking import sorted_neighbourhood_block
        from src.blocking.tfidf_blocking import tfidf_block
        from src.blocking.faiss_blocking import faiss_block
        assert exact_block is not None
        assert sorted_neighbourhood_block is not None
        assert tfidf_block is not None
        assert faiss_block is not None
    
    def test_import_features(self):
        """Test feature modules import."""
        from src.features.name_features import NameFeatureExtractor, extract_name_features
        from src.features.address_features import AddressFeatureExtractor, extract_address_features
        from src.features.pair_features import PairFeatureExtractor, build_feature_matrix
        assert NameFeatureExtractor is not None
        assert extract_name_features is not None
        assert AddressFeatureExtractor is not None
        assert extract_address_features is not None
        assert PairFeatureExtractor is not None
        assert build_feature_matrix is not None
    
    def test_import_models(self):
        """Test model modules import."""
        from src.models.train_xgb import train
        from src.models.predict import predict_matches
        assert train is not None
        assert predict_matches is not None
    
    def test_import_decision(self):
        """Test decision modules import."""
        from src.decision.threshold import sweep_thresholds
        from src.decision.margin import apply_margin_filter
        from src.decision.singleton import build_full_submission
        assert sweep_thresholds is not None
        assert apply_margin_filter is not None
        assert build_full_submission is not None
    
    def test_import_evaluation(self):
        """Test evaluation modules import."""
        from src.evaluation.evaluate_f05 import evaluate_f05
        assert evaluate_f05 is not None
    
    def test_import_pipeline(self):
        """Test pipeline module imports."""
        from src.pipeline import run_pipeline
        assert run_pipeline is not None


class TestProjectStructure:
    """Test that required directories and files exist."""
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        required_dirs = [
            'data',
            'data/train',
            'data/test',
            'src',
            'src/preprocessing',
            'src/blocking',
            'src/features',
            'src/models',
            'src/decision',
            'src/evaluation',
            'notebooks',
            'experiments',
            'output',
            'configs',
            'tests'
        ]
        
        for dir_path in required_dirs:
            path = Path(__file__).parent.parent / dir_path
            assert path.exists(), f"Directory {dir_path} does not exist"
            assert path.is_dir(), f"{dir_path} is not a directory"
    
    def test_required_files(self):
        """Test that required files exist."""
        required_files = [
            'src/__init__.py',
            'src/config.py',
            'src/pipeline.py',
            'src/preprocessing/__init__.py',
            'src/blocking/__init__.py',
            'src/features/__init__.py',
            'src/models/__init__.py',
            'src/decision/__init__.py',
            'src/evaluation/__init__.py',
            'configs/config.yaml',
            'requirements.txt',
            'README.md',
            'PROJECT_RULES.md',
            '.gitignore',
            '.env.example'
        ]
        
        for file_path in required_files:
            path = Path(__file__).parent.parent / file_path
            assert path.exists(), f"File {file_path} does not exist"
            assert path.is_file(), f"{file_path} is not a file"
    
    def test_placeholder_modules_exist(self):
        """Test that all modules exist."""
        modules = [
            'src/preprocessing/normalize.py',
            'src/preprocessing/address_parser.py',
            'src/blocking/exact_blocking.py',
            'src/blocking/token_blocking.py',
            'src/blocking/tfidf_blocking.py',
            'src/blocking/faiss_blocking.py',
            'src/features/name_features.py',
            'src/features/address_features.py',
            'src/features/pair_features.py',
            'src/models/train_xgb.py',
            'src/models/predict.py',
            'src/decision/threshold.py',
            'src/decision/margin.py',
            'src/decision/singleton.py',
            'src/evaluation/evaluate_f05.py'
        ]
        
        for module_path in modules:
            path = Path(__file__).parent.parent / module_path
            assert path.exists(), f"Module {module_path} does not exist"
            assert path.is_file(), f"{module_path} is not a file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
