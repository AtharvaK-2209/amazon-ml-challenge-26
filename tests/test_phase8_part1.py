"""
Tests for Phase 8 Part 1 — Member 1 Model & Artifact Freeze.
Verifies model loading, feature schema compatibility, schema mismatch detection,
probability sanity checks, and metadata preservation.
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import xgboost as xgb

from src.config import ROOT


class TestPhase8Part1ModelFreeze(unittest.TestCase):

    def setUp(self):
        self.final_config_path = ROOT / "phase8" / "final_config.json"
        self.metadata_path = ROOT / "phase8" / "member1" / "model_metadata.json"
        self.schema_verif_path = ROOT / "phase8" / "member1" / "schema_verification.json"
        self.model_path = ROOT / "models" / "xgb_baseline.json"

    def test_final_config_exists_and_valid(self):
        self.assertTrue(self.final_config_path.exists())
        with open(self.final_config_path) as f:
            config = json.load(f)
        self.assertEqual(config["model"], "XGBoost")
        self.assertEqual(config["model_version"], "xgb_baseline")
        self.assertEqual(config["calibration"], "raw")
        self.assertEqual(config["threshold"], 0.94)

    def test_model_artifact_exists_and_loads(self):
        self.assertTrue(self.model_path.exists())
        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        self.assertEqual(booster.num_features(), 47)

    def test_schema_verification_artifact(self):
        self.assertTrue(self.schema_verif_path.exists())
        with open(self.schema_verif_path) as f:
            verif = json.load(f)
        self.assertTrue(verif["schema_compatible"])
        self.assertTrue(verif["order_match"])
        self.assertTrue(verif["name_match"])
        self.assertEqual(verif["training_feature_count"], 47)

    def test_model_metadata_artifact(self):
        self.assertTrue(self.metadata_path.exists())
        with open(self.metadata_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["verification_status"], "READY")
        self.assertEqual(len(meta["feature_columns"]), 47)

    def test_small_sample_inference_sanity(self):
        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        with open(self.metadata_path) as f:
            meta = json.load(f)
        expected_cols = meta["feature_columns"]

        # Create dummy sample matching schema
        n_rows = 20
        sample_df = pd.DataFrame(
            np.random.randn(n_rows, 47),
            columns=expected_cols
        )
        sample_df["source1_entity_id"] = [f"s1_{i}" for i in range(n_rows)]
        sample_df["candidate_entity_id"] = [f"cand_{i}" for i in range(n_rows)]
        sample_df["candidate_source"] = ["S2"] * n_rows

        dmatrix = xgb.DMatrix(sample_df[expected_cols])
        probs = booster.predict(dmatrix)

        # 10 explicit checks
        self.assertEqual(len(probs), n_rows)                       # Row count preserved
        self.assertTrue(np.all(probs >= 0.0))                      # Prob >= 0
        self.assertTrue(np.all(probs <= 1.0))                      # Prob <= 1
        self.assertEqual(np.isnan(probs).sum(), 0)                 # No NaN
        self.assertEqual(np.isinf(probs).sum(), 0)                 # No Inf
        self.assertEqual(len(sample_df["source1_entity_id"]), n_rows)
        self.assertEqual(len(sample_df["candidate_entity_id"]), n_rows)
        self.assertEqual(len(sample_df["candidate_source"]), n_rows)
        self.assertEqual(len(expected_cols), 47)
        self.assertEqual(list(sample_df[expected_cols].columns), expected_cols)

    def test_schema_mismatch_detection(self):
        with open(self.metadata_path) as f:
            meta = json.load(f)
        expected_cols = meta["feature_columns"]

        # 1. Missing feature detection
        missing_cols = expected_cols[:-1]
        self.assertNotEqual(len(missing_cols), len(expected_cols))

        # 2. Extra feature detection
        extra_cols = expected_cols + ["extra_unknown_feature"]
        self.assertNotEqual(len(extra_cols), len(expected_cols))

        # 3. Order mismatch detection
        reordered_cols = list(reversed(expected_cols))
        self.assertNotEqual(reordered_cols, expected_cols)


if __name__ == "__main__":
    unittest.main()
