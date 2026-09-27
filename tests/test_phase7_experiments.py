"""
Tests for Phase 7 Member 1 — Feature & Model Experiments.
Verifies that evaluation metrics calculation, training helpers, and experiment runner
functions operate cleanly and return expected schemas and values.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.experiments.phase7_experiments import (
    compute_metrics,
    train_and_eval_xgb,
    train_and_eval_lgbm,
    write_report
)


class TestPhase7Experiments(unittest.TestCase):

    def setUp(self):
        # Create dummy labeled data with 5-fold splits
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "record_id_1": [f"r{i}" for i in range(n)],
            "record_id_2": [f"c{i}" for i in range(n)],
            "source1_entity_id": [f"s1_{i%10}" for i in range(n)],
            "candidate_entity_id": [f"c_{i}" for i in range(n)],
            "true_label": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        })
        self.feature_cols = ["feat_1", "feat_2", "feat_3"]

    def test_compute_metrics(self):
        y_true = np.array([1, 1, 0, 0])
        probs = np.array([0.98, 0.95, 0.12, 0.05])
        metrics = compute_metrics(y_true, probs, threshold=0.94)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f0_5"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertGreater(metrics["pr_auc"], 0.9)

    def test_train_and_eval_xgb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            metrics, oof_probs, y_true, feat_imp_df, models = train_and_eval_xgb(
                df=self.df,
                feature_cols=self.feature_cols,
                exp_dir=exp_dir,
                exp_id="test_xgb"
            )
            self.assertIn("f0_5", metrics)
            self.assertIn("pr_auc", metrics)
            self.assertEqual(len(oof_probs), len(y_true))
            self.assertTrue((exp_dir / "metrics.json").exists())
            self.assertTrue((exp_dir / "feature_importance.csv").exists())
            self.assertTrue((exp_dir / "predictions.parquet").exists())

    def test_train_and_eval_lgbm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            metrics, oof_probs, y_true, feat_imp_df, models = train_and_eval_lgbm(
                df=self.df,
                feature_cols=self.feature_cols,
                exp_dir=exp_dir,
                exp_id="test_lgbm"
            )
            self.assertIn("f0_5", metrics)
            self.assertIn("pr_auc", metrics)
            self.assertEqual(len(oof_probs), len(y_true))
            self.assertTrue((exp_dir / "metrics.json").exists())

    def test_write_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            metrics = {
                "precision": 0.99, "recall": 0.98, "f0_5": 0.988, "f1": 0.985,
                "pr_auc": 0.999, "roc_auc": 0.9999, "runtime_sec": 0.5,
                "false_positives": 1, "false_negatives": 2,
                "dataset_size": 100, "positives": 30
            }
            baseline = {
                "precision": 0.95, "recall": 0.95, "f0_5": 0.95, "f1": 0.95,
                "pr_auc": 0.99, "roc_auc": 0.99,
                "false_positives": 2, "false_negatives": 3
            }
            write_report(
                exp_dir=exp_dir,
                exp_id="E03",
                objective="Test address features",
                changes="+3 address features",
                features=["feat_1", "feat_2"],
                model_str="XGBoost",
                metrics=metrics,
                baseline_metrics=baseline,
                runtime=0.5,
                errors_desc="No major errors",
                conclusion="REJECT — no clear gain"
            )
            self.assertTrue((exp_dir / "experiment_report.md").exists())


if __name__ == "__main__":
    unittest.main()
