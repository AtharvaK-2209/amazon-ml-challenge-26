"""
experiment.py — Experiment tracker and metadata logger for Phase 3.
"""

import csv
import datetime
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash or short string."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ExperimentTracker:
    """Tracks and registers Phase 3 feature engineering experiment metadata."""

    def __init__(self, tracking_csv_path: str | Path = "experiments/phase3/experiments.csv"):
        self.csv_path = Path(tracking_csv_path).resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Create experiments.csv with header if it does not exist."""
        headers = [
            "experiment_id",
            "timestamp",
            "git_commit",
            "config",
            "input_artifact",
            "feature_set",
            "row_count",
            "feature_count",
            "status",
            "notes"
        ]
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_experiment(
        self,
        experiment_id: str,
        config_path: str,
        input_artifact: str,
        feature_set: str,
        row_count: int,
        feature_count: int,
        status: str = "SUCCESS",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Record experiment entry to experiments.csv.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        git_hash = get_git_commit_hash()

        row = [
            experiment_id,
            timestamp,
            git_hash,
            config_path,
            input_artifact,
            feature_set,
            str(row_count),
            str(feature_count),
            status,
            notes
        ]

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.info(f"Logged experiment {experiment_id} to {self.csv_path}")
        return {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "git_commit": git_hash,
            "config": config_path,
            "input_artifact": input_artifact,
            "feature_set": feature_set,
            "row_count": row_count,
            "feature_count": feature_count,
            "status": status,
            "notes": notes,
        }
