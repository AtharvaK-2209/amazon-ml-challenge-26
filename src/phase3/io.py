"""
io.py — Input/Output and AWS S3 artifact management helpers for Phase 3.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureIO:
    """Handles reading Phase 2 candidate outputs and writing Phase 3 parquets."""

    @staticmethod
    def load_candidate_pairs(filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load candidate pairs TSV file.
        Supports both aggregated format [source1_entity_id, candidate_entity_ids]
        and flat format [source1_entity_id, candidate_entity_id].
        
        Returns flat DataFrame with columns: [source1_entity_id, candidate_entity_id]
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Candidate pair file not found: {path}")

        df = pd.read_csv(path, sep="\t", dtype=str)

        if "candidate_entity_ids" in df.columns:
            # Expand aggregated comma-separated candidate IDs into flat pairs
            rows = []
            for s1_id, cand_str in zip(df["source1_entity_id"], df["candidate_entity_ids"].fillna("")):
                cands = [c.strip() for c in cand_str.split(",") if c.strip()]
                for c_id in cands:
                    rows.append((s1_id, c_id))
            flat_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
            logger.info(f"Expanded aggregated candidate pairs ({len(df):,} S1 rows $\rightarrow$ {len(flat_df):,} flat candidate pairs)")
            return flat_df
        elif "candidate_entity_id" in df.columns:
            return df[["source1_entity_id", "candidate_entity_id"]].copy()
        else:
            raise ValueError(f"Unrecognized candidate_pairs schema. Columns: {list(df.columns)}")

    @staticmethod
    def save_feature_parquet(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
        """Save feature matrix to Parquet format."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file, index=False)
        logger.info(f"Saved feature matrix ({df.shape[0]:,} rows × {df.shape[1]} cols) $\rightarrow$ {out_file}")
        return out_file

    @staticmethod
    def save_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Save metadata JSON document."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature metadata $\rightarrow$ {out_file}")
        return out_file


class S3ArtifactManager:
    """Safe read-only AWS CLI helper utility for S3 artifact management."""

    def __init__(self, bucket: str, profile: str = "amazon-ml", region: str = "ap-south-1"):
        self.bucket = bucket
        self.profile = profile
        self.region = region

    def list_s3_prefix(self, prefix: str = "") -> List[str]:
        """List contents under an S3 prefix using read-only AWS CLI command."""
        s3_url = f"s3://{self.bucket}/{prefix.lstrip('/')}"
        cmd = ["aws", "s3", "ls", s3_url, "--profile", self.profile, "--region", self.region]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
        except Exception as e:
            logger.warning(f"Could not list S3 prefix {s3_url}: {e}")
            return []

    def build_upload_command(self, local_path: str, s3_prefix: str) -> str:
        """Generate the exact AWS CLI command for uploading a local file to S3."""
        target_s3 = f"s3://{self.bucket}/{s3_prefix.lstrip('/')}"
        return f"aws s3 cp {local_path} {target_s3} --profile {self.profile} --region {self.region}"

    def build_download_command(self, s3_path: str, local_path: str) -> str:
        """Generate the exact AWS CLI command for downloading an S3 artifact locally."""
        source_s3 = f"s3://{self.bucket}/{s3_path.lstrip('/')}"
        return f"aws s3 cp {source_s3} {local_path} --profile {self.profile} --region {self.region}"
