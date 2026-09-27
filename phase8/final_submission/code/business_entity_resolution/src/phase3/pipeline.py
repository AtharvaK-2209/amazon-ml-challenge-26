"""
pipeline.py — Main Phase 3 Feature Engineering Pipeline & Validation Suite.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.phase3.config import Phase3Config, load_phase3_config
from src.phase3.experiment import ExperimentTracker
from src.phase3.io import FeatureIO, S3ArtifactManager
from src.preprocessing.normalize import Normalizer
from src.features.pair_features import build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("src.phase3.pipeline")


def validate_feature_matrix(
    candidate_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    cfg: Phase3Config
) -> Dict[str, Any]:
    """
    Validate Phase 3 feature matrix.
    FAILS LOUDLY if row count changes or critical data corruption occurs.
    """
    input_rows = len(candidate_df)
    output_rows = len(feature_matrix)

    # 1 & 4. Strict Row Count Preservation check
    if input_rows != output_rows:
        raise ValueError(
            f"[FATAL VALIDATION FAILURE] Row count mismatch! Candidate input has {input_rows:,} rows, "
            f"but output feature matrix has {output_rows:,} rows. Accidental row multiplication or dropping detected."
        )

    # 2. Number of columns check
    num_cols = len(feature_matrix.columns)
    if num_cols == 0:
        raise ValueError("[FATAL VALIDATION FAILURE] Feature matrix has 0 columns!")

    # 3. Candidate-pair ID preservation
    if "source1_entity_id" not in feature_matrix.columns or "candidate_entity_id" not in feature_matrix.columns:
        raise ValueError("[FATAL VALIDATION FAILURE] Output feature matrix is missing source1_entity_id or candidate_entity_id columns!")

    # 5. Duplicate candidate pairs check
    dup_mask = feature_matrix.duplicated(subset=["source1_entity_id", "candidate_entity_id"])
    num_duplicates = int(dup_mask.sum())
    if num_duplicates > 0:
        raise ValueError(f"[FATAL VALIDATION FAILURE] Found {num_duplicates} duplicate (source1_entity_id, candidate_entity_id) pairs in feature matrix!")

    feature_cols = [c for c in feature_matrix.columns if c not in ("source1_entity_id", "candidate_entity_id")]

    # 6. Missing feature values check
    null_counts = feature_matrix[feature_cols].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())

    # 7. Infinite values check
    numeric_df = feature_matrix[feature_cols].select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df.values).sum()
    if cfg.validation.check_infinite and inf_counts > 0:
        raise ValueError(f"[FATAL VALIDATION FAILURE] Found {inf_counts} infinite values (inf / -inf) in feature matrix!")

    # 8. Numeric vs categorical check
    non_numeric_cols = [c for c in feature_cols if not np.issubdtype(feature_matrix[c].dtype, np.number)]
    
    # 9. Constant value features check
    constant_features = [c for c in feature_cols if feature_matrix[c].nunique(dropna=False) <= 1]

    # 10. High-cardinality features check
    high_card_features = [c for c in feature_cols if feature_matrix[c].nunique() > 0.95 * output_rows and output_rows > 100]

    report = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "total_features": len(feature_cols),
        "total_nulls": total_nulls,
        "infinite_values": int(inf_counts),
        "non_numeric_features": non_numeric_cols,
        "constant_features": constant_features,
        "high_cardinality_features": high_card_features,
        "validation_passed": True,
    }

    logger.info(f"Validation PASSED: {output_rows:,} rows × {len(feature_cols)} numeric features | Nulls: {total_nulls} | Infs: {inf_counts}")
    if constant_features:
        logger.warning(f"Constant features detected (consider dropping): {constant_features}")
    return report


class Phase3Pipeline:
    """End-to-end reproducible Phase 3 feature engineering pipeline."""

    def __init__(self, config_path: str | Path = "configs/phase3.yaml"):
        self.cfg = load_phase3_config(config_path)
        self.config_path_str = str(config_path)
        self.tracker = ExperimentTracker(self.cfg.experiments.tracking_file)
        self.s3_mgr = S3ArtifactManager(
            bucket=self.cfg.aws.bucket,
            profile=self.cfg.aws.profile,
            region=self.cfg.aws.region
        )

    def load_enriched_candidates(self) -> pd.DataFrame:
        """
        Load candidate pairs and join with normalized S1 and S2/S3 text representations.
        """
        cand_path = Path(self.cfg.input.candidate_pairs)
        if not cand_path.exists():
            raise FileNotFoundError(f"Candidate pairs file does not exist at {cand_path}")

        candidates = FeatureIO.load_candidate_pairs(cand_path)

        data_dir = Path(self.cfg.input.data_dir)
        split = self.cfg.input.split

        s1_path = data_dir / split / f"{split}_source1.tsv"
        s2_path = data_dir / split / f"{split}_source2.tsv"
        s3_path = data_dir / split / f"{split}_source3.tsv"

        if not s1_path.exists():
            # Fallback to train if test split not found or specified
            s1_path = data_dir / "train" / "train_source1.tsv"
            s2_path = data_dir / "train" / "train_source2.tsv"
            s3_path = data_dir / "train" / "train_source3.tsv"

        logger.info(f"Loading raw sources from {s1_path.parent} ...")
        s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
        s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
        s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

        s1_needed = set(candidates["source1_entity_id"])
        cand_needed = set(candidates["candidate_entity_id"])

        s1 = s1[s1["entity_id"].isin(s1_needed)].copy()
        s2 = s2[s2["entity_id"].isin(cand_needed)].copy()
        s3 = s3[s3["entity_id"].isin(cand_needed)].copy()

        norm = Normalizer()
        s1 = norm.normalize_dataframe(s1)
        s2 = norm.normalize_dataframe(s2)
        s3 = norm.normalize_dataframe(s3)
        s23 = pd.concat([s2, s3], ignore_index=True)

        s1_lookup = s1.set_index("entity_id")[["normalized_name", "normalized_address", "country"]]
        s23_lookup = s23.set_index("entity_id")[["normalized_name", "normalized_address", "country"]]

        enriched = candidates.join(s1_lookup.add_suffix("_s1"), on="source1_entity_id")
        enriched = enriched.join(s23_lookup.add_suffix("_cand"), on="candidate_entity_id")

        return enriched

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete Phase 3 pipeline.
        
        Returns:
            Tuple of (full feature matrix DataFrame, metadata dict).
        """
        logger.info(f"=== Starting Phase 3 Pipeline [{self.cfg.experiments.experiment_id}] ===")

        # 1. Load candidate pairs
        enriched_candidates = self.load_enriched_candidates()
        input_rows = len(enriched_candidates)

        # 2. Build feature matrix using teammate feature engineering modules
        logger.info(f"Building feature matrix for {input_rows:,} candidate pairs ...")
        X = build_feature_matrix(enriched_candidates)

        # Combine IDs with features
        feature_df = pd.concat([
            enriched_candidates[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True),
            X.reset_index(drop=True)
        ], axis=1)

        # 3. Validate feature matrix
        val_report = validate_feature_matrix(enriched_candidates, feature_df, self.cfg)

        # 4. Save outputs (Parquet + Metadata)
        parquet_path = FeatureIO.save_feature_parquet(feature_df, self.cfg.output.parquet_file)

        feature_cols = [c for c in feature_df.columns if c not in ("source1_entity_id", "candidate_entity_id")]
        metadata = {
            "experiment_id": self.cfg.experiments.experiment_id,
            "phase": 3,
            "timestamp": pd.Timestamp.now().isoformat(),
            "config_path": self.config_path_str,
            "input_candidate_file": self.cfg.input.candidate_pairs,
            "input_rows": input_rows,
            "output_rows": len(feature_df),
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "validation_report": val_report,
            "random_seed": self.cfg.project.random_seed,
            "s3_upload_command": self.s3_mgr.build_upload_command(
                str(parquet_path),
                f"{self.cfg.aws.s3_features_prefix}{self.cfg.experiments.experiment_id}/pair_features.parquet"
            )
        }
        meta_path = FeatureIO.save_metadata(metadata, self.cfg.output.metadata_file)

        # 5. Log experiment
        feature_set_str = "name + address + cross" if (self.cfg.features.name and self.cfg.features.address and self.cfg.features.cross_field) else "custom"
        self.tracker.log_experiment(
            experiment_id=self.cfg.experiments.experiment_id,
            config_path=self.config_path_str,
            input_artifact=self.cfg.input.candidate_pairs,
            feature_set=feature_set_str,
            row_count=len(feature_df),
            feature_count=len(feature_cols),
            status="SUCCESS",
            notes="Phase 3 feature matrix generated and validated"
        )

        logger.info(f"=== Phase 3 Pipeline Completed Successfully ===")
        logger.info(f"Local Parquet: {parquet_path}")
        logger.info(f"To upload to S3, run:\n  {metadata['s3_upload_command']}\n")

        return feature_df, metadata


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/phase3.yaml"
    pipeline = Phase3Pipeline(cfg_file)
    pipeline.run()
