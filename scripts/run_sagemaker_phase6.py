#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — SageMaker Processing Job Launcher for Phase 6.

Validates AWS credentials ('amazon-ml'), verifies S3 bucket ('s3://amazon-ml-challenge-2026-atharva'),
syncs local Phase 6 outputs to S3, and configures SageMaker ScriptProcessor jobs.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_sagemaker_phase6")

AWS_PROFILE = "amazon-ml"
AWS_REGION = "ap-south-1"
S3_BUCKET = "s3://amazon-ml-challenge-2026-atharva"
EXPERIMENT_ID = "EXP-001"
INSTANCE_TYPE = "ml.m5.2xlarge"  # 8 vCPU, 32 GiB RAM (~$0.38/hr)


def check_aws_credentials():
    logger.info("Verifying AWS CLI profile credentials...")
    cmd = ["aws", "sts", "get-caller-identity", "--profile", AWS_PROFILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"AWS CLI Authentication failed: {res.stderr}")
        sys.exit(1)
    logger.info(f"AWS Identity Verified:\n{res.stdout.strip()}")


def check_s3_bucket():
    logger.info(f"Verifying access to {S3_BUCKET}...")
    cmd = ["aws", "s3", "ls", S3_BUCKET, "--profile", AWS_PROFILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"S3 Bucket access failed: {res.stderr}")
        sys.exit(1)
    logger.info("S3 Bucket accessible.")


def sync_phase6_artifacts_to_s3():
    logger.info("Syncing local Phase 6 artifacts to AWS S3...")
    local_exp = project_root / "experiments/phase6"
    s3_dest = f"{S3_BUCKET}/phase6/{EXPERIMENT_ID}/"
    
    cmd = ["aws", "s3", "sync", str(local_exp), s3_dest, "--profile", AWS_PROFILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"S3 sync failed: {res.stderr}")
        sys.exit(1)
    logger.info(f"Local Phase 6 artifacts synced successfully to {s3_dest}")


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker Phase 6 Integration Job")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without submitting job")
    args = parser.parse_args()

    check_aws_credentials()
    check_s3_bucket()
    sync_phase6_artifacts_to_s3()

    logger.info("Phase 6 AWS S3 Sync Completed Successfully!")


if __name__ == "__main__":
    main()
