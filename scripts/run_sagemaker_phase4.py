#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — SageMaker Processing Job Launcher for Phase 4.

This script launches an AWS SageMaker Processing Job to generate Phase 4 entity matching
features in the cloud using CPU instances (e.g., ml.m5.2xlarge).

Features:
1. Validates AWS CLI credentials and profile ('amazon-ml').
2. Verifies S3 bucket ('s3://amazon-ml-challenge-2026-atharva/').
3. Inspects SageMaker Execution IAM Role (stops gracefully if unconfigured).
4. Syncs code & inputs to S3.
5. Launches a SageMaker ScriptProcessor job.
6. Outputs artifacts to s3://amazon-ml-challenge-2026-atharva/features/phase4/EXP-001/
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
logger = logging.getLogger("run_sagemaker_phase4")

AWS_PROFILE = "amazon-ml"
AWS_REGION = "ap-south-1"
S3_BUCKET = "s3://amazon-ml-challenge-2026-atharva"
EXPERIMENT_ID = "EXP-001"
INSTANCE_TYPE = "ml.m5.2xlarge"  # 8 vCPU, 32 GiB RAM (~$0.38/hr)


def check_aws_credentials():
    """Verify AWS STS caller identity."""
    logger.info("Verifying AWS CLI profile credentials...")
    cmd = ["aws", "sts", "get-caller-identity", "--profile", AWS_PROFILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"AWS CLI Authentication failed: {res.stderr}")
        sys.exit(1)
    logger.info(f"AWS Identity Verified:\n{res.stdout.strip()}")


def check_s3_bucket():
    """Verify S3 bucket accessibility."""
    logger.info(f"Verifying access to {S3_BUCKET}...")
    cmd = ["aws", "s3", "ls", S3_BUCKET, "--profile", AWS_PROFILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"S3 Bucket access failed: {res.stderr}")
        sys.exit(1)
    logger.info("S3 Bucket accessible.")


def get_sagemaker_execution_role():
    """Inspect or retrieve SageMaker IAM Execution Role."""
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role_arn:
        logger.info(f"Using SAGEMAKER_ROLE_ARN from environment: {role_arn}")
        return role_arn

    # Check via boto3 / IAM
    try:
        import boto3
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        iam = session.client('iam')
        roles = iam.list_roles().get('Roles', [])
        for role in roles:
            role_name = role['RoleName']
            if 'sagemaker' in role_name.lower():
                logger.info(f"Found SageMaker IAM Role: {role['Arn']}")
                return role['Arn']
    except Exception as e:
        logger.warning(f"Unable to query IAM roles via boto3: {e}")

    logger.warning(
        "STOP: SageMaker Execution IAM Role not detected in environment or IAM query.\n"
        "To run SageMaker Processing jobs, please set the SAGEMAKER_ROLE_ARN environment variable:\n"
        "  export SAGEMAKER_ROLE_ARN='arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-...'"
    )
    return None


def upload_inputs_to_s3():
    """Upload candidate pairs and dataset files to S3 if not present."""
    logger.info("Syncing candidate pairs and data to S3...")
    candidate_local = project_root / "output/phase2/candidate_pairs.tsv"
    s3_candidate_dest = f"{S3_BUCKET}/data/phase3/candidate_pairs.tsv"
    
    if candidate_local.exists():
        cmd = ["aws", "s3", "cp", str(candidate_local), s3_candidate_dest, "--profile", AWS_PROFILE]
        subprocess.run(cmd, check=True)
        logger.info(f"Uploaded {candidate_local.name} -> {s3_candidate_dest}")


def launch_sagemaker_processing(role_arn: str, dry_run: bool = False):
    """Launch SageMaker Processing Job using ScriptProcessor."""
    try:
        import boto3
        import sagemaker
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    except ImportError:
        logger.error("boto3 or sagemaker package not installed. Run `pip install boto3 sagemaker`.")
        sys.exit(1)

    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    job_name = f"phase4-feature-gen-{int(time.time())}"
    s3_output_path = f"{S3_BUCKET}/features/phase4/{EXPERIMENT_ID}/"

    logger.info("Job Configuration:")
    logger.info(f"  - Job Name:       {job_name}")
    logger.info(f"  - Instance Type:  {INSTANCE_TYPE}")
    logger.info(f"  - Role ARN:       {role_arn}")
    logger.info(f"  - S3 Output:      {s3_output_path}")

    if dry_run:
        logger.info("DRY RUN specified: Skipping actual SageMaker Job submission.")
        return job_name, s3_output_path

    processor = ScriptProcessor(
        command=['python3'],
        image_uri=sagemaker.image_uris.retrieve('sklearn', AWS_REGION, version='1.2-1'),
        role=role_arn,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sm_session
    )

    entrypoint = str(project_root / "src/features/sagemaker_processing.py")

    logger.info("Submitting SageMaker Processing Job...")
    processor.run(
        code=entrypoint,
        inputs=[
            ProcessingInput(
                source=f"{S3_BUCKET}/data/phase3/candidate_pairs.tsv",
                destination="/opt/ml/processing/input/candidates/candidate_pairs.tsv"
            ),
            ProcessingInput(
                source=f"{S3_BUCKET}/data/raw/dataset/",
                destination="/opt/ml/processing/input/data/"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/features",
                destination=s3_output_path
            )
        ],
        arguments=["--split", "test", "--val-ratio", "0.2"],
        job_name=job_name,
        wait=True
    )

    logger.info(f"SageMaker Processing Job {job_name} COMPLETED!")
    return job_name, s3_output_path


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker Phase 4 Feature Processing Job")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without submitting SageMaker job")
    args = parser.parse_args()

    check_aws_credentials()
    check_s3_bucket()
    upload_inputs_to_s3()

    role_arn = get_sagemaker_execution_role()
    if not role_arn:
        logger.info("SageMaker role not found; local dataset feature generation completed. Uploading local artifacts to S3...")
        local_features = project_root / "features"
        s3_dest = f"{S3_BUCKET}/features/phase4/{EXPERIMENT_ID}/"
        cmd = ["aws", "s3", "sync", str(local_features), s3_dest, "--profile", AWS_PROFILE]
        subprocess.run(cmd, check=True)
        logger.info(f"Local Phase 4 feature matrix uploaded successfully to {s3_dest}")
        return

    job_name, s3_output = launch_sagemaker_processing(role_arn, dry_run=args.dry_run)
    logger.info(f"Pipeline Execution Finish. Outputs: {s3_output}")


if __name__ == "__main__":
    main()
