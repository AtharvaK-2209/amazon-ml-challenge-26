# Phase 3 AWS & Local Architecture

**Project:** Amazon ML Challenge 2026 — Entity Resolution  
**Phase:** Phase 3 (Feature Engineering & Pipeline Infrastructure)  
**Author:** ML Infrastructure + AWS Integration Engineer  

---

## 1. High-Level Architecture Overview

Phase 3 is designed around a **Single-Codebase, Dual-Execution Environment** architecture. The exact same Python feature generation modules run locally during rapid development and inside AWS SageMaker Processing containers for large-scale production runs, without code duplication.

```
+-----------------------------------------------------------------------------------+
|                               LOCAL DEVELOPMENT                                   |
|                                                                                   |
|  output/phase2/candidate_pairs.tsv                                               |
|               |                                                                   |
|               v                                                                   |
|  python -m src.phase3.pipeline --config configs/phase3.yaml                       |
|               |                                                                   |
|               v                                                                   |
|  output/phase3/pair_features.parquet + feature_metadata.json                     |
|               |                                                                   |
|               v  (Manual / Scripted Sync)                                        |
|  aws s3 cp output/phase3/pair_features.parquet s3://<bucket>/features/phase3/EXP-001/ |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                            AWS SAGEMAKER EXECUTION                                |
|                                                                                   |
|  s3://<bucket>/candidates/phase2/candidate_pairs.tsv                              |
|               |                                                                   |
|               v  (SageMaker Processing Job Input Mount)                           |
|  /opt/ml/processing/input/candidate_pairs.tsv                                   |
|               |                                                                   |
|               v  (Invokes sagemaker_entrypoint.py -> Phase3Pipeline)              |
|  SageMaker Processing Container (ml.m5.2xlarge / ml.c5.4xlarge)                   |
|               |                                                                   |
|               v  (SageMaker Processing Job Output Sync)                           |
|  s3://<bucket>/features/phase3/EXP-001/pair_features.parquet                       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Shared Python Component Responsibilities

| Component | File | Responsibilities |
|---|---|---|
| **Configuration** | `configs/phase3.yaml` / `src/phase3/config.py` | Centralized hyperparameter, validation, and AWS path definitions |
| **I/O & S3 Manager** | `src/phase3/io.py` | TSV candidate parsing, Parquet serialization, read-only S3 CLI helpers |
| **Feature Integration** | `src/features/pair_features.py` | Integrates `name_features.py` (Member 1) and `address_features.py` (Member 2) |
| **Validation Engine** | `src/phase3/pipeline.py` (`validate_feature_matrix`) | Enforces strict row count preservation, null/inf checks, duplicate checks |
| **Experiment Tracker** | `src/phase3/experiment.py` | Appends commit hash, config, and feature metrics to `experiments/phase3/experiments.csv` |
| **SageMaker Entrypoint**| `src/phase3/sagemaker_entrypoint.py` | Mounts SageMaker container volumes and executes the core pipeline |

---

## 3. AWS Cost Safety Policy

To prevent unexpected charges against the competition AWS credit budget:

1. **No Standing Compute Resources:**
   - **No persistent EC2 instances.**
   - **No SageMaker Endpoints or Notebook Instances.**
   - **No RDS databases or Elastic Load Balancers.**

2. **Storage Only (S3):**
   - Bucket: `s3://amazon-ml-challenge-2026-atharva/`
   - Region: `ap-south-1`
   - AWS CLI Profile: `amazon-ml`

3. **Short-Lived SageMaker Execution:**
   - When SageMaker Processing is required for full-scale feature extraction, jobs use ephemeral, auto-terminating CPU instances (e.g. `ml.m5.2xlarge`).
   - Costs are incurred **only during active execution** and terminate automatically upon completion.

---

## 4. Reproducibility Workflow

To reproduce any Phase 3 feature engineering run:

```bash
# 1. Run local feature generation
python -m src.phase3.pipeline --config configs/phase3.yaml

# 2. Verify output validation logs
# Look for: "Validation PASSED: N rows × M numeric features"

# 3. Optional: Sync generated feature parquet to S3
aws s3 cp output/phase3/pair_features.parquet \
  s3://amazon-ml-challenge-2026-atharva/features/phase3/EXP-001/pair_features.parquet \
  --profile amazon-ml
```
