# Phase 4 — Pairwise Feature Integration & AWS SageMaker Pipeline

## 1. Overview & Architecture

Phase 4 integrates pre-candidate pair metadata, Member 1 business name features, Member 2 address features, and Member 3 cross-field features into a unified, leak-free, validated machine learning feature matrix for Phase 5 entity matching model training.

```
                  Candidate pairs (Phase 3 TSV)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
   Name Features (Member 1)                   Address Features (Member 2)
   [src/features/name_features.py]            [src/features/address_features.py]
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                   Cross-Field Features (Member 3)
                   [src/features/pair_features.py]
                                │
                                ▼
                     Unified Feature Matrix (47 cols)
                     [src/features/build_features.py]
                                │
                                ▼
                   Entity-Grouped Train/Val Split (80/20)
                   GroupShuffleSplit(group=s1_entity_id)
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
       Train Parquet (22,387)      Validation Parquet (5,590)
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                     S3 & SageMaker Processing
       s3://amazon-ml-challenge-2026-atharva/features/phase4/EXP-001/
```

---

## 2. Feature Source Attribution

| Module | Feature Scope | Description |
|---|---|---|
| **Member 1** | `name_features.py` | Levenshtein distance, RapidFuzz ratios (ratio, partial ratio, token sort ratio, token set ratio), Word/Char 3-Gram Jaccard and Cosine similarities on normalized names. |
| **Member 2** | `address_features.py` | Levenshtein distance, RapidFuzz ratios, Word/Char 3-Gram similarities on normalized addresses, exact name/address match flags, locality, city, state, street, and postal match. |
| **Member 3** | `pair_features.py` | Cross-field features: `country_match` (case/whitespace invariant), `house_number_match` (3-tier regex house/building parser ignoring ZIP codes), `name_length_ratio` / `name_length_diff`, `address_length_ratio` / `address_length_diff`, and missingness indicators (`name_missing`, `address_missing`, `both_missing`). |

---

## 3. Metadata Preservation & Column Identifiers

Every row in the unified dataset retains standard identifiers:
- `s1_entity_id`: Primary reference entity ID.
- `candidate_entity_id`: Candidate entity ID (from Source 2 or Source 3).
- `candidate_source`: String identifier (`'S2'` or `'S3'`).

---

## 4. Train / Validation Splitting Strategy

- **Strategy**: Entity-Grouped Shuffle Split (`sklearn.model_selection.GroupShuffleSplit`).
- **Grouping Column**: `s1_entity_id`.
- **Ratio**: 80% Train (`22,387` candidate pairs), 20% Validation (`5,590` candidate pairs).
- **RATIONALE**: Splitting candidate pairs at random causes severe target/data leakage because multiple candidate pairs share the exact same `s1_entity_id`. Grouping by `s1_entity_id` guarantees that all candidate pairs for any single reference entity reside strictly within either the train set or validation set.

---

## 5. Validation Checks (`src/features/validate_features.py`)

The pipeline runs 7 automated validation checks before output export:
1. **Shape Check**: Asserts exact row retention (`N = 27,977`).
2. **Missing Values**: Asserts 0 nulls across all generated feature columns.
3. **Invalid Numeric Values**: Asserts 0 `NaN`, `+inf`, or `-inf`.
4. **Duplicate Candidate Pairs**: Asserts uniqueness of (`s1_entity_id`, `candidate_entity_id`).
5. **Feature Range Checks**: Asserts values conform to valid bounds (e.g., similarity ratios in `[0, 100]` or `[0, 1]`).
6. **Row Preservation**: Ensures post-merge row count equals input row count.
7. **Target Leakage**: Verifies zero target/ground-truth columns (`target`, `match_label`, `ground_truth`) are present in ML features.

---

## 6. Execution Commands

### Local Execution
```bash
./venv/bin/python -m src.features.build_features \
    --candidate-path output/phase2/candidate_pairs.tsv \
    --split test \
    --output-dir features/
```

### SageMaker Execution
```bash
./venv/bin/python scripts/run_sagemaker_phase4.py
```

---

## 7. S3 Output & Artifact Paths

- **Bucket**: `s3://amazon-ml-challenge-2026-atharva/`
- **Prefix**: `features/phase4/EXP-001/`
- **Artifacts**:
  - `train_features.parquet` (22,387 rows, 47 columns)
  - `validation_features.parquet` (5,590 rows, 47 columns)
  - `feature_schema.json` (Machine-readable schema & feature metadata)
  - `feature_statistics.json` (Comprehensive min/max/mean/std summary)

---

## 8. AWS SageMaker Processing & Cost Control

- **Instance Type**: `ml.m5.2xlarge` (8 vCPU, 32 GiB RAM).
- **Cost**: ~$0.38 / hour. Estimated execution time: ~1–2 minutes.
- **Cost Policy Compliance**:
  - Uses `sagemaker.processing.ScriptProcessor` which terminates automatically upon completion.
  - ZERO persistent notebook instances, EC2 servers, GPU instances, or real-time model endpoints created.

---

## 9. Consumption Guide for Phase 5 (Model Training)

Phase 5 XGBoost/LightGBM model training scripts can load the datasets directly:
```python
import pandas as pd

train_df = pd.read_parquet("features/train_features.parquet") # or S3 path
val_df = pd.read_parquet("features/validation_features.parquet")

# Exclude metadata columns to extract X
meta_cols = ["s1_entity_id", "candidate_entity_id", "candidate_source"]
feature_cols = [c for c in train_df.columns if c not in meta_cols]

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
```
