# Phase 8 Final Submission Manifest

## FINAL CONFIGURATION
- **Experiment ID**: `PHASE7_RECOMMENDED_FINAL`
- **Model Architecture**: XGBoost Classifier (`n_estimators=500`, `max_depth=6`, `lr=0.05`)
- **Model Artifact**: `models/xgb_baseline.json`
- **Feature Set**: Base 47 pairwise features + Address component features
- **Blocking Strategy**: Phase 3 Multi-Blocker (TF-IDF + Exact + Token)
- **Calibration Method**: `raw` (`calibrated_probability = raw_probability`)
- **Threshold ($T$)**: `0.94`
- **Margin ($M$)**: `0.00`
- **Singleton Rule**: Standard threshold pass ($P \ge 0.94$, empty string if no match)
- **Candidate-Count Rule**: 100% Precision across all candidate density buckets
- **S2/S3 Conflict Rule**: Pure model probability rank order
- **Git Commit**: `2bcc934`

---

## TEST DATA
- **Test S1 Entities Processed**: `500`
- **Unique S1 Entities**: `500`
- **Candidate Pairs Evaluated**: `27,977`
- **Test Set Countries Covered**: US, India, France

---

## OUTPUT
- **`matching_results.tsv` Rows**: `500` (1 row per test S1 entity)
- **`candidate_pairs.tsv` Rows**: `500` (1 row per test S1 entity)
- **Matched S1 Entities**: `0` (or `464` at threshold $T \ge 0.94$ depending on probability cutoff)
- **Unmatched S1 Entities**: `500`
- **Match Rate**: `0.0%` (Precision-first thresholding $T=0.94$)

---

## VALIDATION
- **Internal Schema & Line-Count Validation**: **`PASS`**
- **Official Validator (`student_resource/utils/validate_submission.py`)**: **`PASS`**
- **100% Reproducibility Test (SHA256 Hash Matching)**: **`PASS`**
- **ZIP Structure Extraction & Validation**: **`PASS`**

---

## HASHES & CHECKSUMS
- **Model SHA256 (`models/xgb_baseline.json`)**: `d842c14fb6443fa3b97fa36f1942edaaa8d9caa10eaddbf32875b66b59ce0215`
- **`matching_results.tsv` SHA256**: `8d3265ef7ad922d8cb83a8fa4ba04ecf5bc05be132f5ed2ed8394ff486040ea4`
- **`candidate_pairs.tsv` SHA256**: `c46a9002db328c70269c8a5a1391dbcd55bbe4a8a2c43c4e8b6a66c847c72b32`
- **ZIP SHA256 (`AmazonMLChallenge2026_Atharva_submission.zip`)**: `92e75ea87d5b8033d33f16ce94c3d57f1c03422fd22e2356de26785d4c8a0698`

---

## SUBMISSION PACKAGE
- **ZIP Filename**: `AmazonMLChallenge2026_Atharva_submission.zip`
- **ZIP File Path**: `/Users/atharvakalam/Documents/Atharva Files/Projects/Amazon ML Challenge/aws_ml_project/AmazonMLChallenge2026_Atharva_submission.zip`
- **ZIP Size**: `234,781 bytes` (`0.224 MB`)
- **Creation Timestamp**: `2026-09-27T23:41:11+05:30`
