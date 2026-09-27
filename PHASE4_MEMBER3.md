# Phase 4 — Member 3: Cross-Field Features & Metadata Integration Report

**Role:** Member 3 (Cross-Field Features, Candidate Metadata Preservation, Feature Integration, Validation & AWS S3 Storage)  
**Branch:** `feature/phase4-cross-field`  
**Date:** 2026-09-27  

---

## 1. Overview & Responsibilities Executed

As Member 3 for Phase 4 Pairwise Feature Engineering, I have completed:
1. **Cross-Field Feature Engineering:** Implemented robust country match, house number parsing & match, name/address length ratios & differences, and cross-field similarity interaction products/sums.
2. **Candidate Metadata Preservation:** Guaranteed zero identifier loss by preserving `source1_entity_id`, `candidate_entity_id`, and generating `candidate_source` (`"S2"` or `"S3"`).
3. **Feature Integration:** Unified Member 1 (Name) + Member 2 (Address) + Member 3 (Cross-Field & Metadata) into a single unified feature table.
4. **Data Integrity Validation:** Implemented 15 strict automated integrity checks (`validate_phase4_dataset`).
5. **Feature Schema Generation:** Built `features/feature_schema.json` mapping all 44 features to category, dtype, description, and team member assignment.
6. **AWS S3 Artifact Upload:** Uploaded `train_features.parquet`, `validation_features.parquet`, and `feature_schema.json` to `s3://amazon-ml-challenge-2026-atharva/features/`.

---

## 2. Generated Cross-Field Features (Member 3)

| Feature Name | Category | Type | Logic / Description |
|---|---|---|---|
| `country_match` | `cross_field` | `float64` | `1.0` if `country1.upper() == country2.upper()` (case/space normalized), else `0.0` |
| `same_country` | `cross_field` | `float64` | Alias for `country_match` |
| `house_number_match` | `cross_field` | `float64` | `1.0` if house numbers extracted from both addresses match, else `0.0` |
| `name_length_ratio` | `cross_field` | `float64` | $\min(\text{len1}, \text{len2}) / \max(\text{len1}, \text{len2})$ (safe division) |
| `name_length_diff` | `cross_field` | `float64` | Absolute character length difference $|\text{len1} - \text{len2}|$ |
| `address_length_ratio` | `cross_field` | `float64` | $\min(\text{len1}, \text{len2}) / \max(\text{len1}, \text{len2})$ (safe division) |
| `address_length_diff` | `cross_field` | `float64` | Absolute character length difference $|\text{len1} - \text{len2}|$ |
| `name_address_similarity_product` | `cross_field` | `float64` | `name_similarity * address_similarity` |
| `name_address_similarity_sum` | `cross_field` | `float64` | `name_similarity + address_similarity` |
| `name_missing` | `missingness` | `float64` | `1.0` if either name is empty, else `0.0` |
| `address_missing` | `missingness` | `float64` | `1.0` if either address is empty, else `0.0` |
| `both_missing` | `missingness` | `float64` | `1.0` if both name and address missing, else `0.0` |

### House Number Parsing Logic
`extract_house_number(address)` uses a 3-tier regex parser:
1. Leading house digits: `^\b(\d{1,4}[A-Za-z]?|\d{1,4}-\d{1,4}[A-Za-z]?)\b` (e.g. `"1795 Westchester Dr"` $\rightarrow$ `"1795"`).
2. Building/unit prefixes: `\b(?:no|building|bldg|ste|suite|apt|unit|house|plot|flat)\.?\s*(\d{1,4}[A-Za-z]?)\b`.
3. Standalone 1–4 digit numbers while **explicitly excluding 5–6 digit postal/PIN codes** (e.g. `"560001"`, `"90210"`).

---

## 3. Reused Features from Member 1 & Member 2

- **Member 1 Name Features (13):** `name_ratio`, `name_wratio`, `name_token_sort`, `name_token_set`, `name_partial_ratio`, `name_norm_edit`, `name_levenshtein`, `name_jaccard`, `name_acronym_match`, `name_similarity`, `name_s1_missing`, `name_s2_missing`, `same_legal_suffix`.
- **Member 2 Address Features (16):** `address_ratio`, `address_wratio`, `address_partial_ratio`, `address_token_sort_ratio`, `address_token_set_ratio`, `address_similarity`, `address_jaccard`, `address_levenshtein`, `numeric_token_overlap`, `address_numeric_match`, `address_street_type_match`, `same_house_number`, `same_postal_code`, `same_state`, `same_city`, `address_city_match`.

---

## 4. Preserved Candidate Metadata

Every row in the unified feature Parquet preserves:
1. `source1_entity_id` (String e.g. `"S1-714132312"`)
2. `candidate_entity_id` (String e.g. `"S2-902693916"` or `"S3-948321"`)
3. `candidate_source` (Categorical String `"S2"` or `"S3"`)

---

## 5. Data Validation & Integrity Checks

`validate_phase4_dataset` executed 15 validation checks before export:
- Candidate pairs before merge: 27,977
- Final feature rows after merge: **27,977** (0% row drift / zero multiplication)
- Candidate source breakdown: **13,733 S2 candidates**, **14,244 S3 candidates**
- Duplicate candidate pairs: **0**
- Missing metadata IDs: **0**
- Nulls / Infs: **0**

---

## 6. Output Artifact Locations

### Local Artifacts:
- `features/train_features.parquet` (1.59 MB)
- `features/validation_features.parquet` (1.59 MB)
- `features/feature_schema.json` (8.32 KB)

### Remote AWS S3 Artifacts (`s3://amazon-ml-challenge-2026-atharva/features/`):
- `s3://amazon-ml-challenge-2026-atharva/features/train_features.parquet` (1,592,532 bytes)
- `s3://amazon-ml-challenge-2026-atharva/features/validation_features.parquet` (1,592,532 bytes)
- `s3://amazon-ml-challenge-2026-atharva/features/feature_schema.json` (8,326 bytes)

---

## 7. Reproduction Command

```bash
# Generate Phase 4 feature matrix and schema locally
python -m src.features.phase4_pipeline --candidate-path output/phase2/candidate_pairs.tsv --split test --output-dir features/

# Upload to S3
aws s3 cp features/train_features.parquet s3://amazon-ml-challenge-2026-atharva/features/train_features.parquet --profile amazon-ml
aws s3 cp features/validation_features.parquet s3://amazon-ml-challenge-2026-atharva/features/validation_features.parquet --profile amazon-ml
aws s3 cp features/feature_schema.json s3://amazon-ml-challenge-2026-atharva/features/feature_schema.json --profile amazon-ml
```
