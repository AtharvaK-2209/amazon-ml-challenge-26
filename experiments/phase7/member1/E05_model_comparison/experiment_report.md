# Experiment E05

## Objective
Compare LightGBM vs XGBoost (E01) on the identical feature set and validation data.

## Changes
Model changed from XGBoost to LightGBM. Feature set identical to E01 (47 base Phase 6 features). Parameters matched as closely as possible (same n_estimators, lr, depth, subsample).

## Features
47 features used.

<details>
<summary>Full feature list</summary>

- `name_missing`
- `address_missing`
- `both_missing`
- `name_levenshtein`
- `name_fuzz_ratio`
- `name_ratio`
- `name_wratio`
- `name_token_sort_ratio`
- `name_token_set_ratio`
- `name_jaccard`
- `name_length_diff`
- `name_length_ratio`
- `name_token_count_diff`
- `name_tfidf_cosine`
- `name_char_cosine`
- `address_fuzz_ratio`
- `address_wratio`
- `address_token_sort_ratio`
- `address_token_set_ratio`
- `address_similarity`
- `address_ratio`
- `address_partial_ratio`
- `address_jaccard`
- `address_levenshtein`
- `address_tfidf_cosine`
- `address_char_cosine`
- `address_numeric_overlap`
- `address_numeric_match`
- `address_street_type_match`
- `house_number_match`
- `postal_match`
- `city_match`
- `state_match`
- `same_house_number`
- `same_postal_code`
- `same_city`
- `same_state`
- `address_city_match`
- `country_match`
- `same_country`
- `name_len_ratio`
- `name_length_difference`
- `address_length_ratio`
- `address_length_diff`
- `address_length_difference`
- `name_address_similarity_product`
- `name_address_similarity_sum`

</details>

## Model
LightGBM, params: n_estimators=200, max_depth=6, lr=0.05

## Decision Configuration
| Parameter | Value |
|-----------|-------|
| Threshold (T) | `0.94` |
| Margin (M) | `0.0` |
| Calibration | `raw` |

> Threshold and margin were **not tuned** for this experiment. Values are frozen from Phase 5 Member 3.

## Results

| Metric | Result |
|--------|-------:|
| Precision | 0.997093 |
| Recall | 0.991329 |
| F0.5 | 0.995935 |
| F1 | 0.994203 |
| ROC-AUC | 0.999969 |
| PR-AUC | 0.999596 |
| False Positives | 1 |
| False Negatives | 3 |

## Comparison With E01

| Metric | E01 | E05 | Delta |
|--------|----:|--------:|------:|
| Precision | 0.997093 | 0.997093 | +0.000000 |
| Recall | 0.991329 | 0.991329 | +0.000000 |
| F0.5 | 0.995935 | 0.995935 | +0.000000 |
| F1 | 0.994203 | 0.994203 | +0.000000 |
| PR-AUC | 0.999796 | 0.999596 | -0.000200 |
| ROC-AUC | 0.999985 | 0.999969 | -0.000016 |
| FP | 1 | 1 | +0 |
| FN | 3 | 3 | +0 |

## Error Observations
1 FP, 3 FN at T=0.94.

## Runtime
0.36 seconds

## Conclusion
E05 (LightGBM) PR-AUC change vs E01 (XGBoost): -0.000200. Comparison reflects model-family difference on identical information.
