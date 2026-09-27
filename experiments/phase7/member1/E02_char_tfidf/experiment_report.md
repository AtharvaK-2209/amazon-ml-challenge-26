# Experiment E02

## Objective
Test whether character (3-6) n-gram TF-IDF similarity improves matching for typos and transliteration.

## Changes
Added 2 new features:
- `name_char_tfidf_36`: character n-gram (3,6) TF-IDF cosine similarity on business names.
- `address_char_tfidf_36`: character n-gram (3,6) TF-IDF cosine similarity on addresses.

Rationale: Phase 1 EDA found ~12% of entity names contain typos or transliteration variants. Character (3,6) n-grams overlap even when 1-2 characters differ.

## Features
49 features used.

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
- `name_char_tfidf_36`
- `address_char_tfidf_36`

</details>

## Model
XGBoost, same params as E01

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
| ROC-AUC | 0.999983 |
| PR-AUC | 0.999780 |
| False Positives | 1 |
| False Negatives | 3 |

## Comparison With E01

| Metric | E01 | E02 | Delta |
|--------|----:|--------:|------:|
| Precision | 0.997093 | 0.997093 | +0.000000 |
| Recall | 0.991329 | 0.991329 | +0.000000 |
| F0.5 | 0.995935 | 0.995935 | +0.000000 |
| F1 | 0.994203 | 0.994203 | +0.000000 |
| PR-AUC | 0.999796 | 0.999780 | -0.000016 |
| ROC-AUC | 0.999985 | 0.999983 | -0.000001 |
| FP | 1 | 1 | +0 |
| FN | 3 | 3 | +0 |

## Error Observations
1 FP, 3 FN at T=0.94.

## Runtime
0.29 seconds

## Conclusion
E02 PR-AUC change vs E01: -0.000016. Character (3,6) n-gram features contribute to or do not degrade the baseline based on measured PR-AUC and F0.5.
