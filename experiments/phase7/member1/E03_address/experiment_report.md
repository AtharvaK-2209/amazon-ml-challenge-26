# Experiment E03

## Objective
Test whether additional address component features help resolve name collisions.

## Changes
Added 3 new address component features not in Phase 4:
- `zip5_match`: binary — 5-digit ZIP codes match.
- `street_num_match`: binary — first numeric token (street number) matches.
- `address_token_overlap`: Jaccard on cleaned address tokens (normalised, lowercase).

Phase 4 has `postal_match` (regex-based) and `house_number_match` (first-token), but `zip5_match` targets strict 5-digit code agreement and `address_token_overlap` provides broader coverage of shared address terms.

## Features
50 features used.

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
- `zip5_match`
- `street_num_match`
- `address_token_overlap`

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
| Precision | 1.000000 |
| Recall | 0.988439 |
| F0.5 | 0.997666 |
| F1 | 0.994186 |
| ROC-AUC | 0.999978 |
| PR-AUC | 0.999722 |
| False Positives | 0 |
| False Negatives | 4 |

## Comparison With E01

| Metric | E01 | E03 | Delta |
|--------|----:|--------:|------:|
| Precision | 0.997093 | 1.000000 | +0.002907 |
| Recall | 0.991329 | 0.988439 | -0.002890 |
| F0.5 | 0.995935 | 0.997666 | +0.001731 |
| F1 | 0.994203 | 0.994186 | -0.000017 |
| PR-AUC | 0.999796 | 0.999722 | -0.000074 |
| ROC-AUC | 0.999985 | 0.999978 | -0.000006 |
| FP | 1 | 0 | -1 |
| FN | 3 | 4 | +1 |

## Error Observations
0 FP, 4 FN at T=0.94.

## Runtime
0.29 seconds

## Conclusion
E03 PR-AUC change vs E01: -0.000074. Address component agreement features measured for additive value over Phase 4 address features.
