# Experiment E04

## Objective
Test whether cross-field interaction features improve discrimination of hard cases.

## Changes
Added 5 cross-field interaction features:
- `name_x_addr_sim`: name_fuzz_ratio × address_fuzz_ratio — rewards joint similarity.
- `name_x_postal`: name_fuzz_ratio × same_postal_code — name match + same zip.
- `name_x_housenumber`: name_fuzz_ratio × same_house_number — name + street num.
- `name_minus_addr`: name_fuzz_ratio - address_fuzz_ratio — name/address disagreement.
- `token_set_x_postal`: name_token_set_ratio × same_postal_code — token match + zip.

Each interaction has a direct entity-resolution interpretation. No arbitrary exhaustive combinations were added.

## Features
52 features used.

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
- `name_x_addr_sim`
- `name_x_postal`
- `name_x_housenumber`
- `name_minus_addr`
- `token_set_x_postal`

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
| Precision | 0.997085 |
| Recall | 0.988439 |
| F0.5 | 0.995343 |
| F1 | 0.992743 |
| ROC-AUC | 0.999987 |
| PR-AUC | 0.999826 |
| False Positives | 1 |
| False Negatives | 4 |

## Comparison With E01

| Metric | E01 | E04 | Delta |
|--------|----:|--------:|------:|
| Precision | 0.997093 | 0.997085 | -0.000008 |
| Recall | 0.991329 | 0.988439 | -0.002890 |
| F0.5 | 0.995935 | 0.995343 | -0.000592 |
| F1 | 0.994203 | 0.992743 | -0.001460 |
| PR-AUC | 0.999796 | 0.999826 | +0.000030 |
| ROC-AUC | 0.999985 | 0.999987 | +0.000002 |
| FP | 1 | 1 | +0 |
| FN | 3 | 4 | +1 |

## Error Observations
1 FP, 4 FN at T=0.94.

## Runtime
0.29 seconds

## Conclusion
E04 PR-AUC change vs E01: +0.000030. Cross-field features measured for discriminative value on the validation split.
