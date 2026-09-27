# Phase 1 EDA — Data Quality & Noise Specification
## Amazon ML Challenge 2026 · Business Entity Resolution

> **Generated:** 2026-09-27 07:27 UTC  
> **Input:** `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`  
> **Outputs consumed by:** Phase 2 (Normalisation), Phase 3 (Blocking)

> **Machine-readable spec:** [`eda_output/noise_spec.json`](noise_spec.json)

---

## 1. Dataset Scale

| Source | Rows | Null Name | Null Address | Null Country | Duplicate IDs |
|--------|------|-----------|-------------|--------------|---------------|
| **S1** | 1,732,544 | 0 | 0 | 0 | 0 |
| **S2** | 4,887,273 | 46 | 129,408 | 0 | 0 |
| **S3** | 5,082,316 | 59 | 136,098 | 0 | 0 |

> S1 (reference source) has **zero nulls**. S2 & S3 have ~2.65–2.68% null addresses — these records must still be matched on name alone.

## 2. Country Distribution

| Country | S1 | S2 | S3 | Notes |
|---------|----|----|-----|-------|
| India | 809,986 | 2,312,565 | 2,405,000 |  |
| US | 663,106 | 1,871,330 | 1,945,701 |  |
| France | 259,452 | 703,378 | 731,615 | ⚠️ **Zero-shot** — not in training data |

> Country proportions are consistent across all 3 sources (~47% India / 38% US / 15% France).  
> **Action:** Use `country` as a hard partition key in blocking (never match across countries).

## 3. Script Distribution (Business Names)

| Script | S1 | S2 | S3 |
|--------|----|----|-----|
| latin | 1,732,544 | 4,351,900 | 4,783,598 |
| devanagari | 0 | 303,467 | 170,091 |
| other | 0 | 231,749 | 128,330 |
| empty | 0 | 46 | 59 |

> **Critical finding:** S1 is 100% Latin. S2 has 303K Devanagari + 232K other-script names.  
> Direct string similarity will **fail** for these pairs → transliteration is mandatory.

## 4. Noise Pattern Rates

| Pattern | S1 (%) | S2 (%) | S3 (%) | Severity | Normalisation Rule |
|---------|--------|--------|--------|----------|--------------------|
| `name_has_abbrev` | 29.4 | 28.05 | 28.66 | 🔴 High | N02 — Expand legal suffixes |
| `name_has_ampersand` | 5.02 | 4.27 | 4.19 | 🟡 Medium | N03 — Replace & → and |
| `name_all_caps` | 0.0 | 17.25 | 2.72 | 🔴 High | N01 — Lowercase all |
| `name_has_digit` | 1.17 | 3.89 | 3.98 | 🟢 Low | N06 — Keep digits (discriminative) |
| `addr_missing_pin` | 95.63 | 94.93 | 94.96 | 🔴 Critical | B01–B05 — Never use PIN as blocking key |
| `addr_has_landmark` | 5.33 | 4.64 | 3.49 | 🟡 Medium | N07 — Keep; use full address TF-IDF |
| `addr_has_abbrev` | 12.65 | 26.68 | 24.55 | 🔴 High | N07 — Expand address abbreviations |

## 5. Field Length Statistics

| Source | Field | Mean | Median | Std | Min | Max | p95 |
|--------|-------|------|--------|-----|-----|-----|-----|
| S1 | name_len | 23.8 | 24.0 | 7.7 | 3 | 92 | 36.0 |
| S1 | addr_len | 57.2 | 50.0 | 25.0 | 11 | 268 | 105.0 |
| S1 | name_tokens | 3.5 | 4.0 | 0.9 | 1 | 14 | 5.0 |
| S1 | addr_tokens | 8.6 | 8.0 | 3.6 | 2 | 43 | 16.0 |
| S2 | name_len | 25.7 | 25.0 | 9.1 | 0 | 102 | 42.0 |
| S2 | addr_len | 50.4 | 43.0 | 25.3 | 0 | 269 | 99.0 |
| S2 | name_tokens | 3.6 | 4.0 | 1.2 | 0 | 15 | 5.0 |
| S2 | addr_tokens | 7.8 | 7.0 | 3.7 | 0 | 43 | 15.0 |
| S3 | name_len | 25.7 | 25.0 | 9.6 | 0 | 103 | 42.0 |
| S3 | addr_len | 48.7 | 43.0 | 22.6 | 0 | 267 | 94.0 |
| S3 | name_tokens | 3.6 | 4.0 | 1.3 | 0 | 16 | 6.0 |
| S3 | addr_tokens | 7.5 | 7.0 | 3.5 | 0 | 43 | 15.0 |

> p95 name length is 36–42 chars — cap TF-IDF token length at 50 chars.  
> p95 address length is 94–105 chars — no truncation needed for BM25/TF-IDF.

## 6. Normalisation Rules (Priority-Ordered)

These rules must be applied **in priority order** to produce a clean, canonical form for each field.

| ID | Priority | Field | Trigger | Action | Key Params |
|----|----------|-------|---------|--------|------------|
| **N07** | 1 | `business_address` | addr_has_abbrev: S1=12.65%, S2=26.68%, S3=24.55%… | `expand_address_abbreviations` | `expansions={'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'hwy': 'highway', 'fwy': 'freeway', 'apt': 'apartment', 'ste': 'suite', 'fl': 'floor', 'nagar': 'nagar', 'marg': 'marg', 'vihar': 'vihar', 'enclave': 'enclave', 'colony': 'colony', 'extn': 'extension', 'tq': 'taluka', 'dist': 'district', 'opp': 'opposite', 'rue': 'rue', 'bd': 'boulevard', 'av': 'avenue'}` |
| **N08** | 2 | `business_address` | null_business_address: S2=129408 (2.65%), S3=136098 (2.68%)… | `fill_null_address` | `fill_value=` |
| **N09** | 3 | `business_address` | addr_missing_pin: ~95% across all sources… | `do_not_use_pin_as_blocking_key` | — |
| **N01** | 1 | `business_name` | name_all_caps: S2=17.25%, S3=2.72%… | `lowercase` | — |
| **N02** | 2 | `business_name` | name_has_abbrev: ~29% across all sources… | `expand_legal_suffixes` | `expansions={'pvt': 'private', 'ltd': 'limited', 'llc': 'limited liability company', 'corp': 'corporation', 'inc': 'incorporated', 'co': 'company', 'llp': 'limited liability partnership', 'plc': 'public limited company', 'huf': 'hindu undivided family', 'sarl': 'societe a responsabilite limitee', 'sas': 'societe par actions simplifiee', 'sa': 'societe anonyme'}` |
| **N03** | 3 | `business_name` | name_has_ampersand: ~4-5% across all sources… | `replace_token` | `find=&`, `replace=and` |
| **N04** | 4 | `business_name` | script_dist: S2=10.9% non-Latin, S3=5.8% non-Latin; S1=100% … | `transliterate_to_latin` | `tool=indic_transliteration or anyascii`, `source_scripts=['devanagari', 'arabic', 'other']` |
| **N05** | 5 | `business_name` | French accented characters in France set (15% of all records… | `strip_accents` | `method=unicodedata.normalize('NFKD') + ascii encode`, `examples={'é': 'e', 'è': 'e', 'ê': 'e', 'à': 'a', 'ç': 'c', 'ô': 'o', 'î': 'i'}` |
| **N06** | 6 | `business_name` | residual punctuation after above steps… | `strip_punctuation_and_extra_whitespace` | `keep_hyphens=True`, `keep_digits=True` |

### Normalisation Pipeline Order (per record)

```
For business_name:
  1. [N04] Transliterate non-Latin → Latin  (S2/S3 India only)
  2. [N01] Lowercase
  3. [N05] Strip accents / diacritics        (France)
  4. [N02] Expand legal suffixes             (all)
  5. [N03] Replace & → and                  (all)
  6. [N06] Strip punctuation, collapse whitespace

For business_address:
  1. [N08] Fill nulls with ''
  1. [N01] Lowercase (reuse)
  2. [N05] Strip accents (reuse, France)
  3. [N07] Expand address abbreviations
  4. [N06] Strip punctuation, collapse whitespace
```

## 7. Blocking Strategy Recommendations

| ID | Strategy | Key Params | Derived From | Notes |
|----|----------|------------|--------------|-------|
| **B01** | name_trigram_tfidf | analyzer=char_wb, ngram_range=[3, 3] | N01, N02, N03, N04, N05, N06 | High-recall anchor since ~29% of names share legal suffix tokens — TF-IDF natura |
| **B02** | sorted_neighbourhood_name_prefix | window_size=10 | N01, N02, N06 | Fast O(N log N) complement to TF-IDF. Catches transpositions that survive normal |
| **B03** | country_partition | partition_key=country | country_dist | 100% safe assumption per problem statement. Reduces search space by ~3x. |
| **B04** | address_token_overlap | min_token_len=4, idf_threshold=3.0 | N07, N08 | Compensates for name-only blocking misses when addresses are clean. Disable for  |
| **B05** | name_first_token_exact | min_token_len=3 | N01, N02 | Very fast O(N) grouping. High precision on name-stable businesses. High false-po |
| **B06** | devanagari_transliteration_fallback | phonetic_algo=metaphone, applies_to_scripts=['devanagari'] | N04 | Handles transliteration noise: 'Shri' vs 'Sri', 'Kumar' vs 'Kumar'. Essential fo |

> **Recommended blocking stack:** B03 (country partition) → B01 (name TF-IDF) ∪ B04 (address token) ∪ B05 (first-token exact)  
> Target: **recall ceiling > 90%** while reducing candidate pairs from 8.47T → <50M

## 8. Per-Country Handling Flags

### US
- **Script:** latin
- **Address format:** Street number + street name + city + state abbreviation + optional ZIP. ZIP present in <5% of records (addr_missing_pin ~95%).

### India
- **Script:** latin_and_devanagari
- **Address format:** Highly variable: Plot/Flat/Survey numbers, locality names, taluka/district/state. Landmark-based references common (~5%). PIN codes rarely present.
- **Special handling:** transliterate_devanagari, expand_tq_dist_abbreviations

### France ⚠️ ZERO-SHOT
- **Script:** latin_with_accents
- **Address format:** French format: street number + rue/boulevard/avenue + city + postal code (5-digit). Postal code present more often than US/India.
- **Special handling:** strip_french_accents, handle_sarl_sas_sa_suffixes
- ⚠️ **Warning:** France does NOT appear in the training data. Do not hard-code country logic. All normalisation must be generic enough to handle unseen countries.

## 9. Global Pipeline Flags

| Flag | Value | Consequence |
|------|-------|-------------|
| `pin_zip_usable_as_blocking_key` | ❌ False | 95% coverage gap — skip PIN-based blocking |
| `safe_to_drop_null_address_records` | ❌ False | ~265K records have no address; must match on name |
| `must_handle_zero_shot_countries` | ✅ True | France has no training examples — pipeline must generalise |
| `evaluation_metric` | F_0.5 macro | Precision-heavy — false merges cost 2× more than missed links |
| `total_test_s1_records_to_submit` | 1,732,544 | Every S1 record must appear in output — including singletons |

## 10. EDA Output Files

| File | Description |
|------|-------------|
| `eda_output/noise_spec.json` | Machine-readable spec — import in normalisation + blocking scripts |
| `eda_output/data_quality_report.md` | This document |
| `eda_output/01_overview.csv` | Shape + null counts per source |
| `eda_output/06_noise_patterns.csv` | Noise flag rates (%) per source |
| `eda_output/11_length_stats.csv` | Length statistics (mean/median/std/p95) |
| `eda_output/12_samples.csv` | 5 sample records per country per source |
| `eda_output/plots/` | All 8 visualisation PNGs |