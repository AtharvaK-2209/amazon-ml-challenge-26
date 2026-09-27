# Phase 1 / Phase 2 Data Contract

**Project:** Amazon ML Challenge 2026 — Entity Resolution  
**Last Updated:** 2026-09-27  
**Status:** Verified against committed code at `d28e5e4`  
**Author:** Infrastructure & Integration Engineer  

> All field names, file paths and formats in this document are derived **directly from the
> source code** (`src/config.py`, `src/pipeline.py`, `src/blocking/*.py`).  
> Nothing is inferred or invented.

---

## 0. Path Constants (from `src/config.py`)

| Constant | Resolved Path |
|---|---|
| `ROOT` | project root (parent of `src/`) |
| `DATA_DIR` | `ROOT/student_resource/dataset` ⚠️ **see Risk §7.1** |
| `TRAIN_DIR` | `DATA_DIR/train` |
| `TEST_DIR` | `DATA_DIR/test` |
| `EDA_DIR` | `ROOT/eda_output` |
| `OUTPUT_DIR` | `ROOT/output` |
| `FEATURES_DIR` | `ROOT/features` |
| `MODELS_DIR` | `ROOT/src/models/saved` |
| `OUTPUT_CANDIDATES` | `ROOT/output/candidate_pairs.tsv` |
| `OUTPUT_MATCHING` | `ROOT/output/matching_results.tsv` |

> **Active config file:** `src/config.py`.  
> `configs/config.yaml` is the Phase 0 scaffolding file and is **not loaded** by any Phase 1/2 code.

---

## 1. Phase 1 Inputs

| # | File | Format | Columns | Purpose |
|---|---|---|---|---|
| 1 | `data/train/train_source1.tsv` | TSV, UTF-8 | `entity_id`, `business_name`, `business_address`, `country` | S1 master reference entities (train) |
| 2 | `data/train/train_source2.tsv` | TSV, UTF-8 | `entity_id`, `business_name`, `business_address`, `country` | S2 secondary entities (train) |
| 3 | `data/train/train_source3.tsv` | TSV, UTF-8 | `entity_id`, `business_name`, `business_address`, `country` | S3 tertiary entities (train) |
| 4 | `data/train/train_ground_truth.tsv` | TSV, UTF-8 | `source1_entity_id`, `matched_entity_ids` | Ground-truth matches for S1 (comma-separated list in `matched_entity_ids`) |
| 5 | `data/test/test_source1.tsv` | TSV, UTF-8 | same as train | S1 entities (test) |
| 6 | `data/test/test_source2.tsv` | TSV, UTF-8 | same as train | S2 entities (test) |
| 7 | `data/test/test_source3.tsv` | TSV, UTF-8 | same as train | S3 entities (test) |

### Record Counts (measured)

| File | Row Count |
|---|---|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

### Column Semantics

| Column | Type | Notes |
|---|---|---|
| `entity_id` | string | Unique per source. Prefix encodes source: `S1-`, `S2-`, `S3-`. Never null. 100% unique within each file. |
| `business_name` | string | Raw business name. May contain non-Latin scripts (Devanagari, Tamil, Kannada). Never null in S1; 46/59 nulls in S2/S3. |
| `business_address` | string | Raw address. Null in ~3.35% of S2 and S3 records. Never null in S1. |
| `country` | string | `"US"` or `"India"` in train. **`"France"` appears in test (~14.97%)** — open-set. Never null. |

---

## 2. Phase 1 Outputs

All outputs are written to `output/phase1/` and `eda_output/`. Raw data is never modified.

| # | File | Format | Key Contents | Purpose |
|---|---|---|---|---|
| 1 | `output/phase1/EDA_REPORT.md` | Markdown | Full statistical analysis | Human-readable EDA report |
| 2 | `output/phase1/data_statistics.json` | JSON | Row counts, missing value rates, length stats | Machine-readable dataset summary |
| 3 | `output/phase1/noise_patterns.csv` | CSV | `category, pattern, source, frequency, percentage, example_1, example_2, notes` | Measured noise pattern catalogue |
| 4 | `output/phase1/plots/*.png` | PNG (8 files) | Dataset sizes, country dist., length dists., missing values heatmap, match dist., duplicate freqs. | EDA visualisations |
| 5 | `eda_output/noise_spec.json` | JSON | Noise rates by pattern × source, length stats, country/script distribution | **Machine-readable spec consumed by Phase 2 normaliser** |
| 6 | `eda_output/data_quality_report.md` | Markdown | Data quality findings | Supplementary quality report |
| 7 | `eda_output/eda_test_report.md` | Markdown | EDA run on test split | Test-split analysis |

### `noise_spec.json` Schema (consumed by `src/config.py`)

```json
{
  "meta": { "generated_at": "...", "phase": "1", "next_phase": "2" },
  "dataset_overview": { "S1": { "rows": int, "null_business_name": int, ... }, ... },
  "country_distribution": { "S1": { "US": int, "India": int, "France": int }, ... },
  "script_distribution": { "S1": { "latin": int, "devanagari": int, ... }, ... },
  "noise_pattern_rates_pct": {
    "name_has_abbrev": { "S1": float, "S2": float, "S3": float },
    "name_has_ampersand": { ... },
    "name_all_caps": { ... },
    "name_has_digit": { ... },
    "addr_missing_pin": { ... },
    "addr_has_landmark": { ... },
    "addr_has_abbrev": { ... }
  },
  "length_stats": { "S1_name_len": { "mean": float, "median": float, ... }, ... }
}
```

---

## 3. Phase 2 Inputs

Phase 2 reads the same raw TSV files as Phase 1 (read-only) plus the noise spec.

| # | File | Format | Purpose |
|---|---|---|---|
| 1–7 | Same raw TSV files as Phase 1 inputs | TSV | Entity records |
| 8 | `eda_output/noise_spec.json` | JSON | Drives normalisation rule selection in `Normalizer` |

### Normalisation applied (from `src/preprocessing/normalize.py`)

The `Normalizer` class adds these columns to each source DataFrame in-place:

| New Column | Type | Description |
|---|---|---|
| `original_name` | str | Preserved raw `business_name` — never modified |
| `normalized_name` | str | Full pipeline: transliterate → lowercase → strip accents → expand legal suffixes → `&`→`and` → strip punct → collapse whitespace |
| `alphanumeric_name` | str | `normalized_name` with all non-`[a-z0-9 ]` removed |
| `tokenized_name` | list[str] | `normalized_name.split()` |
| `name_without_legal_suffix` | str | `normalized_name` with trailing legal suffix tokens stripped |
| `original_address` | str | Preserved raw `business_address` — never modified |
| `normalized_address` | str | Pipeline: lowercase → strip accents → expand address abbreviations → strip punct → collapse whitespace |
| `address_tokens` | list[str] | `normalized_address.split()` |
| `numbers` | list[str] | All `\b\d+\b` tokens extracted from raw `business_address` |

**Legal suffix expansion map** (from `src/config.py` `NORM`):
`pvt→private`, `ltd→limited`, `llc→limited liability company`, `corp→corporation`, `inc→incorporated`, `co→company`, `llp→limited liability partnership`, `plc→public limited company`, `huf→hindu undivided family`, `sarl→...`, `sas→...`, `sa→...`

**Address abbreviation expansion map** (from `src/config.py` `NORM`):
`rd→road`, `st→street`, `ave→avenue`, `blvd→boulevard`, `dr→drive`, `ln→lane`, `hwy→highway`, `apt→apartment`, `ste→suite`, `fl→floor`, `tq→taluka`, `dist→district`, `opp→opposite`, `extn→extension`

---

## 4. Phase 2 Outputs

### 4.1 Intermediate: Flat Candidate Pairs DataFrame (in-memory only)

This is the working DataFrame used internally during the pipeline run. It is **not saved to disk directly** — it is the pre-groupby form.

| Column | Type | Source | Description |
|---|---|---|---|
| `source1_entity_id` | str | S1 `entity_id` | Identifies the S1 entity in this candidate pair |
| `candidate_entity_id` | str | S2 or S3 `entity_id` | Identifies the S2 or S3 candidate entity |
| `tfidf_score` | float | TF-IDF blocker only | Cosine similarity score from char-3gram TF-IDF (present only for TF-IDF pairs; absent for exact/sorted-neighbourhood pairs) |
| `normalized_name_s1` | str | joined from S1 | After `.join(s1_lookup...)` in `pipeline.py` |
| `normalized_address_s1` | str | joined from S1 | After `.join(s1_lookup...)` |
| `country_s1` | str | joined from S1 | After `.join(s1_lookup...)` |
| `normalized_name_cand` | str | joined from S2/S3 | After `.join(s23_lookup...)` |
| `normalized_address_cand` | str | joined from S2/S3 | After `.join(s23_lookup...)` |
| `country_cand` | str | joined from S2/S3 | After `.join(s23_lookup...)` |

**Key properties:**
- One row per unique `(source1_entity_id, candidate_entity_id)` pair
- A single S1 entity can appear in many rows (one per candidate)
- `candidate_entity_id` may be any S2-prefixed or S3-prefixed entity ID — they are merged into one pool (`s2s3`) before blocking
- Deduplication is applied: `.drop_duplicates(subset=["source1_entity_id","candidate_entity_id"])`
- Country-partitioned: TF-IDF and sorted-neighbourhood blocking only pairs records with matching `country`

### 4.2 Saved Output 1: `output/candidate_pairs.tsv`

Written by `build_full_submission()` then renamed, after groupby-aggregation.

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | str | S1 entity ID (e.g. `S1-925783039`) |
| `candidate_entity_ids` | str | Comma-separated list of all candidate S2/S3 entity IDs for this S1 entity |

**Properties:**
- Format: TSV (`\t` separator), no index column
- Exactly one row per S1 entity (guaranteed by `build_full_submission`)
- S1 entities with zero blocking candidates get an empty string `""` in `candidate_entity_ids`
- Row count equals number of S1 entities in the split (2,206,821 for train; 1,732,544 for test)
- S2 and S3 candidates are **not distinguished** — both appear in the same comma-separated list

### 4.3 Saved Output 2: `features/{split}_features.parquet`

Written when `save_features=True` (default).

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | str | S1 entity ID |
| `candidate_entity_id` | str | S2 or S3 candidate entity ID |
| *feature columns* | float | Output of `build_feature_matrix(candidates)` — **NOT YET IMPLEMENTED** (Phase 3) |

**Properties:**
- Format: Parquet (via `pyarrow`)
- Path: `ROOT/features/{split}_features.parquet` (e.g. `features/test_features.parquet`)
- One row per `(source1_entity_id, candidate_entity_id)` pair — same granularity as §4.1
- **⚠️ `build_feature_matrix` is imported from `src.features.pair_features` but is NOT defined there. The pipeline will raise `ImportError` at Phase 3. This is the primary blocker for Phase 3.**

### 4.4 Saved Output 3: `output/matching_results.tsv`

Final submission file.

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | str | S1 entity ID |
| `matched_entity_ids` | str | Comma-separated list of matched S2/S3 IDs after scoring + thresholding. Empty string `""` for singletons. |

**Properties:**
- Format: TSV
- Exactly one row per S1 entity
- Requires trained model at `src/models/saved/model_v1.joblib` — **does not exist yet**

---

## 5. Phase 3 Expected Input

**Phase 3 (Feature Engineering) consumes the in-memory enriched candidates DataFrame** described in §4.1, passed directly from `pipeline.py` as the `candidates` argument to `build_feature_matrix()`.

### Required columns on the `candidates` DataFrame entering Phase 3

| Column | Type | Guaranteed Present? | Notes |
|---|---|---|---|
| `source1_entity_id` | str | ✅ Yes | S1 entity ID |
| `candidate_entity_id` | str | ✅ Yes | S2 or S3 entity ID |
| `tfidf_score` | float | ⚠️ Partial | Only for TF-IDF-sourced pairs; NaN for exact/SNB pairs after concat |
| `normalized_name_s1` | str | ✅ Yes | Normalised S1 name (joined in pipeline.py) |
| `normalized_address_s1` | str | ✅ Yes | Normalised S1 address (joined) |
| `country_s1` | str | ✅ Yes | S1 country (joined) |
| `normalized_name_cand` | str | ✅ Yes | Normalised candidate name (joined) |
| `normalized_address_cand` | str | ✅ Yes | Normalised candidate address (joined) |
| `country_cand` | str | ✅ Yes | Candidate country (joined) |

### What `build_feature_matrix(candidates)` must return

A `pd.DataFrame` of shape `(N_pairs, N_features)` where:
- Each row corresponds to one row of `candidates` (same order)
- All columns are numeric (`float32` or `float64`)
- No `source1_entity_id` or `candidate_entity_id` columns in the returned matrix (those are prepended by `pipeline.py` before saving to parquet)

### What Phase 3 must define

```python
# src/features/pair_features.py  — must add this top-level function
def build_feature_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  candidates DataFrame with columns listed above (§5)
    Output: numeric feature DataFrame, same row count, no ID columns
    """
    ...
```

This function is the **single integration point** between Phase 2 blocking and Phase 3 features.

---

## 6. Blocking Methods Implemented (Phase 2)

| Blocker | File | Config Key | Returns `tfidf_score`? | Country-partitioned? |
|---|---|---|---|---|
| TF-IDF char-3gram | `src/blocking/tfidf_blocking.py` | `BLOCKING["tfidf_top_k"]` = 50, min_score = 0.10 | ✅ Yes | ✅ Yes |
| Exact first-token | `src/blocking/exact_blocking.py` | `BLOCKING["address_min_token_len"]` = 4 | ❌ No | ✅ Yes (on `country`) |
| Sorted Neighbourhood | `src/blocking/token_blocking.py` | `BLOCKING["sorted_neighbourhood_window"]` = 10 | ❌ No | ✅ Yes (filters on `country`) |
| FAISS | `src/blocking/faiss_blocking.py` | — | N/A | **Not called from pipeline.py** |

---

## 7. Known Issues / Risks

| # | Severity | Issue | Location | Impact |
|---|---|---|---|---|
| 7.1 | 🔴 CRITICAL | `DATA_DIR` points to `student_resource/dataset/` which does not exist. Actual data is in `data/` | `src/config.py` line 13 | Pipeline fails immediately on any split |
| 7.2 | 🔴 CRITICAL | `build_feature_matrix` is imported in `pipeline.py` but not defined in `src/features/pair_features.py` | `src/pipeline.py` line 26 | `ImportError` at startup; Phase 3 cannot run |
| 7.3 | 🟠 HIGH | All feature functions in `name_features.py`, `address_features.py` return hardcoded 0/False (TODO stubs) | `src/features/*.py` | No real features will be computed until Phase 3 is implemented |
| 7.4 | 🟠 HIGH | No trained model exists at `src/models/saved/model_v1.joblib` | `src/models/predict.py` | `predict_matches()` will raise `FileNotFoundError` |
| 7.5 | 🟡 MEDIUM | `tfidf_score` column only present for TF-IDF pairs; NaN for exact/SNB pairs after concat. `build_feature_matrix` must handle this | `src/pipeline.py` lines 65–69 | Potential NaN leakage into feature matrix |
| 7.6 | 🟡 MEDIUM | `configs/config.yaml` (Phase 0 scaffolding) is not loaded by any current code but exists alongside `src/config.py`. Risk of confusion | `configs/config.yaml` | Future developers may edit the wrong config file |
| 7.7 | 🟡 MEDIUM | Experiment branches (`experiment/*`) exist locally but have not been pushed to `origin` | `.git/` | Branches will be lost if local repo is deleted |
| 7.8 | 🟡 MEDIUM | S3 bucket has all prefix structure but zero data files — all prefixes are empty placeholders | S3 | No cloud backup of datasets or artifacts |
| 7.9 | 🟢 LOW | `faiss_blocking.py` exists in `src/blocking/` but is never called | `src/blocking/faiss_blocking.py` | Dead code; may cause confusion about what blocking is active |
| 7.10 | 🟢 LOW | `anyascii` is listed in `requirements.txt` but may not be installed in `venv` | `requirements.txt` | Transliteration silently falls back to `unidecode` or NFKD |
