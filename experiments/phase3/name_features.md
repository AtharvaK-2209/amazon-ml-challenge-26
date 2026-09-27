# Phase 3 / Part 1: Business Name Pairwise Features

## Selected Features

The features implemented in `src/features/name_features.py` are explicitly designed to capture the noise patterns observed during Phase 1 EDA. We avoid dumping an arbitrary laundry list of string metrics, opting for a small, purposeful set.

| Feature | Why it exists | Range | Missing-value behavior | Phase 1 evidence |
|---------|---------------|-------|------------------------|------------------|
| `name_ratio` | Standard string similarity. Good for minor typos and character substitutions. | [0, 1] | 0.0 (if only 1 missing), 1.0 (if both missing) | ~11% of names have minor typos or spacing differences. |
| `name_token_sort` | Handles token reordering. Alphabetizes tokens before comparison. | [0, 1] | Same as above | DBA names or names like "Corp ABC" vs "ABC Corp" are common. |
| `name_norm_edit` | Normalized edit distance penalizing length differences more strictly than ratio. | [0, 1] | Same as above | Good for separating short acronyms from full words. |
| `name_jaccard` | Token intersection over union. Requires exact token matches. | [0, 1] | Same as above | 28% of records share abbreviations; exact token overlap is a strong signal. |
| `name_len_ratio` | Structural feature: `min(len)/max(len)`. | [0, 1] | Same as above | Lengths vary from 3 to 100+ chars. Detects abbreviation-to-full-name pairs. |
| `same_legal_suffix` | Binary flag if legal suffixes exactly match. | {0, 1} | 0.0 if missing or no suffix | ~29% of S1/S2 names contain a legal suffix. |
| `name_s1_missing` | Binary flag for missing S1 name. | {0, 1} | 1.0 if missing | S2/S3 have a small number of null business names (46 and 59). |
| `name_s2_missing` | Binary flag for missing S2/S3 name. | {0, 1} | 1.0 if missing | S2/S3 have a small number of null business names (46 and 59). |

## Rejected Features

Several features were investigated but rejected to maintain a clean, high-signal feature matrix:

1. **`fuzz.partial_ratio`**:
   - *Why rejected*: Produces 100% matches when a short acronym matches a tiny substring of a longer name (e.g., "A" matches 100% inside "Apple"). It is highly unstable for short business names.
2. **`fuzz.token_set_ratio`**:
   - *Why rejected*: Similar to partial ratio, it aggressively drops duplicated/subset tokens. Highly correlated with `token_sort_ratio` but produces more false positives on generic words (e.g. "India").
3. **TF-IDF Cosine Similarity inside feature extractor**:
   - *Why rejected*: Requires fitting a vocabulary across the whole corpus. This is already handled at scale during the **Blocking Phase (B01)**. Re-computing TF-IDF dynamically per candidate pair inside `name_features.py` is computationally wasteful and breaks the stateless requirement of the feature extractor. It will be appended later during the join in `pair_features.py`.
4. **Token Count Difference**:
   - *Why rejected*: Found to be highly redundant with `name_len_ratio` and `name_jaccard`.
5. **One-hot legal suffix vectors**:
   - *Why rejected*: Creating 20+ columns for every possible suffix (Pvt, Ltd, LLC) dilutes the XGBoost tree depth. A binary `same_legal_suffix` provides the necessary discriminative signal.
