# Name Feature Engineering

## 1. Objective
This module (`src/features/name_features.py`) generates reusable pairwise business-name similarity features for determining if an S1 entity matches a candidate S2/S3 entity.

## 2. Normalization Assumptions
The features assume the input names (`s1_name`, `candidate_name`) have already passed through the Phase 2 normalizer, meaning they are lowercased, stripped of diacritics/punctuation, transliterated (if Devanagari), and abbreviations like "St." are expanded. Specifically, it expects the `normalized_name` representation.

## 3. Features

| Feature | Description | Range | Missing-value behavior | Normalization assumption |
|---|---|---|---|---|
| `name_levenshtein` | Raw Levenshtein edit distance. | `[0, max_len]` | 0 if both missing, else `len(non_missing)` | Clean string, no punctuation. |
| `name_fuzz_ratio` | RapidFuzz standard ratio. | `[0, 100]` | 100.0 if both missing, 0.0 if one missing | Strings are already lowercased. |
| `name_wratio` | RapidFuzz WRatio. | `[0, 100]` | Same as above | Weighted ratio for strings of diff length. |
| `name_token_sort_ratio` | RapidFuzz token sort ratio. | `[0, 100]` | Same as above | Robust to word order differences. |
| `name_token_set_ratio` | RapidFuzz token set ratio. | `[0, 100]` | Same as above | Robust to subset matching and duplicates. |
| `name_jaccard` | Token-level intersection over union. | `[0.0, 1.0]` | 1.0 if both missing, 0.0 if one missing | Assumes standard whitespace tokenization. |
| `name_tfidf_cosine` | Cosine similarity using word-level TF-IDF. | `[0.0, 1.0]` | 1.0 if both missing, 0.0 if one missing | Uses a pre-fit `TfidfVectorizer(analyzer="word")`. |
| `name_char_cosine` | Cosine similarity using char-ngram TF-IDF. | `[0.0, 1.0]` | Same as above | Uses a pre-fit `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3))`. |
| `name_length_diff` | `abs(len(n1) - len(n2))` | `[0, ∞)` | 0 if both missing, else `len(non_missing)` | Character count. |
| `name_length_ratio` | `min(len) / max(len)` | `[0.0, 1.0]` | 1.0 if both missing, 0.0 if one missing | Character count. |
| `name_token_count_diff`| `abs(len(tokens1) - len(tokens2))` | `[0, ∞)` | 0 if both missing, else `len(non_missing_tokens)` | Whitespace tokenization. |

## 4. Feature Definitions
- **name_levenshtein**: Minimum number of single-character edits to change n1 into n2.
- **name_fuzz_ratio**: Simple normalized Levenshtein similarity metric scaled to 100.
- **name_wratio**: RapidFuzz's weighted ratio that applies different algorithms depending on string lengths.
- **name_token_sort_ratio**: Sorts the words alphabetically before computing the fuzz ratio.
- **name_token_set_ratio**: Identifies common tokens and unique tokens, scoring the subsets.
- **name_jaccard**: `|A ∩ B| / |A ∪ B|` where A and B are sets of whitespace-delimited tokens.
- **name_length_diff / _ratio**: Arithmetic operations on the character lengths of both strings.
- **name_token_count_diff**: Absolute difference in the number of tokens.

## 5. Phase 1 Noise Patterns
The following patterns were explicitly observed in the Phase 1 EDA (`noise_spec.json`) and motivate these features:
- **Abbreviations/Legal Suffixes**: "pvt" vs "private", "ltd" vs "limited". (Normalisation expands these, but the metrics capture residual mismatches).
- **Ampersands**: "&" vs "and" (handled by normalizer, but variations create Jaccard noise).
- **Typos**: RapidFuzz ratio and Levenshtein distance directly address spelling errors and OCR noise.
- **Word Order**: DBA names frequently have reordered tokens (e.g. "Corporation of Chennai" vs "Chennai Corporation"), which is perfectly captured by `name_token_sort_ratio` and `name_jaccard`.

## 6. TF-IDF Configuration
- **Word Analyzer**: `TfidfVectorizer(analyzer='word')`
- **Fitting/Lifecycle assumptions**: The feature generator expects the vectorizer to be pre-fit on the global corpus (S1+S2+S3) *outside* this function, then passed in. This avoids catastrophically slow performance of fitting a vectorizer on just two strings per function call.

## 7. Character Cosine Configuration
- **Character n-gram Configuration**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3))`
- **Fitting/Lifecycle assumptions**: Same as above. Expects a pre-fit vectorizer to be passed to `generate_name_features`.

## 8. Missing Values
Deterministic behavior prevents NaN/Infinity bleeding into XGBoost:
- `None`, `np.nan`, `"NaN"`, and empty strings are treated as empty strings `""`.
- If *both* are empty: Distance is 0, similarity ratios are 100.0/1.0, and differences are 0.
- If *one* is empty: Similarity is 0.0, lengths and differences reflect the length of the non-empty string.

## 9. Testing
Unit tests in `tests/test_name_features.py` validate behavior against identical names, missing values (NaN/None/empty), typos, word reordering, legal suffix mismatches, and ampersand translations.
