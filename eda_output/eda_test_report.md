# Phase 1 Complete — EDA + Noise Specification
## Amazon ML Challenge 2026 · Business Entity Resolution

---

## 📁 `eda_output/` Structure

```
eda_output/
├── noise_spec.json            ← 🔑 Machine-readable spec (consumed by Phase 2 & 3)
├── data_quality_report.md     ← 📄 Human-readable quality report
├── 01_overview.csv            ← Dataset shape + null counts
├── 06_noise_patterns.csv      ← Noise flag rates (%) per source
├── 11_length_stats.csv        ← Length stats: mean / median / std / p95
├── 12_samples.csv             ← 5 sample records × country × source
└── plots/
    ├── 02_country_distribution.png
    ├── 03_length_distributions.png
    ├── 04_token_distributions.png
    ├── 05_script_distribution.png
    ├── 06_noise_patterns.png
    ├── 07_top_tokens.png
    ├── 08_missing_values.png
    └── 10_cross_source_country.png
```

---

## 🔑 `noise_spec.json` — Spec Schema

The JSON is structured into 8 top-level keys that downstream phases import directly:

```json
{
  "meta":                    // Phase label, generated_at, next_phase
  "dataset_overview":        // rows / nulls / duplicates per source
  "country_distribution":    // record counts per country per source
  "script_distribution":     // Latin / Devanagari / other counts per source
  "noise_pattern_rates_pct": // % of records affected by each noise type
  "length_stats":            // mean/median/std/min/max/p95 per field
  "normalisation_rules":     // 9 priority-ordered rules (see below)
  "blocking_hints":          // 6 strategy recommendations (see below)
  "country_flags":           // per-country handling flags
  "global_flags":            // pipeline-wide constraints
}
```

---

## ✅ Normalisation Rules (from spec)

Ordered by `field` then `priority`:

| ID | Field | Action | Trigger |
|----|-------|--------|---------|
| **N04** | `business_name` | `transliterate_to_latin` | S2=6.2%, S3=3.3% Devanagari — S1 is 100% Latin |
| **N01** | `business_name` | `lowercase` | S2 has **17.25% ALL-CAPS** names |
| **N05** | `business_name` | `strip_accents` | France = 15% of test; zero-shot |
| **N02** | `business_name` | `expand_legal_suffixes` | **~29%** of all names use Pvt/Ltd/Corp/Inc/SARL… |
| **N03** | `business_name` | `replace & → and` | ~4–5% across all sources |
| **N06** | `business_name` | `strip_punctuation` | Residual cleanup |
| **N08** | `business_address` | `fill_null_address → ""` | S2=129K, S3=136K null addresses |
| **N07** | `business_address` | `expand_address_abbreviations` | S2/S3 abbreviation rate **2× higher** than S1 |
| **N09** | `business_address` | `do_not_use_pin_as_blocking_key` | **95%** addresses have no PIN/ZIP |

---

## 🔒 Blocking Strategy Hints (from spec)

| ID | Strategy | Key Insight from EDA |
|----|----------|----------------------|
| **B03** | Country partition | Hard filter — country proportions identical across sources |
| **B01** | Name TF-IDF (char 3-grams) | Most reliable anchor; handles abbrev noise post-N02 |
| **B04** | Address token overlap (IDF-weighted) | Catches name-only failures; skip for null-address records |
| **B05** | First-token exact match | O(N), fast; filter out low-IDF first tokens |
| **B02** | Sorted neighbourhood on name | Catches transpositions missed by TF-IDF |
| **B06** | Devanagari phonetic fallback | Essential for 303K S2 Devanagari names |

---

## ⚠️ Critical Global Constraints

| Constraint | Value |
|-----------|-------|
| PIN/ZIP usable as blocking key | ❌ No — 95% missing |
| Safe to drop null-address records | ❌ No — ~265K records, must match on name |
| Zero-shot country (France) | ⚠️ Yes — no training examples, pipeline must generalise |
| Evaluation metric | F₀.₅ macro (precision-heavy — false merges cost 2×) |
| Must submit ALL S1 records | ✅ 1,732,544 rows required in output |

---

## How Downstream Phases Consume This

```python
# Phase 2 — Normalisation
import json
spec = json.load(open("eda_output/noise_spec.json"))
rules = spec["normalisation_rules"]          # iterate priority-ordered
country_flags = spec["country_flags"]        # per-country logic
global_flags = spec["global_flags"]

# Phase 3 — Blocking
blocking_hints = spec["blocking_hints"]      # strategy list
noise_rates = spec["noise_pattern_rates_pct"]
length_stats = spec["length_stats"]          # set TF-IDF params
```
