"""
=============================================================
  Amazon ML Challenge 2026
  Phase 1 Output: Noise Specification Generator

  Reads the CSVs produced by eda_test_data.py and emits:
    eda_output/noise_spec.json          ← machine-readable spec
    eda_output/data_quality_report.md   ← human-readable report

  These two files are the ONLY outputs consumed by Phase 2
  (Normalisation) and Phase 3 (Blocking).
=============================================================
"""

import json, csv, datetime
from pathlib import Path

OUT   = Path("eda_output")
PLOTS = OUT / "plots"

# ── Read EDA CSVs ──────────────────────────────────────────

def read_csv_dict(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

overview_rows  = read_csv_dict(OUT / "01_overview.csv")
noise_rows     = read_csv_dict(OUT / "06_noise_patterns.csv")
stats_rows     = read_csv_dict(OUT / "11_length_stats.csv")

# --- parse overview ---
overview = {}
for r in overview_rows:
    src = r["Source"]
    overview[src] = {
        "rows":              int(r["Rows"]),
        "null_business_name":    int(r["Null business_name"]),
        "null_business_address": int(r["Null business_address"]),
        "null_country":          int(r["Null country"]),
        "unique_countries":      int(r["Unique countries"]),
        "duplicate_entity_id":   int(r["Duplicate entity_id"]),
    }

# --- parse noise patterns ---
noise = {}
for r in noise_rows:
    pattern = r[""]
    noise[pattern] = {s: float(r[s]) for s in ["S1", "S2", "S3"]}

# --- parse length stats ---
stats = {}
for r in stats_rows:
    key = (r["source"], r["field"])
    stats[key] = {k: float(r[k]) for k in ["mean","median","std","min","max","p95"]}

# Country distribution (hardcoded from EDA printout — already accurate)
country_dist = {
    "S1": {"India": 809986,  "US": 663106,  "France": 259452},
    "S2": {"India": 2312565, "US": 1871330, "France": 703378},
    "S3": {"India": 2405000, "US": 1945701, "France": 731615},
}

# Script distribution (from EDA printout)
script_dist = {
    "S1": {"latin": 1732544, "devanagari": 0,      "other": 0,      "empty": 0},
    "S2": {"latin": 4351900, "devanagari": 303467,  "other": 231749, "empty": 46},
    "S3": {"latin": 4783598, "devanagari": 170091,  "other": 128330, "empty": 59},
}

# ── Build Normalisation Rules ──────────────────────────────
# Each rule has:
#   id, priority, trigger_condition, applies_to, action, params, rationale

norm_rules = [
    {
        "id": "N01",
        "priority": 1,
        "field": "business_name",
        "trigger": "name_all_caps: S2=17.25%, S3=2.72%",
        "action": "lowercase",
        "params": {},
        "rationale": "S2 has 17.25% ALL-CAPS names vs 0% in S1. Lowercasing unifies case before any string comparison.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N02",
        "priority": 2,
        "field": "business_name",
        "trigger": "name_has_abbrev: ~29% across all sources",
        "action": "expand_legal_suffixes",
        "params": {
            "expansions": {
                "pvt": "private", "ltd": "limited", "llc": "limited liability company",
                "corp": "corporation", "inc": "incorporated", "co": "company",
                "llp": "limited liability partnership", "plc": "public limited company",
                "huf": "hindu undivided family", "sarl": "societe a responsabilite limitee",
                "sas": "societe par actions simplifiee", "sa": "societe anonyme"
            }
        },
        "rationale": "~29% of names use abbreviated legal suffixes. Expanding to canonical form prevents Corp/Corporation mismatch. French legal forms (SARL, SAS, SA) included for the zero-shot France set.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N03",
        "priority": 3,
        "field": "business_name",
        "trigger": "name_has_ampersand: ~4-5% across all sources",
        "action": "replace_token",
        "params": {"find": "&", "replace": "and"},
        "rationale": "4-5% of names use '&' instead of 'and'. Normalise to 'and' for consistent token matching.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N04",
        "priority": 4,
        "field": "business_name",
        "trigger": "script_dist: S2=10.9% non-Latin, S3=5.8% non-Latin; S1=100% Latin",
        "action": "transliterate_to_latin",
        "params": {
            "tool": "indic_transliteration or anyascii",
            "source_scripts": ["devanagari", "arabic", "other"],
            "target_script": "latin"
        },
        "rationale": "S1 is 100% Latin but S2 has 303K (6.2%) and S3 has 170K (3.3%) Devanagari names. Direct string similarity fails cross-script. Transliterate non-Latin to Latin before any similarity computation.",
        "applies_to_sources": ["S2", "S3"],
        "applies_to_countries": ["India"],
    },
    {
        "id": "N05",
        "priority": 5,
        "field": "business_name",
        "trigger": "French accented characters in France set (15% of all records)",
        "action": "strip_accents",
        "params": {
            "method": "unicodedata.normalize('NFKD') + ascii encode",
            "examples": {"é": "e", "è": "e", "ê": "e", "à": "a", "ç": "c", "ô": "o", "î": "i"}
        },
        "rationale": "France is a zero-shot country (15% of test set). French business names contain accented Latin characters that cause false mismatches. Strip diacritics after transliteration.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["France"],
    },
    {
        "id": "N06",
        "priority": 6,
        "field": "business_name",
        "trigger": "residual punctuation after above steps",
        "action": "strip_punctuation_and_extra_whitespace",
        "params": {"keep_hyphens": True, "keep_digits": True},
        "rationale": "Remove residual commas, dots, brackets. Keep hyphens (compound names) and digits (e.g. '3M', '7-Eleven' are discriminative).",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N07",
        "priority": 1,
        "field": "business_address",
        "trigger": "addr_has_abbrev: S1=12.65%, S2=26.68%, S3=24.55%",
        "action": "expand_address_abbreviations",
        "params": {
            "expansions": {
                "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
                "dr": "drive", "ln": "lane", "hwy": "highway", "fwy": "freeway",
                "apt": "apartment", "ste": "suite", "fl": "floor",
                "nagar": "nagar", "marg": "marg", "vihar": "vihar",
                "enclave": "enclave", "colony": "colony", "extn": "extension",
                "tq": "taluka", "dist": "district", "opp": "opposite",
                "rue": "rue", "bd": "boulevard", "av": "avenue"
            }
        },
        "rationale": "Address abbreviation rate is 2x higher in S2/S3 (26-27%) vs S1 (12-13%). Expanding abbreviations is the single highest-impact normalisation for address matching.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N08",
        "priority": 2,
        "field": "business_address",
        "trigger": "null_business_address: S2=129408 (2.65%), S3=136098 (2.68%)",
        "action": "fill_null_address",
        "params": {"fill_value": ""},
        "rationale": "2.65-2.68% of S2/S3 records have no address. Fill with empty string so they still participate in name-only matching. Do NOT drop these records.",
        "applies_to_sources": ["S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
    {
        "id": "N09",
        "priority": 3,
        "field": "business_address",
        "trigger": "addr_missing_pin: ~95% across all sources",
        "action": "do_not_use_pin_as_blocking_key",
        "params": {},
        "rationale": "95% of addresses lack a 5-6 digit PIN/ZIP code. Any blocking strategy that requires a PIN will drop 95% of record pairs. Use full normalised address string for TF-IDF/BM25 blocking instead.",
        "applies_to_sources": ["S1", "S2", "S3"],
        "applies_to_countries": ["US", "India", "France"],
    },
]

# ── Build Blocking Hints ───────────────────────────────────
blocking_hints = [
    {
        "id": "B01",
        "strategy": "name_trigram_tfidf",
        "description": "Build a TF-IDF index (char 3-grams) over normalised business names. For each S1 record, retrieve top-K candidates from S2+S3 by cosine similarity.",
        "derived_from": ["N01","N02","N03","N04","N05","N06"],
        "params": {"analyzer": "char_wb", "ngram_range": [3,3], "top_k": 50, "min_score": 0.1},
        "recall_ceiling_note": "High-recall anchor since ~29% of names share legal suffix tokens — TF-IDF naturally down-weights these."
    },
    {
        "id": "B02",
        "strategy": "sorted_neighbourhood_name_prefix",
        "description": "Sort all records by normalised name; slide a window of size W. Any record within the window is a candidate pair.",
        "derived_from": ["N01","N02","N06"],
        "params": {"window_size": 10},
        "recall_ceiling_note": "Fast O(N log N) complement to TF-IDF. Catches transpositions that survive normalisation."
    },
    {
        "id": "B03",
        "strategy": "country_partition",
        "description": "Always restrict candidate pairs to same-country records. Never match US↔India or US↔France.",
        "derived_from": ["country_dist"],
        "params": {"partition_key": "country"},
        "recall_ceiling_note": "100% safe assumption per problem statement. Reduces search space by ~3x."
    },
    {
        "id": "B04",
        "strategy": "address_token_overlap",
        "description": "Build inverted index over normalised address tokens (min 4 chars). Candidate if S1 and S2/S3 share ≥1 rare token (IDF > threshold).",
        "derived_from": ["N07","N08"],
        "params": {"min_token_len": 4, "idf_threshold": 3.0},
        "recall_ceiling_note": "Compensates for name-only blocking misses when addresses are clean. Disable for null-address records."
    },
    {
        "id": "B05",
        "strategy": "name_first_token_exact",
        "description": "Group records by their first normalised name token. All records sharing the same first token are candidates.",
        "derived_from": ["N01","N02"],
        "params": {"min_token_len": 3},
        "recall_ceiling_note": "Very fast O(N) grouping. High precision on name-stable businesses. High false-positive rate for generic first tokens (e.g. 'india', 'national') — filter by IDF."
    },
    {
        "id": "B06",
        "strategy": "devanagari_transliteration_fallback",
        "description": "For non-Latin S2/S3 records that map to a transliterated form, also generate candidate pairs using phonetic hashing (Soundex/Metaphone) of the transliterated name.",
        "derived_from": ["N04"],
        "params": {"phonetic_algo": "metaphone", "applies_to_scripts": ["devanagari"]},
        "recall_ceiling_note": "Handles transliteration noise: 'Shri' vs 'Sri', 'Kumar' vs 'Kumar'. Essential for 303K Devanagari records in S2."
    },
]

# ── Per-country flags ──────────────────────────────────────
country_flags = {
    "US": {
        "zero_shot": False,
        "dominant_script": "latin",
        "address_format_notes": "Street number + street name + city + state abbreviation + optional ZIP. ZIP present in <5% of records (addr_missing_pin ~95%).",
        "special_handling": []
    },
    "India": {
        "zero_shot": False,
        "dominant_script": "latin_and_devanagari",
        "address_format_notes": "Highly variable: Plot/Flat/Survey numbers, locality names, taluka/district/state. Landmark-based references common (~5%). PIN codes rarely present.",
        "special_handling": ["transliterate_devanagari", "expand_tq_dist_abbreviations"]
    },
    "France": {
        "zero_shot": True,
        "dominant_script": "latin_with_accents",
        "address_format_notes": "French format: street number + rue/boulevard/avenue + city + postal code (5-digit). Postal code present more often than US/India.",
        "special_handling": ["strip_french_accents", "handle_sarl_sas_sa_suffixes"],
        "warning": "France does NOT appear in the training data. Do not hard-code country logic. All normalisation must be generic enough to handle unseen countries."
    }
}

# ── Assemble full spec ─────────────────────────────────────
spec = {
    "meta": {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "phase": "1 — EDA + Noise Discovery",
        "next_phase": "2 — Normalisation (names + addresses)",
        "description": "Machine-readable noise specification derived from Phase 1 EDA on test data. Consumed directly by the normalisation pipeline."
    },
    "dataset_overview": overview,
    "country_distribution": country_dist,
    "script_distribution": script_dist,
    "noise_pattern_rates_pct": noise,
    "length_stats": {
        f"{src}_{field}": vals
        for (src, field), vals in stats.items()
    },
    "normalisation_rules": norm_rules,
    "blocking_hints": blocking_hints,
    "country_flags": country_flags,
    "global_flags": {
        "pin_zip_usable_as_blocking_key": False,
        "pin_coverage_pct": {"S1": 4.37, "S2": 5.07, "S3": 5.04},
        "safe_to_drop_null_address_records": False,
        "must_handle_zero_shot_countries": True,
        "evaluation_metric": "F_0.5 (macro-averaged, precision-heavy)",
        "precision_recall_tradeoff": "Prefer precision — false merges penalised 2x vs missed matches",
        "total_test_s1_records_to_submit": 1732544,
    }
}

spec_path = OUT / "noise_spec.json"
with open(spec_path, "w") as f:
    json.dump(spec, f, indent=2)
print(f"✅ Written: {spec_path}")

# ── Generate data_quality_report.md ───────────────────────
md_lines = []
A = md_lines.append

A("# Phase 1 EDA — Data Quality & Noise Specification")
A("## Amazon ML Challenge 2026 · Business Entity Resolution")
A(f"\n> **Generated:** {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ")
A("> **Input:** `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`  ")
A("> **Outputs consumed by:** Phase 2 (Normalisation), Phase 3 (Blocking)\n")
A("> **Machine-readable spec:** [`eda_output/noise_spec.json`](noise_spec.json)\n")

A("---\n")

# --- Dataset Scale ---
A("## 1. Dataset Scale\n")
A("| Source | Rows | Null Name | Null Address | Null Country | Duplicate IDs |")
A("|--------|------|-----------|-------------|--------------|---------------|")
for src, o in overview.items():
    A(f"| **{src}** | {o['rows']:,} | {o['null_business_name']} | {o['null_business_address']:,} | {o['null_country']} | {o['duplicate_entity_id']} |")
A("")
A("> S1 (reference source) has **zero nulls**. S2 & S3 have ~2.65–2.68% null addresses — these records must still be matched on name alone.")
A("")

# --- Country Distribution ---
A("## 2. Country Distribution\n")
A("| Country | S1 | S2 | S3 | Notes |")
A("|---------|----|----|-----|-------|")
for country in ["India", "US", "France"]:
    note = "⚠️ **Zero-shot** — not in training data" if country == "France" else ""
    vals = [f"{country_dist[s][country]:,}" for s in ["S1","S2","S3"]]
    A(f"| {country} | {' | '.join(vals)} | {note} |")
A("")
A("> Country proportions are consistent across all 3 sources (~47% India / 38% US / 15% France).  ")
A("> **Action:** Use `country` as a hard partition key in blocking (never match across countries).")
A("")

# --- Script Distribution ---
A("## 3. Script Distribution (Business Names)\n")
A("| Script | S1 | S2 | S3 |")
A("|--------|----|----|-----|")
for script in ["latin","devanagari","other","empty"]:
    vals = [f"{script_dist[s].get(script, 0):,}" for s in ["S1","S2","S3"]]
    A(f"| {script} | {' | '.join(vals)} |")
A("")
A("> **Critical finding:** S1 is 100% Latin. S2 has 303K Devanagari + 232K other-script names.  ")
A("> Direct string similarity will **fail** for these pairs → transliteration is mandatory.")
A("")

# --- Noise Pattern Table ---
A("## 4. Noise Pattern Rates\n")
A("| Pattern | S1 (%) | S2 (%) | S3 (%) | Severity | Normalisation Rule |")
A("|---------|--------|--------|--------|----------|--------------------|")
noise_meta = {
    "name_has_abbrev":     ("🔴 High",   "N02 — Expand legal suffixes"),
    "name_has_ampersand":  ("🟡 Medium", "N03 — Replace & → and"),
    "name_all_caps":       ("🔴 High",   "N01 — Lowercase all"),
    "name_has_digit":      ("🟢 Low",    "N06 — Keep digits (discriminative)"),
    "addr_missing_pin":    ("🔴 Critical","B01–B05 — Never use PIN as blocking key"),
    "addr_has_landmark":   ("🟡 Medium", "N07 — Keep; use full address TF-IDF"),
    "addr_has_abbrev":     ("🔴 High",   "N07 — Expand address abbreviations"),
}
for pattern, rates in noise.items():
    severity, rule = noise_meta.get(pattern, ("🟢 Low", "—"))
    A(f"| `{pattern}` | {rates['S1']} | {rates['S2']} | {rates['S3']} | {severity} | {rule} |")
A("")

# --- Length Stats ---
A("## 5. Field Length Statistics\n")
A("| Source | Field | Mean | Median | Std | Min | Max | p95 |")
A("|--------|-------|------|--------|-----|-----|-----|-----|")
for (src, field), vals in stats.items():
    A(f"| {src} | {field} | {vals['mean']} | {vals['median']} | {vals['std']} | {int(vals['min'])} | {int(vals['max'])} | {vals['p95']} |")
A("")
A("> p95 name length is 36–42 chars — cap TF-IDF token length at 50 chars.  ")
A("> p95 address length is 94–105 chars — no truncation needed for BM25/TF-IDF.")
A("")

# --- Normalisation Rules ---
A("## 6. Normalisation Rules (Priority-Ordered)\n")
A("These rules must be applied **in priority order** to produce a clean, canonical form for each field.\n")
A("| ID | Priority | Field | Trigger | Action | Key Params |")
A("|----|----------|-------|---------|--------|------------|")
for r in sorted(norm_rules, key=lambda x: (x["field"], x["priority"])):
    params_str = ", ".join(f"`{k}={v}`" for k, v in list(r["params"].items())[:2]) if r["params"] else "—"
    A(f"| **{r['id']}** | {r['priority']} | `{r['field']}` | {r['trigger'][:60]}… | `{r['action']}` | {params_str} |")
A("")
A("### Normalisation Pipeline Order (per record)\n")
A("```")
A("For business_name:")
A("  1. [N04] Transliterate non-Latin → Latin  (S2/S3 India only)")
A("  2. [N01] Lowercase")
A("  3. [N05] Strip accents / diacritics        (France)")
A("  4. [N02] Expand legal suffixes             (all)")
A("  5. [N03] Replace & → and                  (all)")
A("  6. [N06] Strip punctuation, collapse whitespace")
A("")
A("For business_address:")
A("  1. [N08] Fill nulls with ''")
A("  1. [N01] Lowercase (reuse)")
A("  2. [N05] Strip accents (reuse, France)")
A("  3. [N07] Expand address abbreviations")
A("  4. [N06] Strip punctuation, collapse whitespace")
A("```")
A("")

# --- Blocking Hints ---
A("## 7. Blocking Strategy Recommendations\n")
A("| ID | Strategy | Key Params | Derived From | Notes |")
A("|----|----------|------------|--------------|-------|")
for h in blocking_hints:
    params_str = ", ".join(f"{k}={v}" for k, v in list(h["params"].items())[:2]) if h["params"] else "—"
    rules_str = ", ".join(h["derived_from"])
    A(f"| **{h['id']}** | {h['strategy']} | {params_str} | {rules_str} | {h['recall_ceiling_note'][:80]} |")
A("")
A("> **Recommended blocking stack:** B03 (country partition) → B01 (name TF-IDF) ∪ B04 (address token) ∪ B05 (first-token exact)  ")
A("> Target: **recall ceiling > 90%** while reducing candidate pairs from 8.47T → <50M")
A("")

# --- Country flags ---
A("## 8. Per-Country Handling Flags\n")
for country, flags in country_flags.items():
    A(f"### {country}" + (" ⚠️ ZERO-SHOT" if flags["zero_shot"] else ""))
    A(f"- **Script:** {flags['dominant_script']}")
    A(f"- **Address format:** {flags['address_format_notes']}")
    if flags["special_handling"]:
        A(f"- **Special handling:** {', '.join(flags['special_handling'])}")
    if flags.get("warning"):
        A(f"- ⚠️ **Warning:** {flags['warning']}")
    A("")

# --- Global Flags ---
A("## 9. Global Pipeline Flags\n")
A("| Flag | Value | Consequence |")
A("|------|-------|-------------|")
A("| `pin_zip_usable_as_blocking_key` | ❌ False | 95% coverage gap — skip PIN-based blocking |")
A("| `safe_to_drop_null_address_records` | ❌ False | ~265K records have no address; must match on name |")
A("| `must_handle_zero_shot_countries` | ✅ True | France has no training examples — pipeline must generalise |")
A("| `evaluation_metric` | F_0.5 macro | Precision-heavy — false merges cost 2× more than missed links |")
A("| `total_test_s1_records_to_submit` | 1,732,544 | Every S1 record must appear in output — including singletons |")
A("")

# --- Output files ---
A("## 10. EDA Output Files\n")
A("| File | Description |")
A("|------|-------------|")
A("| `eda_output/noise_spec.json` | Machine-readable spec — import in normalisation + blocking scripts |")
A("| `eda_output/data_quality_report.md` | This document |")
A("| `eda_output/01_overview.csv` | Shape + null counts per source |")
A("| `eda_output/06_noise_patterns.csv` | Noise flag rates (%) per source |")
A("| `eda_output/11_length_stats.csv` | Length statistics (mean/median/std/p95) |")
A("| `eda_output/12_samples.csv` | 5 sample records per country per source |")
A("| `eda_output/plots/` | All 8 visualisation PNGs |")

report_path = OUT / "data_quality_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"✅ Written: {report_path}")

print("\n📁 Final eda_output/ structure:")
for p in sorted(OUT.rglob("*")):
    indent = "  " * (len(p.relative_to(OUT).parts) - 1)
    size = f"  ({p.stat().st_size:,} bytes)" if p.is_file() else "/"
    print(f"  {indent}{p.name}{size}")
