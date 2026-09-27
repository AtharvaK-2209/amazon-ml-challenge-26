"""
=============================================================
  Amazon ML Challenge 2026 — Phase 1: EDA on TEST DATA
  Sources: test_source1.tsv, test_source2.tsv, test_source3.tsv
=============================================================
"""

import os, re, unicodedata, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves PNGs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
BASE   = Path(__file__).parent / "student_resource" / "dataset" / "test"
OUT    = Path(__file__).parent / "eda_output"
OUT.mkdir(exist_ok=True)

FILES  = {
    "S1": BASE / "test_source1.tsv",
    "S2": BASE / "test_source2.tsv",
    "S3": BASE / "test_source3.tsv",
}

PALETTE = {"S1": "#4C6EF5", "S2": "#20C997", "S3": "#F76707"}
sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Helpers ────────────────────────────────────────────────
def detect_script(text: str) -> str:
    """Classify dominant Unicode script block in a string."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    scripts = Counter()
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
        except Exception:
            continue
        if "LATIN" in name:
            scripts["latin"] += 1
        elif "DEVANAGARI" in name:
            scripts["devanagari"] += 1
        elif "ARABIC" in name:
            scripts["arabic"] += 1
        elif ch.isalpha():
            scripts["other"] += 1
    if not scripts:
        return "non-alpha"
    return scripts.most_common(1)[0][0]


def token_freq(series: pd.Series, n: int = 30) -> pd.Series:
    tokens = []
    for val in series.dropna():
        tokens.extend(re.findall(r"[A-Za-z]+", str(val).lower()))
    return pd.Series(Counter(tokens)).nlargest(n)


def noise_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean noise-indicator columns."""
    df = df.copy()
    name = df["business_name"].fillna("").astype(str)
    addr = df["business_address"].fillna("").astype(str)

    df["name_has_abbrev"]    = name.str.contains(r"\b(pvt|ltd|llc|corp|inc|co|llp|plc)\b",
                                                   case=False, regex=True)
    df["name_has_ampersand"] = name.str.contains(r"&", regex=False)
    df["name_all_caps"]      = name.str.isupper() & (name.str.len() > 3)
    df["name_has_digit"]     = name.str.contains(r"\d", regex=True)
    df["addr_missing_pin"]   = ~addr.str.contains(r"\b\d{5,6}\b", regex=True)
    df["addr_has_landmark"]  = addr.str.contains(
        r"\b(near|opp|opposite|beside|behind|next to|adj)\b", case=False, regex=True)
    df["addr_has_abbrev"]    = addr.str.contains(
        r"\b(rd|st|ave|blvd|dr|ln|nagar|marg|vihar|enclave)\b", case=False, regex=True)
    return df


# ── Load data ──────────────────────────────────────────────
print("Loading test files …")
dfs = {}
for key, path in FILES.items():
    print(f"  Reading {path.name} …")
    dfs[key] = pd.read_csv(path, sep="\t", low_memory=False)
print("Done.\n")

# Add noise columns
for k in dfs:
    dfs[k] = noise_flags(dfs[k])
    dfs[k]["name_len"]   = dfs[k]["business_name"].fillna("").str.len()
    dfs[k]["addr_len"]   = dfs[k]["business_address"].fillna("").str.len()
    dfs[k]["name_tokens"]= dfs[k]["business_name"].fillna("").str.split().str.len()
    dfs[k]["addr_tokens"]= dfs[k]["business_address"].fillna("").str.split().str.len()
    dfs[k]["name_script"]= dfs[k]["business_name"].apply(detect_script)
    dfs[k]["source"]     = k

combined = pd.concat(dfs.values(), ignore_index=True)

# ══════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1 — DATASET OVERVIEW")
print("=" * 60)

overview_rows = []
for k, df in dfs.items():
    row = {
        "Source": k,
        "Rows": len(df),
        "Cols": df.shape[1],
        "Null business_name": df["business_name"].isna().sum(),
        "Null business_address": df["business_address"].isna().sum(),
        "Null country": df["country"].isna().sum(),
        "Unique countries": df["country"].nunique(),
        "Duplicate entity_id": df["entity_id"].duplicated().sum(),
    }
    overview_rows.append(row)

overview_df = pd.DataFrame(overview_rows)
print(overview_df.to_string(index=False))
overview_df.to_csv(OUT / "01_overview.csv", index=False)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — COUNTRY DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2 — COUNTRY DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (k, df) in zip(axes, dfs.items()):
    counts = df["country"].value_counts()
    counts.plot.bar(ax=ax, color=PALETTE[k], edgecolor="white", linewidth=0.8)
    ax.set_title(f"Source {k} — Country Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Record count")
    ax.tick_params(axis="x", rotation=30)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "02_country_distribution.png", dpi=150)
plt.close()
print("  → Plot saved: 02_country_distribution.png")

for k, df in dfs.items():
    print(f"\n  Source {k}:")
    print(df["country"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════
# SECTION 3 — NAME & ADDRESS LENGTH DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3 — NAME & ADDRESS LENGTH DISTRIBUTIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for col_idx, (k, df) in enumerate(dfs.items()):
    # Name length
    ax = axes[0, col_idx]
    ax.hist(df["name_len"].clip(0, 150), bins=60, color=PALETTE[k], edgecolor="white", alpha=0.85)
    ax.set_title(f"Source {k} — Business Name Length", fontsize=11, fontweight="bold")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    med = df["name_len"].median()
    ax.axvline(med, color="red", linestyle="--", linewidth=1.5, label=f"Median={med:.0f}")
    ax.legend(fontsize=9)

    # Address length
    ax2 = axes[1, col_idx]
    ax2.hist(df["addr_len"].clip(0, 300), bins=60, color=PALETTE[k], edgecolor="white", alpha=0.85)
    ax2.set_title(f"Source {k} — Business Address Length", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Characters")
    ax2.set_ylabel("Count")
    med2 = df["addr_len"].median()
    ax2.axvline(med2, color="red", linestyle="--", linewidth=1.5, label=f"Median={med2:.0f}")
    ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "03_length_distributions.png", dpi=150)
plt.close()
print("  → Plot saved: 03_length_distributions.png")

# Length stats
for k, df in dfs.items():
    print(f"\n  Source {k} — name_len stats:")
    print(df["name_len"].describe().round(1).to_string())
    print(f"  Source {k} — addr_len stats:")
    print(df["addr_len"].describe().round(1).to_string())

# ══════════════════════════════════════════════════════════════
# SECTION 4 — TOKEN COUNT DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4 — TOKEN COUNT DISTRIBUTIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for col_idx, (k, df) in enumerate(dfs.items()):
    for row_idx, (col, label) in enumerate([("name_tokens", "Name Tokens"), ("addr_tokens", "Address Tokens")]):
        ax = axes[row_idx, col_idx]
        clip_max = 20 if col == "name_tokens" else 40
        ax.hist(df[col].clip(0, clip_max), bins=clip_max, color=PALETTE[k], edgecolor="white", alpha=0.85)
        ax.set_title(f"Source {k} — {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Count")
        med = df[col].median()
        ax.axvline(med, color="red", linestyle="--", linewidth=1.5, label=f"Median={med:.0f}")
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "04_token_distributions.png", dpi=150)
plt.close()
print("  → Plot saved: 04_token_distributions.png")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — SCRIPT / LANGUAGE ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5 — SCRIPT / LANGUAGE ANALYSIS (Business Names)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (k, df) in zip(axes, dfs.items()):
    counts = df["name_script"].value_counts()
    counts.plot.bar(ax=ax, color=PALETTE[k], edgecolor="white")
    ax.set_title(f"Source {k} — Name Script Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "05_script_distribution.png", dpi=150)
plt.close()
print("  → Plot saved: 05_script_distribution.png")

for k, df in dfs.items():
    print(f"\n  Source {k}:")
    print(df["name_script"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════
# SECTION 6 — NOISE PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6 — NOISE PATTERN ANALYSIS")
print("=" * 60)

noise_cols = [
    "name_has_abbrev", "name_has_ampersand", "name_all_caps",
    "name_has_digit", "addr_missing_pin", "addr_has_landmark", "addr_has_abbrev"
]

noise_pcts = {}
for k, df in dfs.items():
    noise_pcts[k] = (df[noise_cols].mean() * 100).round(2)

noise_df = pd.DataFrame(noise_pcts, index=noise_cols)
print(noise_df.to_string())
noise_df.to_csv(OUT / "06_noise_patterns.csv")

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(noise_cols))
width = 0.28
for i, k in enumerate(["S1", "S2", "S3"]):
    ax.bar(x + i * width, noise_df[k], width, label=f"Source {k}",
           color=PALETTE[k], edgecolor="white", alpha=0.9)
ax.set_xticks(x + width)
ax.set_xticklabels(noise_cols, rotation=30, ha="right")
ax.set_ylabel("Percentage of records (%)")
ax.set_title("Noise Pattern Prevalence Across Sources", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "06_noise_patterns.png", dpi=150)
plt.close()
print("  → Plot saved: 06_noise_patterns.png")

# ══════════════════════════════════════════════════════════════
# SECTION 7 — TOP TOKENS (NAME & ADDRESS)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7 — TOP TOKENS IN NAMES & ADDRESSES")
print("=" * 60)

# STOPWORDS to ignore
STOPWORDS = {"the", "of", "and", "a", "an", "in", "at", "on", "for", "to",
             "s", "pvt", "ltd", "llc", "inc", "co", "corp", "no", "near",
             "road", "street", "st", "rd", "dr"}

fig, axes = plt.subplots(3, 2, figsize=(22, 18))
for row_idx, (k, df) in enumerate(dfs.items()):
    for col_idx, (col, label) in enumerate([("business_name", "Name"), ("business_address", "Address")]):
        ax = axes[row_idx, col_idx]
        tokens = []
        for val in df[col].dropna():
            toks = re.findall(r"[A-Za-z]{3,}", str(val).lower())
            tokens.extend([t for t in toks if t not in STOPWORDS])
        freq = pd.Series(Counter(tokens)).nlargest(25)
        freq[::-1].plot.barh(ax=ax, color=PALETTE[k], edgecolor="white", alpha=0.9)
        ax.set_title(f"Source {k} — Top 25 {label} Tokens", fontsize=11, fontweight="bold")
        ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig(OUT / "07_top_tokens.png", dpi=150)
plt.close()
print("  → Plot saved: 07_top_tokens.png")

# ══════════════════════════════════════════════════════════════
# SECTION 8 — MISSING VALUES HEATMAP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 8 — MISSING VALUES")
print("=" * 60)

core_cols = ["entity_id", "business_name", "business_address", "country"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (k, df) in zip(axes, dfs.items()):
    null_pct = df[core_cols].isna().mean() * 100
    ax.barh(core_cols, null_pct, color=PALETTE[k], edgecolor="white")
    ax.set_title(f"Source {k} — Null %", fontsize=11, fontweight="bold")
    ax.set_xlabel("% Null")
    ax.set_xlim(0, max(null_pct.max() + 5, 5))
    for i, v in enumerate(null_pct):
        ax.text(v + 0.2, i, f"{v:.2f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "08_missing_values.png", dpi=150)
plt.close()
print("  → Plot saved: 08_missing_values.png")

# ══════════════════════════════════════════════════════════════
# SECTION 9 — ENTITY ID PREFIX SANITY CHECK
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 9 — ENTITY ID PREFIX SANITY CHECK")
print("=" * 60)

for k, df in dfs.items():
    prefixes = df["entity_id"].str.extract(r"^([A-Z0-9]+)-", expand=False).value_counts()
    print(f"  Source {k} entity_id prefixes:\n{prefixes.to_string()}\n")

# ══════════════════════════════════════════════════════════════
# SECTION 10 — CROSS-SOURCE COUNTRY OVERLAP (combined view)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 10 — CROSS-SOURCE COUNTRY COMPOSITION (stacked)")
print("=" * 60)

country_pivot = pd.DataFrame({
    k: df["country"].value_counts() for k, df in dfs.items()
}).fillna(0).astype(int)

country_pct = country_pivot.div(country_pivot.sum(axis=0), axis=1) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Absolute counts
country_pivot.T.plot.bar(stacked=True, ax=axes[0], colormap="Set2", edgecolor="white", linewidth=0.6)
axes[0].set_title("Absolute Record Counts by Country & Source", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Source")
axes[0].set_ylabel("Records")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Country", bbox_to_anchor=(1.01, 1), loc="upper left")

# Percentage
country_pct.T.plot.bar(stacked=True, ax=axes[1], colormap="Set2", edgecolor="white", linewidth=0.6)
axes[1].set_title("Percentage Composition by Country & Source", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Source")
axes[1].set_ylabel("% of Records")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Country", bbox_to_anchor=(1.01, 1), loc="upper left")

plt.tight_layout()
plt.savefig(OUT / "10_cross_source_country.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Plot saved: 10_cross_source_country.png")
print("\nCountry pivot (absolute counts):")
print(country_pivot.to_string())

# ══════════════════════════════════════════════════════════════
# SECTION 11 — SUMMARY STATS TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 11 — LENGTH SUMMARY STATS TABLE")
print("=" * 60)

stats_rows = []
for k, df in dfs.items():
    for field in ["name_len", "addr_len", "name_tokens", "addr_tokens"]:
        s = df[field].describe()
        stats_rows.append({
            "source": k, "field": field,
            "mean": round(s["mean"], 1), "median": round(s["50%"], 1),
            "std": round(s["std"], 1), "min": int(s["min"]),
            "max": int(s["max"]), "p95": round(df[field].quantile(0.95), 1)
        })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))
stats_df.to_csv(OUT / "11_length_stats.csv", index=False)

# ══════════════════════════════════════════════════════════════
# SECTION 12 — SAMPLE RECORDS PER COUNTRY PER SOURCE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 12 — SAMPLE RECORDS (5 per country per source)")
print("=" * 60)

sample_rows = []
for k, df in dfs.items():
    for country in df["country"].dropna().unique():
        sub = df[df["country"] == country][["entity_id", "business_name", "business_address", "country"]]
        sample_rows.append(sub.head(5))

sample_df = pd.concat(sample_rows, ignore_index=True)
sample_df.to_csv(OUT / "12_samples.csv", index=False)
print(f"  Saved {len(sample_df)} sample rows to 12_samples.csv")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA COMPLETE — Output files written to:", OUT.resolve())
print("=" * 60)
outputs = sorted(OUT.glob("*"))
for o in outputs:
    print(f"  {o.name}  ({o.stat().st_size:,} bytes)")
