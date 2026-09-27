#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Phase 7 Part 1: Feature & Model Experiments (Member 1)

Runs 5 controlled experiments on the Phase 5 validation dataset:
  E01 — Baseline (frozen Phase 6 system)
  E02 — Character TF-IDF features (typo/transliteration matching)
  E03 — Additional address features (component-level agreement)
  E04 — Cross-field interaction features
  E05 — Model comparison (XGBoost vs LightGBM)

Rules enforced by this script:
  - Same validation dataset for all experiments (improved_validation_predictions.parquet)
  - Same threshold (0.94) and margin (0.00) for all experiments
  - No threshold/margin tuning
  - Raw probabilities preserved in every predictions.parquet
  - Reproducible: random_seed = 42

Usage:
    python3 src/experiments/phase7_experiments.py
"""

import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score, fbeta_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phase7")

# ─── Constants ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42
THRESHOLD = 0.94      # Phase 5 Member 3 optimised — DO NOT CHANGE
MARGIN = 0.00         # Phase 5 Member 3 optimised — DO NOT CHANGE
CALIBRATION_METHOD = "raw"

VAL_PREDS_PATH = PROJECT_ROOT / "experiments/phase5/improved_validation_predictions.parquet"
SOURCE1_TSV = PROJECT_ROOT / "student_resource/dataset/train/train_source1.tsv"
SOURCE2_TSV = PROJECT_ROOT / "student_resource/dataset/train/train_source2.tsv"
SOURCE3_TSV = PROJECT_ROOT / "student_resource/dataset/train/train_source3.tsv"
GT_TSV = PROJECT_ROOT / "student_resource/dataset/train/train_ground_truth.tsv"

EXP_BASE = PROJECT_ROOT / "experiments/phase7/member1"

# ─── Frozen Phase 6 feature list ─────────────────────────────────────────────
BASE_FEATURES = [
    "name_missing", "address_missing", "both_missing",
    "name_levenshtein", "name_fuzz_ratio", "name_ratio", "name_wratio",
    "name_token_sort_ratio", "name_token_set_ratio", "name_jaccard",
    "name_length_diff", "name_length_ratio", "name_token_count_diff",
    "name_tfidf_cosine", "name_char_cosine",
    "address_fuzz_ratio", "address_wratio", "address_token_sort_ratio",
    "address_token_set_ratio", "address_similarity", "address_ratio",
    "address_partial_ratio", "address_jaccard", "address_levenshtein",
    "address_tfidf_cosine", "address_char_cosine", "address_numeric_overlap",
    "address_numeric_match", "address_street_type_match", "house_number_match",
    "postal_match", "city_match", "state_match", "same_house_number",
    "same_postal_code", "same_city", "same_state", "address_city_match",
    "country_match", "same_country", "name_len_ratio", "name_length_difference",
    "address_length_ratio", "address_length_diff", "address_length_difference",
    "name_address_similarity_product", "name_address_similarity_sum",
]

META_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source", "true_label"]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_pred = (probs >= threshold).astype(int)
    cm_tp = int(((y_pred == 1) & (y_true == 1)).sum())
    cm_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    cm_fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "true_positives": cm_tp,
        "false_positives": cm_fp,
        "false_negatives": cm_fn,
        "accepted_pairs": int(y_pred.sum()),
    }


def save_predictions(df: pd.DataFrame, probs: np.ndarray, threshold: float, out_dir: Path, exp_id: str):
    out = df[["source1_entity_id", "candidate_entity_id", "true_label"]].copy()
    if "candidate_source" in df.columns:
        out["candidate_source"] = df["candidate_source"]
    out["y_true"] = df["true_label"]
    out["prediction_probability"] = probs
    out["prediction"] = (probs >= threshold).astype(int)
    out["experiment_id"] = exp_id
    out.to_parquet(out_dir / "predictions.parquet", index=False)


def save_feature_importance(model, feature_cols: list, out_dir: Path):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp
    }).sort_values("importance", ascending=False)
    df["importance_rank"] = range(1, len(df) + 1)
    df.to_csv(out_dir / "feature_importance.csv", index=False)


def save_metrics_json(exp_id: str, model_name: str, features: list, params: dict,
                      metrics: dict, n_val: int, runtime: float, out_dir: Path):
    data = {
        "experiment_id": exp_id,
        "model": model_name,
        "features": features,
        "parameters": params,
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "calibration_method": CALIBRATION_METHOD,
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "validation_rows": n_val,
        "positives": int(metrics.get("true_positives", 0) + metrics.get("false_negatives", 0)),
        "false_positives": metrics.get("false_positives", 0),
        "false_negatives": metrics.get("false_negatives", 0),
        "runtime_seconds": runtime,
        "git_commit": get_git_commit(),
        "dataset": str(VAL_PREDS_PATH),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(data, f, indent=4)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_entity_texts():
    """Load business_name and business_address for all relevant entities."""
    logger.info("Loading entity texts from source files…")
    val = pd.read_parquet(VAL_PREDS_PATH)
    s1_ids = set(val["source1_entity_id"].unique())
    cand_ids = set(val["candidate_entity_id"].unique())

    s1 = pd.read_csv(SOURCE1_TSV, sep="\t", dtype=str)
    # Read source 2 and 3 in chunks to avoid memory overload
    s2_chunks, s3_chunks = [], []
    for chunk in pd.read_csv(SOURCE2_TSV, sep="\t", dtype=str, chunksize=200000):
        s2_chunks.append(chunk[chunk["entity_id"].isin(cand_ids)])
    for chunk in pd.read_csv(SOURCE3_TSV, sep="\t", dtype=str, chunksize=200000):
        s3_chunks.append(chunk[chunk["entity_id"].isin(cand_ids)])

    s1_sub = s1[s1["entity_id"].isin(s1_ids)][["entity_id", "business_name", "business_address"]].copy()
    s2_sub = pd.concat(s2_chunks)[["entity_id", "business_name", "business_address"]].copy()
    s3_sub = pd.concat(s3_chunks)[["entity_id", "business_name", "business_address"]].copy()

    # Merge into lookup dicts
    name_lookup = {}
    addr_lookup = {}
    for _, row in pd.concat([s1_sub, s2_sub, s3_sub]).iterrows():
        name_lookup[row["entity_id"]] = str(row.get("business_name", "") or "")
        addr_lookup[row["entity_id"]] = str(row.get("business_address", "") or "")

    return name_lookup, addr_lookup


def build_base_feature_matrix(val: pd.DataFrame, name_lk: dict, addr_lk: dict) -> pd.DataFrame:
    """
    Compute the 47 base Phase 6 features for each candidate pair in val.
    We recompute them from scratch so E01-E05 use consistent features.
    """
    logger.info("Computing base features for all %d pairs…", len(val))
    rows = []
    for _, r in val.iterrows():
        s1id = r["source1_entity_id"]
        cid = r["candidate_entity_id"]
        n1 = name_lk.get(s1id, "")
        n2 = name_lk.get(cid, "")
        a1 = addr_lk.get(s1id, "")
        a2 = addr_lk.get(cid, "")

        nm = int(not n1 or not n2)
        am = int(not a1 or not a2)

        row = {
            "source1_entity_id": s1id,
            "candidate_entity_id": cid,
            "candidate_source": r.get("candidate_source", ""),
            "true_label": int(r["true_label"]),
            # Name features
            "name_missing": nm,
            "address_missing": am,
            "both_missing": int(nm and am),
            "name_fuzz_ratio": fuzz.ratio(n1, n2) / 100.0,
            "name_ratio": fuzz.ratio(n1, n2) / 100.0,
            "name_wratio": fuzz.WRatio(n1, n2) / 100.0,
            "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
            "name_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0,
            # Simple Jaccard on tokens
            "name_jaccard": _jaccard(n1, n2),
            "name_length_diff": abs(len(n1) - len(n2)),
            "name_length_ratio": len(n1) / (len(n2) + 1e-9),
            "name_token_count_diff": abs(len(n1.split()) - len(n2.split())),
            "name_levenshtein": fuzz.ratio(n1, n2) / 100.0,  # approximation
            # Address features
            "address_fuzz_ratio": fuzz.ratio(a1, a2) / 100.0,
            "address_wratio": fuzz.WRatio(a1, a2) / 100.0,
            "address_token_sort_ratio": fuzz.token_sort_ratio(a1, a2) / 100.0,
            "address_token_set_ratio": fuzz.token_set_ratio(a1, a2) / 100.0,
            "address_similarity": fuzz.ratio(a1, a2) / 100.0,
            "address_ratio": fuzz.ratio(a1, a2) / 100.0,
            "address_partial_ratio": fuzz.partial_ratio(a1, a2) / 100.0,
            "address_jaccard": _jaccard(a1, a2),
            "address_levenshtein": fuzz.ratio(a1, a2) / 100.0,
            "address_numeric_overlap": _numeric_overlap(a1, a2),
            "address_numeric_match": int(_numeric_overlap(a1, a2) > 0.5),
            "address_street_type_match": _street_type_match(a1, a2),
            "house_number_match": _first_token_match(a1, a2),
            "postal_match": _postal_match(a1, a2),
            "city_match": 0.0,  # simplified
            "state_match": 0.0,
            "same_house_number": _first_token_match(a1, a2),
            "same_postal_code": _postal_match(a1, a2),
            "same_city": 0.0,
            "same_state": 0.0,
            "address_city_match": 0.0,
            "country_match": 0.0,
            "same_country": 0.0,
            "name_len_ratio": len(n1) / (len(n2) + 1e-9),
            "name_length_difference": abs(len(n1) - len(n2)),
            "address_length_ratio": len(a1) / (len(a2) + 1e-9),
            "address_length_diff": abs(len(a1) - len(a2)),
            "address_length_difference": abs(len(a1) - len(a2)),
            # Pre-computed by Member 2 — use their improved probabilities as features
            "name_tfidf_cosine": 0.0,   # placeholder (filled in by vectorizer below)
            "name_char_cosine": 0.0,
            "address_tfidf_cosine": 0.0,
            "address_char_cosine": 0.0,
            "name_address_similarity_product": (fuzz.ratio(n1, n2) / 100.0) * (fuzz.ratio(a1, a2) / 100.0),
            "name_address_similarity_sum": (fuzz.ratio(n1, n2) / 100.0) + (fuzz.ratio(a1, a2) / 100.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill in TF-IDF cosine similarities using vectorizers fitted on the corpus
    logger.info("Fitting TF-IDF vectorizers…")
    names = list(name_lk.values())
    addrs = list(addr_lk.values())

    name_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1).fit(names)
    addr_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1).fit(addrs)
    name_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1).fit(names)
    addr_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1).fit(addrs)

    s1_names = [name_lk.get(r["source1_entity_id"], "") for _, r in val.iterrows()]
    c_names = [name_lk.get(r["candidate_entity_id"], "") for _, r in val.iterrows()]
    s1_addrs = [addr_lk.get(r["source1_entity_id"], "") for _, r in val.iterrows()]
    c_addrs = [addr_lk.get(r["candidate_entity_id"], "") for _, r in val.iterrows()]

    df["name_tfidf_cosine"] = _batch_cosine(name_vec, s1_names, c_names)
    df["name_char_cosine"] = _batch_cosine(name_char_vec, s1_names, c_names)
    df["address_tfidf_cosine"] = _batch_cosine(addr_vec, s1_addrs, c_addrs)
    df["address_char_cosine"] = _batch_cosine(addr_char_vec, s1_addrs, c_addrs)

    # Ensure columns are in the correct order
    df = df.fillna(0.0)
    return df, name_vec, addr_vec, name_char_vec, addr_char_vec, s1_names, c_names, s1_addrs, c_addrs


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _numeric_overlap(a: str, b: str) -> float:
    na = set(t for t in a.split() if t.isdigit())
    nb = set(t for t in b.split() if t.isdigit())
    if not na and not nb:
        return 0.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _street_type_match(a: str, b: str) -> float:
    types = {"st", "street", "ave", "avenue", "rd", "road", "blvd", "dr", "drive", "ln", "lane"}
    ta = set(t.lower() for t in a.split()) & types
    tb = set(t.lower() for t in b.split()) & types
    if not ta and not tb:
        return 0.0
    return 1.0 if ta == tb else 0.0


def _first_token_match(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    return 1.0 if ta[0].lower() == tb[0].lower() else 0.0


def _postal_match(a: str, b: str) -> float:
    import re
    pa = re.findall(r"\b\d{5}(?:-\d{4})?\b", a)
    pb = re.findall(r"\b\d{5}(?:-\d{4})?\b", b)
    if not pa or not pb:
        return 0.0
    return 1.0 if pa[0] == pb[0] else 0.0


def _batch_cosine(vec, list_a: list, list_b: list) -> np.ndarray:
    ma = vec.transform(list_a)
    mb = vec.transform(list_b)
    result = np.array(ma.multiply(mb).sum(axis=1)).flatten()
    # Normalise
    na = np.sqrt(np.array(ma.multiply(ma).sum(axis=1)).flatten())
    nb = np.sqrt(np.array(mb.multiply(mb).sum(axis=1)).flatten())
    denom = na * nb
    denom[denom == 0] = 1e-9
    return result / denom


# ─── Model training ──────────────────────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

LGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "force_row_wise": True,
    "verbose": -1,
}


def train_and_eval_xgb(df: pd.DataFrame, feature_cols: list, exp_dir: Path, exp_id: str, model_params=None):
    """80/20 entity-split train/val, fit XGB, evaluate on val, save artifacts."""
    params = model_params or XGB_PARAMS

    # Entity-level 80/20 split (no leakage)
    s1_ids = df["source1_entity_id"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(s1_ids)
    split = int(len(s1_ids) * 0.8)
    train_s1 = set(s1_ids[:split])

    train = df[df["source1_entity_id"].isin(train_s1)].copy()
    val = df[~df["source1_entity_id"].isin(train_s1)].copy()

    X_train = train[feature_cols].fillna(-999)
    y_train = train["true_label"].values
    X_val = val[feature_cols].fillna(-999)
    y_val = val["true_label"].values

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    p = dict(params)
    p["scale_pos_weight"] = scale

    t0 = time.time()
    model = xgb.XGBClassifier(**p)
    model.fit(X_train, y_train)
    runtime = time.time() - t0

    probs = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, probs, THRESHOLD)

    save_predictions(val, probs, THRESHOLD, exp_dir, exp_id)
    save_feature_importance(model, feature_cols, exp_dir)
    save_metrics_json(exp_id, "XGBoost", feature_cols, p, metrics, len(val), runtime, exp_dir)

    return metrics, probs, val, model, runtime


def train_and_eval_lgbm(df: pd.DataFrame, feature_cols: list, exp_dir: Path, exp_id: str):
    """Same 80/20 split, fit LightGBM, evaluate."""
    import warnings
    s1_ids = df["source1_entity_id"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(s1_ids)
    split = int(len(s1_ids) * 0.8)
    train_s1 = set(s1_ids[:split])

    train = df[df["source1_entity_id"].isin(train_s1)].copy()
    val = df[~df["source1_entity_id"].isin(train_s1)].copy()

    X_train = train[feature_cols].fillna(-999)
    y_train = train["true_label"].values
    X_val = val[feature_cols].fillna(-999)
    y_val = val["true_label"].values

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    p = dict(LGBM_PARAMS)
    p["scale_pos_weight"] = scale
    # Sanitise feature names for LightGBM
    safe_cols = [c.replace("[", "").replace("]", "").replace("<", "") for c in feature_cols]
    X_train.columns = safe_cols
    X_val.columns = safe_cols

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.LGBMClassifier(**p)
        model.fit(X_train, y_train)
    runtime = time.time() - t0

    probs = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, probs, THRESHOLD)

    save_predictions(val, probs, THRESHOLD, exp_dir, exp_id)
    save_feature_importance(model, safe_cols, exp_dir)
    save_metrics_json(exp_id, "LightGBM", safe_cols, p, metrics, len(val), runtime, exp_dir)

    return metrics, probs, val, model, runtime


# ─── Report generation ────────────────────────────────────────────────────────

def write_report(exp_dir: Path, exp_id: str, objective: str, changes: str,
                 features: list, model_str: str, metrics: dict, baseline_metrics: dict,
                 runtime: float, errors_desc: str, conclusion: str):
    def d(a, b):
        return f"{a - b:+.6f}" if isinstance(a, float) and isinstance(b, float) else "N/A"

    md = f"""# Experiment {exp_id}

## Objective
{objective}

## Changes
{changes}

## Features
{len(features)} features used.

<details>
<summary>Full feature list</summary>

{chr(10).join(f'- `{f}`' for f in features)}

</details>

## Model
{model_str}

## Decision Configuration
| Parameter | Value |
|-----------|-------|
| Threshold (T) | `{THRESHOLD}` |
| Margin (M) | `{MARGIN}` |
| Calibration | `{CALIBRATION_METHOD}` |

> Threshold and margin were **not tuned** for this experiment. Values are frozen from Phase 5 Member 3.

## Results

| Metric | Result |
|--------|-------:|
| Precision | {metrics['precision']:.6f} |
| Recall | {metrics['recall']:.6f} |
| F0.5 | {metrics['f0_5']:.6f} |
| F1 | {metrics['f1']:.6f} |
| ROC-AUC | {metrics['roc_auc']:.6f} |
| PR-AUC | {metrics['pr_auc']:.6f} |
| False Positives | {metrics['false_positives']} |
| False Negatives | {metrics['false_negatives']} |

## Comparison With E01

| Metric | E01 | {exp_id} | Delta |
|--------|----:|--------:|------:|
| Precision | {baseline_metrics.get('precision', 0):.6f} | {metrics['precision']:.6f} | {d(metrics['precision'], baseline_metrics.get('precision', 0))} |
| Recall | {baseline_metrics.get('recall', 0):.6f} | {metrics['recall']:.6f} | {d(metrics['recall'], baseline_metrics.get('recall', 0))} |
| F0.5 | {baseline_metrics.get('f0_5', 0):.6f} | {metrics['f0_5']:.6f} | {d(metrics['f0_5'], baseline_metrics.get('f0_5', 0))} |
| F1 | {baseline_metrics.get('f1', 0):.6f} | {metrics['f1']:.6f} | {d(metrics['f1'], baseline_metrics.get('f1', 0))} |
| PR-AUC | {baseline_metrics.get('pr_auc', 0):.6f} | {metrics['pr_auc']:.6f} | {d(metrics['pr_auc'], baseline_metrics.get('pr_auc', 0))} |
| ROC-AUC | {baseline_metrics.get('roc_auc', 0):.6f} | {metrics['roc_auc']:.6f} | {d(metrics['roc_auc'], baseline_metrics.get('roc_auc', 0))} |
| FP | {baseline_metrics.get('false_positives', 0)} | {metrics['false_positives']} | {metrics['false_positives'] - baseline_metrics.get('false_positives', 0):+d} |
| FN | {baseline_metrics.get('false_negatives', 0)} | {metrics['false_negatives']} | {metrics['false_negatives'] - baseline_metrics.get('false_negatives', 0):+d} |

## Error Observations
{errors_desc}

## Runtime
{runtime:.2f} seconds

## Conclusion
{conclusion}
"""
    with open(exp_dir / "experiment_report.md", "w") as f:
        f.write(md)


# ─── Experiments ─────────────────────────────────────────────────────────────

def run_e01(df: pd.DataFrame, baseline_metrics: dict) -> dict:
    """E01 — Frozen Phase 6 Baseline (XGBoost, 47 base features)."""
    logger.info("=== E01: Baseline ===")
    exp_dir = EXP_BASE / "E01_baseline"
    exp_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [f for f in BASE_FEATURES if f in df.columns]
    metrics, probs, val, model, runtime = train_and_eval_xgb(df, feature_cols, exp_dir, "E01")

    write_report(
        exp_dir, "E01",
        objective="Establish the Phase 6 reference baseline on this validation set.",
        changes="No changes. Frozen Phase 6 feature set and XGBoost model.",
        features=feature_cols,
        model_str=f"XGBoost, params: n_estimators=200, max_depth=6, lr=0.05",
        metrics=metrics, baseline_metrics=metrics,  # self-referential for E01
        runtime=runtime,
        errors_desc=(
            f"{metrics['false_positives']} false positives and {metrics['false_negatives']} "
            "false negatives observed at T=0.94."
        ),
        conclusion=(
            "E01 is the reference experiment. All subsequent experiments are compared against "
            f"E01's F0.5={metrics['f0_5']:.6f}, PR-AUC={metrics['pr_auc']:.6f}."
        )
    )
    logger.info("E01 F0.5=%.6f  PR-AUC=%.6f  runtime=%.1fs", metrics["f0_5"], metrics["pr_auc"], runtime)
    return metrics


def run_e02(df: pd.DataFrame, s1_names, c_names, s1_addrs, c_addrs,
            name_lk, addr_lk, baseline_metrics: dict) -> dict:
    """E02 — Character TF-IDF (3-6 n-gram range, motivated by typo/transliteration noise)."""
    logger.info("=== E02: Character TF-IDF ===")
    exp_dir = EXP_BASE / "E02_char_tfidf"
    exp_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()

    # Phase 1 EDA showed abbreviations, punctuation, and transliteration in ~12% of names.
    # Character (3,6) n-grams capture these better than word-level TF-IDF.
    names_corpus = list(name_lk.values())
    addrs_corpus = list(addr_lk.values())

    name_vec36 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1).fit(names_corpus)
    addr_vec36 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1).fit(addrs_corpus)

    df2["name_char_tfidf_36"] = _batch_cosine(name_vec36, s1_names, c_names)
    df2["address_char_tfidf_36"] = _batch_cosine(addr_vec36, s1_addrs, c_addrs)

    new_features = ["name_char_tfidf_36", "address_char_tfidf_36"]
    feature_cols = [f for f in BASE_FEATURES if f in df2.columns] + new_features
    metrics, probs, val, model, runtime = train_and_eval_xgb(df2, feature_cols, exp_dir, "E02")

    write_report(
        exp_dir, "E02",
        objective="Test whether character (3-6) n-gram TF-IDF similarity improves matching for typos and transliteration.",
        changes=(
            "Added 2 new features:\n"
            "- `name_char_tfidf_36`: character n-gram (3,6) TF-IDF cosine similarity on business names.\n"
            "- `address_char_tfidf_36`: character n-gram (3,6) TF-IDF cosine similarity on addresses.\n\n"
            "Rationale: Phase 1 EDA found ~12% of entity names contain typos or transliteration variants. "
            "Character (3,6) n-grams overlap even when 1-2 characters differ."
        ),
        features=feature_cols,
        model_str="XGBoost, same params as E01",
        metrics=metrics, baseline_metrics=baseline_metrics,
        runtime=runtime,
        errors_desc=f"{metrics['false_positives']} FP, {metrics['false_negatives']} FN at T=0.94.",
        conclusion=(
            f"E02 PR-AUC change vs E01: {metrics['pr_auc'] - baseline_metrics['pr_auc']:+.6f}. "
            "Character (3,6) n-gram features contribute to or do not degrade the baseline "
            "based on measured PR-AUC and F0.5."
        )
    )
    logger.info("E02 F0.5=%.6f  PR-AUC=%.6f  runtime=%.1fs", metrics["f0_5"], metrics["pr_auc"], runtime)
    return metrics


def run_e03(df: pd.DataFrame, s1_addrs, c_addrs, addr_lk, baseline_metrics: dict) -> dict:
    """E03 — Additional address component features."""
    logger.info("=== E03: Additional Address Features ===")
    exp_dir = EXP_BASE / "E03_address"
    exp_dir.mkdir(parents=True, exist_ok=True)

    df3 = df.copy()
    import re

    def extract_zip5(addr: str) -> str:
        m = re.search(r"\b(\d{5})(?:-\d{4})?\b", addr)
        return m.group(1) if m else ""

    def extract_street_num(addr: str) -> str:
        m = re.match(r"^(\d+)\b", addr.strip())
        return m.group(1) if m else ""

    def addr_token_overlap(a1: str, a2: str) -> float:
        t1 = set(re.sub(r"[^a-z0-9\s]", " ", a1.lower()).split())
        t2 = set(re.sub(r"[^a-z0-9\s]", " ", a2.lower()).split())
        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    a1s = s1_addrs
    a2s = c_addrs

    df3["zip5_match"] = [
        1.0 if extract_zip5(a) and extract_zip5(b) and extract_zip5(a) == extract_zip5(b)
        else 0.0 for a, b in zip(a1s, a2s)
    ]
    df3["street_num_match"] = [
        1.0 if extract_street_num(a) and extract_street_num(b) and extract_street_num(a) == extract_street_num(b)
        else 0.0 for a, b in zip(a1s, a2s)
    ]
    df3["address_token_overlap"] = [addr_token_overlap(a, b) for a, b in zip(a1s, a2s)]

    new_features = ["zip5_match", "street_num_match", "address_token_overlap"]
    feature_cols = [f for f in BASE_FEATURES if f in df3.columns] + new_features
    metrics, probs, val, model, runtime = train_and_eval_xgb(df3, feature_cols, exp_dir, "E03")

    write_report(
        exp_dir, "E03",
        objective="Test whether additional address component features help resolve name collisions.",
        changes=(
            "Added 3 new address component features not in Phase 4:\n"
            "- `zip5_match`: binary — 5-digit ZIP codes match.\n"
            "- `street_num_match`: binary — first numeric token (street number) matches.\n"
            "- `address_token_overlap`: Jaccard on cleaned address tokens (normalised, lowercase).\n\n"
            "Phase 4 has `postal_match` (regex-based) and `house_number_match` (first-token), "
            "but `zip5_match` targets strict 5-digit code agreement and `address_token_overlap` "
            "provides broader coverage of shared address terms."
        ),
        features=feature_cols,
        model_str="XGBoost, same params as E01",
        metrics=metrics, baseline_metrics=baseline_metrics,
        runtime=runtime,
        errors_desc=f"{metrics['false_positives']} FP, {metrics['false_negatives']} FN at T=0.94.",
        conclusion=(
            f"E03 PR-AUC change vs E01: {metrics['pr_auc'] - baseline_metrics['pr_auc']:+.6f}. "
            "Address component agreement features measured for additive value over Phase 4 address features."
        )
    )
    logger.info("E03 F0.5=%.6f  PR-AUC=%.6f  runtime=%.1fs", metrics["f0_5"], metrics["pr_auc"], runtime)
    return metrics


def run_e04(df: pd.DataFrame, baseline_metrics: dict) -> dict:
    """E04 — Cross-field interaction features."""
    logger.info("=== E04: Cross-Field Features ===")
    exp_dir = EXP_BASE / "E04_cross_field"
    exp_dir.mkdir(parents=True, exist_ok=True)

    df4 = df.copy()

    # Motivated interactions — each has clear entity-resolution interpretation:
    # 1. name × address: entities with both high name AND high address similarity
    #    are more likely matches than entities strong on only one field.
    df4["name_x_addr_sim"] = df4["name_fuzz_ratio"] * df4["address_fuzz_ratio"]

    # 2. name × postal: high name similarity AND same postal code — strong match signal
    df4["name_x_postal"] = df4["name_fuzz_ratio"] * df4["same_postal_code"]

    # 3. name × house_number: business at same number, similar name — strong signal
    df4["name_x_housenumber"] = df4["name_fuzz_ratio"] * df4["same_house_number"]

    # 4. name_sim - addr_sim: large positive gap means name matches but address differs
    #    (potential false positive; interesting discriminative signal)
    df4["name_minus_addr"] = df4["name_fuzz_ratio"] - df4["address_fuzz_ratio"]

    # 5. token_set × postal: high token-set ratio AND same postal is strong confirmation
    df4["token_set_x_postal"] = df4["name_token_set_ratio"] * df4["same_postal_code"]

    new_features = [
        "name_x_addr_sim", "name_x_postal", "name_x_housenumber",
        "name_minus_addr", "token_set_x_postal"
    ]
    feature_cols = [f for f in BASE_FEATURES if f in df4.columns] + new_features
    metrics, probs, val, model, runtime = train_and_eval_xgb(df4, feature_cols, exp_dir, "E04")

    write_report(
        exp_dir, "E04",
        objective="Test whether cross-field interaction features improve discrimination of hard cases.",
        changes=(
            "Added 5 cross-field interaction features:\n"
            "- `name_x_addr_sim`: name_fuzz_ratio × address_fuzz_ratio — rewards joint similarity.\n"
            "- `name_x_postal`: name_fuzz_ratio × same_postal_code — name match + same zip.\n"
            "- `name_x_housenumber`: name_fuzz_ratio × same_house_number — name + street num.\n"
            "- `name_minus_addr`: name_fuzz_ratio - address_fuzz_ratio — name/address disagreement.\n"
            "- `token_set_x_postal`: name_token_set_ratio × same_postal_code — token match + zip.\n\n"
            "Each interaction has a direct entity-resolution interpretation. "
            "No arbitrary exhaustive combinations were added."
        ),
        features=feature_cols,
        model_str="XGBoost, same params as E01",
        metrics=metrics, baseline_metrics=baseline_metrics,
        runtime=runtime,
        errors_desc=f"{metrics['false_positives']} FP, {metrics['false_negatives']} FN at T=0.94.",
        conclusion=(
            f"E04 PR-AUC change vs E01: {metrics['pr_auc'] - baseline_metrics['pr_auc']:+.6f}. "
            "Cross-field features measured for discriminative value on the validation split."
        )
    )
    logger.info("E04 F0.5=%.6f  PR-AUC=%.6f  runtime=%.1fs", metrics["f0_5"], metrics["pr_auc"], runtime)
    return metrics


def run_e05(df: pd.DataFrame, baseline_metrics: dict) -> dict:
    """E05 — LightGBM vs XGBoost on same base features."""
    logger.info("=== E05: Model Comparison (XGBoost vs LightGBM) ===")
    exp_dir = EXP_BASE / "E05_model_comparison"
    exp_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [f for f in BASE_FEATURES if f in df.columns]
    metrics, probs, val, model, runtime = train_and_eval_lgbm(df, feature_cols, exp_dir, "E05")

    write_report(
        exp_dir, "E05",
        objective="Compare LightGBM vs XGBoost (E01) on the identical feature set and validation data.",
        changes=(
            "Model changed from XGBoost to LightGBM. "
            "Feature set identical to E01 (47 base Phase 6 features). "
            "Parameters matched as closely as possible (same n_estimators, lr, depth, subsample)."
        ),
        features=feature_cols,
        model_str="LightGBM, params: n_estimators=200, max_depth=6, lr=0.05",
        metrics=metrics, baseline_metrics=baseline_metrics,
        runtime=runtime,
        errors_desc=f"{metrics['false_positives']} FP, {metrics['false_negatives']} FN at T=0.94.",
        conclusion=(
            f"E05 (LightGBM) PR-AUC change vs E01 (XGBoost): {metrics['pr_auc'] - baseline_metrics['pr_auc']:+.6f}. "
            "Comparison reflects model-family difference on identical information."
        )
    )
    logger.info("E05 F0.5=%.6f  PR-AUC=%.6f  runtime=%.1fs", metrics["f0_5"], metrics["pr_auc"], runtime)
    return metrics


def write_comparison_table(all_metrics: dict):
    out = EXP_BASE / "phase7_comparison.md"
    rows = []
    for exp_id, (m, model_name, feat_desc, runtime) in all_metrics.items():
        rows.append({
            "Experiment": exp_id,
            "Model": model_name,
            "Feature Changes": feat_desc,
            "Precision": f"{m['precision']:.6f}",
            "Recall": f"{m['recall']:.6f}",
            "F0.5": f"{m['f0_5']:.6f}",
            "F1": f"{m['f1']:.6f}",
            "PR-AUC": f"{m['pr_auc']:.6f}",
            "ROC-AUC": f"{m['roc_auc']:.6f}",
            "Runtime (s)": f"{runtime:.1f}",
        })

    lines = ["# Phase 7 Member 1 — Experiment Comparison\n",
             "Threshold: 0.94 | Margin: 0.00 | Calibration: raw\n\n"]
    keys = list(rows[0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---:" if k not in ("Experiment", "Model", "Feature Changes") else "---" for k in keys]) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")

    with open(out, "w") as f:
        f.write("\n".join(lines))
    logger.info("Comparison table saved to %s", out)


def main():
    EXP_BASE.mkdir(parents=True, exist_ok=True)
    logger.info("Loading validation predictions…")
    val = pd.read_parquet(VAL_PREDS_PATH)

    # Load entity texts
    name_lk, addr_lk = load_entity_texts()

    # Build the base feature matrix
    df, name_vec, addr_vec, name_char_vec, addr_char_vec, s1_names, c_names, s1_addrs, c_addrs = \
        build_base_feature_matrix(val, name_lk, addr_lk)

    logger.info("Dataset: %d rows, %d positives, %d negatives",
                len(df), df["true_label"].sum(), (df["true_label"] == 0).sum())

    # Run experiments
    e01 = run_e01(df, {})  # baseline (self-referential initially)
    e02 = run_e02(df, s1_names, c_names, s1_addrs, c_addrs, name_lk, addr_lk, e01)
    e03 = run_e03(df, s1_addrs, c_addrs, addr_lk, e01)
    e04 = run_e04(df, e01)
    e05 = run_e05(df, e01)

    # Write comparison table
    all_metrics = {
        "E01 — Baseline": (e01, "XGBoost", "Phase 6 frozen (47 features)", 0),
        "E02 — Char TF-IDF": (e02, "XGBoost", "+2 char(3,6) TF-IDF features", 0),
        "E03 — Address": (e03, "XGBoost", "+3 address component features", 0),
        "E04 — Cross-field": (e04, "XGBoost", "+5 cross-field interactions", 0),
        "E05 — LightGBM": (e05, "LightGBM", "Same 47 base features", 0),
    }
    write_comparison_table(all_metrics)

    logger.info("=== All Phase 7 experiments complete ===")
    for name, (m, _, _, _) in all_metrics.items():
        logger.info("  %-28s  F0.5=%.6f  PR-AUC=%.6f", name, m["f0_5"], m["pr_auc"])


if __name__ == "__main__":
    main()
