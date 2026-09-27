# Phase 7 Final Report — Model & Feature Selection

## Executive Summary

Phase 7 evaluated 5 controlled experiments across candidate feature extensions and model families on the 5-fold GroupKFold validation dataset (25,074 candidate pairs, 1,646 ground truth matches across 500 S1 entities).

The conclusion of Phase 7 is that the frozen **XGBoost Baseline (`models/xgb_baseline.json`)** using the **47 Phase 6 pairwise features** with **Raw Calibration**, **Threshold $T = 0.94$**, and **Margin $M = 0.00$** remains the selected production configuration.

---

## Authoritative Selected Configuration

| Parameter | Value |
|---|---|
| **Model** | XGBoost |
| **Model Version** | `xgb_baseline` |
| **Model Artifact** | `models/xgb_baseline.json` |
| **Feature Set** | Phase 6 frozen pairwise feature set (47 features) |
| **Feature Count** | 47 |
| **Blocking** | Multi-attribute TF-IDF and candidate indexing |
| **Calibration Method** | `raw` |
| **Calibration Artifact** | `None` (Raw probability is identity transform) |
| **Threshold ($T$)** | `0.94` |
| **Margin ($M$)** | `0.00` |
| **Singleton Rule** | `assign_highest_above_threshold` |
| **Candidate Count Rule** | `process_all_candidates` |
| **S2 / S3 Handling** | `prioritize_higher_probability_source` |

---

## Phase 7 Experiment Comparison Summary

| Experiment | Model | Feature Count | Precision | Recall | $F_{0.5}$ | $F_1$ | PR-AUC | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| **E01 — Baseline** | XGBoost | 47 | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999796 | **SELECTED** |
| **E02 — Char TF-IDF** | XGBoost | 49 | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999780 | Rejected |
| **E03 — Address** | XGBoost | 50 | 1.000000 | 0.988439 | 0.997666 | 0.994186 | 0.999722 | Experimental |
| **E04 — Cross-field** | XGBoost | 52 | 0.997085 | 0.988439 | 0.995343 | 0.992743 | 0.999826 | Rejected |
| **E05 — LightGBM** | LightGBM | 47 | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999596 | Rejected |

---

## Verification & Handoff Directive

Phase 8 Part 1 (Member 1) is instructed to freeze and verify `models/xgb_baseline.json` against the 47 training feature schema without retraining or altering threshold, margin, or calibration rules.
