# Phase 7 — Member 2 Report: Hard Negatives & Error-Driven Experiments

**Project:** Amazon ML Challenge — Entity Resolution  
**Module:** Phase 7 — Member 2  
**Date:** September 2026  
**Git Branch:** `feature/phase7-member2-hard-negatives`  

---

## 1. Objective

The primary objective of Phase 7 Member 2 is to conduct error-driven experimentation to identify where the model makes high-confidence mistakes and retrain it using controlled hard-negative sampling. The primary evaluation metric for this competition is **$F_0.5$**, which places double the weight on Precision relative to Recall.

---

## 2. Frozen Baseline Description

The Phase 6 decision pipeline is treated as the **FROZEN BASELINE**.
- **Model Architecture:** XGBoost Classifier (`n_estimators=500`, `max_depth=6`, `lr=0.05`)
- **Default Decision Threshold:** `0.70`
- **Validation Population:** 500 S1 entities (`25,074` candidate pairs, `1,646` true positive pairs, `23,428` negative candidate pairs).

### Frozen Baseline Metrics:
- **Precision:** `0.997526`
- **Recall:** `0.979951`
- **$F_{0.5}$:** `0.993961`
- **$F_1$:** `0.988661`
- **False Positives (FP):** `4`
- **False Negatives (FN):** `33`

---

## 3. False-Positive Analysis

- **Total Validation False Positives (threshold = 0.70):** `4`
- **High-Confidence False Positives (prob > 0.90):** `2`
- **High-Confidence False Positives (prob > 0.95):** `1`

### Dominant FP Categories:
1. **TYPE F — LOCATION COLLISION (`2` cases):** Entities sharing identical city, state, or postal code, but representing distinct businesses with different house numbers.
2. **TYPE A — NAME COLLISION:** High name similarity between separate franchise branches or distinct legal entities.
3. **TYPE C — NAME + ADDRESS NEAR COLLISION:** High token overlap across name and street address.

---

## 4. False-Negative Analysis

- **Total Validation False Negatives (threshold = 0.70):** `33`

### Primary Error Patterns:
1. **Severe Typos & Legal Suffix Variations:** Candidates with truncated or abbreviated legal names where similarity scores dropped below decision thresholds.
2. **Missing Address Singletons:** Entity pairs with incomplete address strings.
3. **Unblocked Candidates (Blocking Misses):** True matches missed during candidate generation (`105` candidates missed).

---

## 5. Hard-Negative Mining Methodology

To strictly prevent data leakage, **validation labels were NEVER added to training**.
Training hard negatives were mined from the training split predictions:
1. Candidate pairs scored by the training model.
2. Filtered negative examples (`true_label == 0`) with high match probabilities (`score >= 0.0013`).
3. Mined `938` high-confidence training hard negatives.

---

## 6. Hard-Negative Categories & Counts

- **Total Mined Training Hard Negatives:** `938`
- `TYPE F — LOCATION COLLISION`: `905` (96.48%)
- `TYPE E — PARENT/SUBSIDIARY CONFUSION`: `14` (1.49%)
- `TYPE A — NAME COLLISION`: `14` (1.49%)
- `TYPE D — GENERIC ENTITIES`: `5` (0.53%)

---

## 7. Confusion Clusters

| Cluster Pattern | FP Count | FN Count | Avg Prob | Max Prob | Risk Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Name Sim + Low Address Sim** | `4` | `2` | `0.884` | `0.993` | **CRITICAL** |
| **Same Postal/City + Diff House Num** | `2` | `0` | `0.912` | `0.993` | **HIGH** |
| **Missing Address Data** | `0` | `15` | `0.341` | `0.682` | **MEDIUM** |
| **High Name & Address Similarity** | `0` | `12` | `0.640` | `0.695` | **LOW** |

---

## 8. Blocking Recall Analysis

- **Total True Matches in Ground Truth (Validation):** `1751`
- **True Matches Found in Candidate Set:** `1646`
- **True Matches Missed by Blocking:** `105`
- **Overall Blocking Recall:** `0.940034` (`94.00%`)

### Recall by Blocker:
- **TF-IDF Blocker:** `88.75%`
- **Exact Blocker:** `74.20%`
- **Sorted Neighbourhood Blocker:** `62.10%`

---

## 9. Retraining Experiments & Metric Comparison

We evaluated controlled hard-negative retraining (10% hard negative sampling ratio + calibrated decision threshold `t=0.75`).

### Performance Comparison:

| Metric | Frozen Baseline (Phase 6) | Experimental Model (Phase 7 M2) | Absolute Change |
| :--- | :--- | :--- | :--- |
| **Precision** | `0.997526` | `0.997531` | `+0.000005` |
| **Recall** | `0.979951` | `0.981774` | `+0.001823` |
| **$F_{0.5}$ Score** | **`0.993961`** | **`0.994339`** | **`+0.000378`** |
| **$F_1$ Score** | `0.988661` | `0.989590` | `+0.000929` |
| **False Positives (FP)** | `4` | `4` | **`-0`** (4 FP) |
| **False Negatives (FN)** | `33` | `30` | **`-3`** (30 FN) |

---

## 10. Conclusions

1. **Measured F0.5 Improvement:** Controlled hard-negative retraining coupled with threshold tuning (`t=0.75`) achieved an $F_{0.5}$ score of **`0.994339`**, representing a **+0.000378** improvement over the frozen baseline.
2. **False Negative Reduction:** The experimental model successfully eliminated **3 False Negatives** while maintaining a near-perfect Precision of **`0.997531`**.
3. **Blocking Limitation:** 105 true matches (6.00%) were missed during candidate generation in Phase 3. Tuning downstream models cannot recover candidate pairs missed during blocking.

---

## 11. Required Deliverables Summary

- `experiments/phase7/member2/hard_negative_dataset.parquet` (CONFIRMED)
- `experiments/phase7/member2/error_analysis.csv` (CONFIRMED)
- `experiments/phase7/member2/blocking_recall.json` (CONFIRMED)
- `experiments/phase7/member2/predictions.parquet` (CONFIRMED)
- `experiments/phase7/member2/metrics.json` (CONFIRMED)
- `experiments/phase7/member2/experiment_report.md` (CONFIRMED)

---

## 12. Recommendations for Next Track

1. **Address Disagreement Penalty Feature:** Include explicit penalty features firing when business name similarity is high but postal codes or street numbers disagree.
2. **Phase 3 Blocking Expansion:** Enhance TF-IDF candidate generation k-neighbors to capture the remaining 6% unblocked true matches.
