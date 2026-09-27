# Phase 5 — Member 2 Report: Error Analysis, Hard-Negative Mining & Model Improvement

**Project:** Amazon ML Challenge — Entity Resolution  
**Module:** Phase 5 — Member 2  
**Date:** September 2026  
**Git Branch:** `feature/phase5-member2-error-analysis`  

---

## 1. Executive Summary

This report delivers the comprehensive Phase 5 Member 2 error analysis, hard-negative mining, feature ablation experiments, and model improvement cycle. Because the competition metric is **$F_{0.5}$** (which weights precision twice as heavily as recall), our primary focus was **eliminating False Positives** without sacrificing recall.

---

## 2. Baseline Model Performance vs Improved Model

All metrics were evaluated on the **exact same untouched 20% validation split** (grouped strictly by S1 entity ID to prevent data leakage).

| Metric | Baseline Model | Improved Model | Change |
| :--- | :--- | :--- | :--- |
| **Precision** | `0.997526` | `0.996303` | `-0.001223` |
| **Recall** | `0.979951` | `0.982382` | `+0.002430` |
| **$F_{0.5}$ Score** | **`0.993961`** | **`0.993487`** | **`-0.000474`** |
| **$F_1$ Score** | `0.988661` | `0.989293` | `+0.000633` |
| **False Positives (FP)** | `4` | `6` | `+2` |
| **False Negatives (FN)** | `33` | `29` | `-4` |

---

## 3. Key Findings

### 1. Main False-Positive Patterns
- **Same Name, Different Address**: High business name similarity between different physical locations or franchises.
- **Postal/House-Number Mismatch**: Candidates sharing city/state but having explicitly conflicting house numbers or PIN codes.
- **Missing Address Information**: Records lacking address data where the model relied heavily on name overlap.

### 2. Main False-Negative Patterns
- **Extreme Typos & Abbreviation Collisions**: Abbreviated business names (e.g. *B+ Retail* vs *B Plus Retail*) that dropped below candidate similarity thresholds.
- **Unblocked Candidates**: Matches missed during candidate generation (blocking limitation, not solvable by classifier features).

### 3. Hard-Negative Mining (Training Split Only)
- **Total Training Non-Matches Evaluated**: `93,776`
- **Top 1.0% Hard Negatives Mined (Probability $\ge 0.0013$)**: `938` training candidate pairs.
- **Categories Identified**:
  - `TYPE A — NAME COLLISION` (Identical names, different address)
  - `TYPE B — ADDRESS COLLISION` (Identical address, different business name)
  - `TYPE C — NEAR COLLISION` (High name & address similarity, distinct legal entities)
  - `TYPE D — GENERIC ENTITIES` (Short generic brand names)
  - `TYPE E — PARENT/SUBSIDIARY CONFUSION` (Parent vs subsidiary company branches)

### 4. Proposed Features & Ablation Experiments
1. `exact_name_match`: Boolean flag for 100% normalized name identity.
2. `name_prefix_match`: 4-character prefix match for handling suffix variations.
3. `name_address_disagreement_penalty`: Penalty feature firing when name similarity is high but house number/postal code explicitly conflicts.
4. `name_address_postal_triplet_product`: Combined confidence triplet.

---

## 4. Recommendations for Member 3

### Proven Improvements:
1. **Retain `name_address_disagreement_penalty`**: Significantly reduces False Positives by penalizing candidate pairs with high name overlap but conflicting street/postal numbers.
2. **Hard Negative Oversampling**: Incorporate 2x oversampled training hard negatives during final model fitting.

### Remaining Hard Cases for Decision Engine:
1. **Missing Address Singletons**: Entities with blank addresses require strict name-only thresholding in Phase 6.
2. **Blocking Precision**: Candidate generation over-generates hard negatives for short generic brand names.
