# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Atharva  
**Team Members:** Member 1, Member 2, Member 3  
**Submission Date:** September 2026  

---

## 1. Executive Summary
Our solution uses a precision-first multi-stage Entity Resolution pipeline combining country-aware text normalization, multi-blocker candidate generation (TF-IDF N-gram cosine similarity, exact key matching, and sorted neighbourhood blocking), a 47-feature pairwise gradient boosted classifier (XGBoost) trained with location-collision hard negatives, and an $F_{0.5}$-optimized decision engine ($T=0.94, M=0.00$). On validation, the pipeline achieves an $F_{0.5}$ score of **`0.994790`** with 100% Precision (`0` False Positives) and 100% Singleton Accuracy.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights from EDA:
- **Location Collisions (Type F Errors)**: Business entities sharing identical postal codes, cities, or street names but representing distinct physical businesses.
- **Multilingual / Multi-Country Scope**: Entity datasets span US, India, and France. Legal suffixes (e.g., `Inc`, `Corp`, `Sarl`, `Pvt Ltd`) create artificial string variance.
- **Precision Weighting ($F_{0.5}$)**: The competition metric $F_{0.5}$ heavily penalizes False Positives (wrong merges). Thresholding at $T=0.94$ eliminates all false merges ($FP=0$).

### 2.2 Solution Strategy

**Approach Type:** Multi-Stage Blocker + Pairwise Feature Extraction + Gradient Boosted Classifier + Decision Engine  
**Core Innovation:** Location collision hard-negative mining (10% ratio) combined with high-precision thresholding ($T=0.94$) and pure model probability ranking.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Character N-gram TF-IDF Cosine Similarity (k=50)
  2. Exact normalized name / address key matching
  3. Sorted Neighbourhood Token Blocking
- **Candidate pairs generated:** 27,977 candidate pairs across 500 test S1 entities (mean: 50.15 candidates per S1).
- **How you ensured true matches were not lost:** Multi-blocker union achieved a Blocking Recall of **`94.00%`** (1,646 of 1,751 ground truth pairs).

---

## 4. Matching Model

**Features used:**
- **Name features**: Jaccard, Levenshtein distance, RapidFuzz token sort/set ratio, TF-IDF cosine similarity, character n-gram cosine similarity, length ratio, token count difference.
- **Address features**: Street number match, house number equality, postal code match, city/state match, partial ratio, token overlap, numeric overlap.
- **Cross-field features**: Name * Address similarity product, Name + Address similarity sum.

**Model type:** XGBoost Classifier (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`)  
**Threshold selection method:** Grid search on validation set maximizing $F_{0.5}$ macro score ($T=0.94$, $M=0.00$).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** `0.994790` (Validation set: 25,074 candidate pairs, 500 S1 entities). Precision: `1.000000`, Recall: `0.974484`.
- **Common false positives (wrong merges):** Zero false positives ($FP=0$) at operating threshold $T=0.94$.
- **Common false negatives (missed matches):** Severe name truncations/abbreviations and candidates unblocked during Phase 3 candidate generation (105 unblocked pairs).

---

## 6. Conclusion
Our solution demonstrates that combining multi-blocker candidate generation with XGBoost and precision-first decision rules ($T=0.94$) produces optimal entity resolution performance ($F_{0.5} = 0.994790$) without introducing false merges.

---

## Appendix

### A. Code Artefacts
Complete, self-contained runnable code is located in `code/business_entity_resolution/` (source in `src/`, entry point in `src/phase8/pipeline.py`, with `README.md` and `requirements.txt`).

To reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python3 -m src.phase8.pipeline --config phase8/final_config.json --test-dir dataset/test --output-dir output
```

### B. Additional Results
Validation Metrics Summary:
- Precision: `1.000000`
- Recall: `0.974484`
- $F_{0.5}$ Score: `0.994790`
- Singleton Accuracy: `100.0%`
- Match Rate: `92.8%` (464 S1 entities matched out of 500)
