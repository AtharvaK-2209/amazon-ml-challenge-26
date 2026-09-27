# Business Entity Resolution — Amazon ML Challenge 2026 Code Package

This self-contained code package implements the production entity resolution pipeline for the Amazon ML Challenge 2026.

---

## 1. Project Overview
The objective is to map Source 1 reference/master entities (`S1`) to matching Source 2 (`S2`) and/or Source 3 (`S3`) candidate entities across US, India, and France.

The pipeline comprises:
1. **Normalization**: Country-specific legal suffix stripping, Unicode cleaning, whitespace normalization.
2. **Candidate Blocking**: Multi-blocker combining TF-IDF N-gram cosine similarity, exact key matching, and sorted neighbourhood blocking.
3. **Pairwise Feature Engineering**: 47 pairwise similarity features (string distances, token overlap, numeric/address element matching, cross-field interactions).
4. **ML Model**: XGBoost Classifier trained with controlled location collision hard-negative sampling.
5. **Probability Calibration**: `raw` (Raw model output probabilities).
6. **Decision Engine**: Precision-first thresholding ($T = 0.94$), margin checking ($M = 0.00$), and pure probability rank order.

---

## 2. Directory Structure
```text
code/business_entity_resolution/
├── src/
│   ├── preprocessing/       # Normalization & address parser
│   ├── blocking/            # Candidate blocking modules
│   ├── features/            # Feature extraction & matrix construction
│   ├── models/              # XGBoost training & inference
│   ├── decision/            # Decision engine & thresholding
│   ├── evaluation/          # F0.5 macro & pair metrics
│   ├── phase6/              # Integration wrappers & validation gate
│   ├── phase7/              # Experimentation & decision control
│   └── phase8/              # Master production submission pipeline
├── README.md                # Submission & reproduction documentation
└── requirements.txt         # Dependencies
```

---

## 3. Environment Setup & Dependencies
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Running Full Inference & Submission Generation
To run the complete production pipeline and generate submission files:
```bash
python3 -m src.phase8.pipeline --config phase8/final_config.json --test-dir dataset/test --output-dir output
```

Output files generated:
- `output/matching_results.tsv`: Leaderboard submission matching results (1 row per test S1 entity).
- `output/candidate_pairs.tsv`: Final candidate set fed into inference (1 row per test S1 entity).

---

## 5. Validating Submission Files
To run the official submission validator:
```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
