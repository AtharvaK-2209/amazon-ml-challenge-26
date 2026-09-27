# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Structure

```
amazon_ml/
├── data/                          # symlink or copy of student_resource/dataset/
│   ├── train/
│   └── test/
│
├── src/
│   ├── config.py                  # all paths, hyperparams, norm/blocking/model config
│   │
│   ├── preprocessing/
│   │   ├── normalize.py           # Phase 2: name + address normalisation engine
│   │   └── address_parser.py      # structured address component extraction
│   │
│   ├── blocking/
│   │   ├── exact_blocking.py      # B05: first-token exact match
│   │   ├── token_blocking.py      # B02: sorted neighbourhood
│   │   ├── tfidf_blocking.py      # B01: TF-IDF char-ngram (primary)
│   │   └── faiss_blocking.py      # B01 variant: dense ANN (SageMaker)
│   │
│   ├── features/
│   │   ├── name_features.py       # Phase 4: RapidFuzz + Jaccard name features
│   │   ├── address_features.py    # Phase 4: address similarity features
│   │   └── pair_features.py       # Phase 4: full pairwise feature assembly
│   │
│   ├── models/
│   │   ├── train_xgb.py           # Phase 5: XGBoost + hard negative mining
│   │   ├── predict.py             # Phase 5/6: inference + thresholding
│   │   └── saved/                 # model artifacts (joblib)
│   │
│   ├── decision/
│   │   ├── threshold.py           # Phase 6: threshold sweep
│   │   ├── margin.py              # Phase 6: margin-based precision filter
│   │   └── singleton.py           # Phase 6: singleton filling for submission
│   │
│   ├── evaluation/
│   │   └── evaluate_f05.py        # F_0.5 macro-averaged evaluator
│   │
│   └── pipeline.py                # end-to-end orchestrator
│
├── notebooks/                     # Jupyter / Colab EDA notebooks
├── experiments/                   # exp001_baseline/, exp002_char_tfidf/, …
├── output/                        # matching_results.tsv + candidate_pairs.tsv
├── configs/                       # YAML experiment configs
├── eda_output/
│   ├── noise_spec.json            # 🔑 machine-readable noise spec (drives Phase 2+3)
│   ├── data_quality_report.md     # human-readable Phase 1 report
│   ├── *.csv                      # EDA summary tables
│   └── plots/                     # all EDA visualisations
│
├── eda_test_data.py               # Phase 1: EDA script
├── generate_noise_spec.py         # Phase 1: noise spec generator
├── requirements.txt
└── README.md
```

## Phase Map

| Phase | Name | Key Output |
|-------|------|-----------|
| 0 | AWS + Project Setup | this structure |
| 1 | EDA + Noise Discovery | `eda_output/noise_spec.json` |
| 2 | Normalisation Engine | normalised DataFrames |
| 3 | High-Recall Multi-Pass Blocking | `output/candidate_pairs.tsv` |
| 4 | Pairwise Feature Engineering | `features/*.parquet` |
| 5 | XGBoost + Hard Negatives | `src/models/saved/model_vN.joblib` |
| 6 | Precision-First Decision Engine | `output/matching_results.tsv` |
| 7 | Experimentation War Room | `experiments/` |
| 8 | Final Pipeline + Submission | submission zip |

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA on test data (Phase 1)
python eda_test_data.py
python generate_noise_spec.py

# Run full test pipeline (Phase 2–6)
python -m src.pipeline --split test --threshold 0.70

# Validate output
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

## Evaluation Metric

**F_0.5 macro-averaged** (precision-heavy):
```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```
False merges cost 2× more than missed links. Correctly predicting singletons (no-match entities) earns full credit.
