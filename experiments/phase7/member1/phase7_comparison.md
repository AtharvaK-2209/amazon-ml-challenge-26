# Phase 7 Member 1 — Experiment Comparison

Threshold: 0.94 | Margin: 0.00 | Calibration: raw


| Experiment | Model | Feature Changes | Precision | Recall | F0.5 | F1 | PR-AUC | ROC-AUC | Runtime (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E01 — Baseline | XGBoost | Phase 6 frozen (47 features) | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999796 | 0.999985 | 0.0 |
| E02 — Char TF-IDF | XGBoost | +2 char(3,6) TF-IDF features | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999780 | 0.999983 | 0.0 |
| E03 — Address | XGBoost | +3 address component features | 1.000000 | 0.988439 | 0.997666 | 0.994186 | 0.999722 | 0.999978 | 0.0 |
| E04 — Cross-field | XGBoost | +5 cross-field interactions | 0.997085 | 0.988439 | 0.995343 | 0.992743 | 0.999826 | 0.999987 | 0.0 |
| E05 — LightGBM | LightGBM | Same 47 base features | 0.997093 | 0.991329 | 0.995935 | 0.994203 | 0.999596 | 0.999969 | 0.0 |