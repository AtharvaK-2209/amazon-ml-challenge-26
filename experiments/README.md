# Experiments Directory

This directory tracks all experiments for the entity resolution pipeline. Every experiment must be documented for reproducibility and comparison.

## Experiment Tracking Template

For each experiment, create a subdirectory or markdown file documenting:

### Required Information

```
## Experiment: [EXPERIMENT_ID]

**Date:** YYYY-MM-DD
**Git Commit:** [commit hash]
**Experimenter:** [Name]

### Configuration

**Preprocessing:**
- Normalization settings
- Address parsing settings

**Blocking:**
- Blocking strategies used
- TF-IDF parameters
- FAISS parameters

**Features:**
- Feature set used
- Feature engineering parameters

**Model:**
- XGBoost hyperparameters
- Calibration method

**Decision:**
- Match threshold
- Margin threshold
- Singleton handling

### Results

**Validation Metrics:**
- F0.5 score
- Precision
- Recall
- Candidate recall

**Runtime:**
- Total pipeline time
- Blocking time
- Feature extraction time
- Training time
- Prediction time

### Notes

- Key observations
- Issues encountered
- Insights gained

### Next Steps

- What to try next
- Changes to make
```

## Experiment Organization

### Directory Structure

```
experiments/
├── README.md (this file)
├── exp_001_baseline.md
├── exp_002_tfidf_blocking/
│   ├── config.yaml
│   ├── results.md
│   └── analysis.ipynb
├── exp_003_feature_engineering/
│   └── ...
└── leaderboard/
    └── submissions.csv
```

### Naming Convention

- **Format:** `exp_XXX_descriptive_name`
- **Example:** `exp_001_baseline`, `exp_015_hard_negative_mining`

## Experiment Log

Track all experiments in chronological order:

| Exp ID | Date | Description | Val F0.5 | Notes |
|--------|------|-------------|----------|-------|
| exp_001 | 2026-09-28 | Baseline model | TBD | Initial setup |
| ... | ... | ... | ... | ... |

## Comparison Methodology

When comparing experiments, consider:

### Primary Metrics
- **F0.5 score** (competition metric)
- **Precision** (more important than recall)
- **Candidate recall** (are we missing true matches in blocking?)

### Secondary Metrics
- Training time
- Prediction time
- Model size
- Feature count

### Statistical Significance
- Use cross-validation for reliable estimates
- Report mean ± std for metrics
- Consider significance tests for comparisons

## Hyperparameter Search

Document all hyperparameter searches:

### Grid Search
- Parameters explored
- Search space
- Best parameters found

### Random Search
- Number of trials
- Search distributions
- Best parameters found

### Bayesian Optimization
- Tool used (e.g., Optuna)
- Number of trials
- Acquisition function
- Best parameters found

## Ablation Studies

Track component contributions:

| Component Removed | F0.5 | Precision | Recall | Δ F0.5 |
|-------------------|------|-----------|--------|--------|
| Full model | 0.85 | 0.90 | 0.75 | - |
| Without FAISS blocking | - | - | - | - |
| Without address features | - | - | - | - |
| Without calibration | - | - | - | - |

## Leaderboard Tracking

Track competition submissions:

| Submission | Date | Public F0.5 | Private F0.5 | Experiment ID | Notes |
|------------|------|-------------|--------------|---------------|-------|
| 001 | 2026-09-30 | - | - | exp_001 | Baseline |
| ... | ... | ... | ... | ... | ... |

## Best Practices

### Before Running Experiment

1. **Create experiment directory:**
   ```bash
   mkdir experiments/exp_XXX_description
   ```

2. **Save configuration:**
   ```bash
   cp configs/config.yaml experiments/exp_XXX_description/config.yaml
   ```

3. **Record Git commit:**
   ```bash
   git rev-parse HEAD > experiments/exp_XXX_description/commit.txt
   ```

### During Experiment

1. **Log all outputs:**
   ```bash
   python -m src.pipeline 2>&1 | tee experiments/exp_XXX_description/log.txt
   ```

2. **Save intermediate results:**
   - Candidate pairs
   - Feature importance
   - Confusion matrix

### After Experiment

1. **Document results in markdown file**
2. **Create analysis notebook if needed**
3. **Update experiment comparison table**
4. **Archive model artifacts**

## Reproducibility Checklist

Before claiming an experiment is reproducible:

- [ ] Configuration saved
- [ ] Git commit recorded
- [ ] Random seed documented
- [ ] Environment specified
- [ ] Results documented
- [ ] Can re-run and get same results

---

**Remember:** Every experiment should contribute to understanding. Even failed experiments provide value if properly documented.
