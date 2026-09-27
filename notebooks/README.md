# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, visualization, and quick experiments.

## Purpose

Notebooks are for:
- **Exploratory Data Analysis (EDA):** Understanding data distributions, patterns, and quality issues
- **Visualization:** Creating plots and charts for analysis and presentation
- **Quick Experiments:** Prototyping ideas before implementing in production code
- **Debugging:** Interactive debugging of pipeline components
- **Analysis:** Post-hoc analysis of results and errors

## Notebook Policy

**Important:** The final production pipeline must live under `src/`. Do not put core production logic exclusively inside notebooks.

### Guidelines

1. **Prototype in notebooks, production in `src/`:**
   - Use notebooks for exploration and prototyping
   - Move validated logic to proper modules in `src/`
   - Do not create critical functionality that only exists in notebooks

2. **Notebook naming convention:**
   - Use descriptive names: `01_data_exploration.ipynb`
   - Number sequentially for ordering
   - Include purpose in filename: `02_blocking_analysis.ipynb`

3. **Keep notebooks focused:**
   - One notebook per analysis task
   - Do not create monolithic "everything" notebooks
   - Clear outputs before committing (reduce file size)

4. **Document findings:**
   - Use markdown cells to explain analysis
   - Document insights and decisions
   - Reference notebooks in experiment logs

5. **Reproducibility:**
   - Set random seeds in notebooks
   - Document assumptions
   - Note data versions used

## Example Notebook Structure

```
# Title: Data Exploration - Business Names

## 1. Introduction
- Purpose of this analysis
- Questions to answer

## 2. Data Loading
- Load data
- Basic statistics

## 3. Analysis
- Distribution analysis
- Pattern discovery
- Quality assessment

## 4. Insights
- Key findings
- Implications for pipeline

## 5. Next Steps
- Actions to take
- Decisions made
```

## Notebook Best Practices

1. **Clear outputs before committing:**
   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

2. **Use relative imports from src:**
   ```python
   import sys
   sys.path.append('..')
   from src.config import get_config
   ```

3. **Set random seeds:**
   ```python
   import numpy as np
   import random
   np.random.seed(42)
   random.seed(42)
   ```

4. **Keep notebooks version-controlled:**
   - Git can track notebooks
   - But they should be small (clear outputs)
   - Consider using jupytext for text-based notebook format

## Sample Notebooks

The following notebooks will be created in Phase 1:

- `01_data_exploration.ipynb` - Explore entity distributions, missing values, patterns
- `02_country_analysis.ipynb` - Analyze country distributions and handling
- `03_blocking_analysis.ipynb` - Experiment with blocking strategies
- `04_feature_engineering.ipynb` - Develop and test feature engineering
- `05_model_development.ipynb` - XGBoost hyperparameter tuning
- `06_error_analysis.ipynb` - Analyze prediction errors

---

**Remember:** Notebooks are for humans. Production code is for machines. Keep the production pipeline in `src/`.
