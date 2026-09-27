# Amazon ML Challenge 2026 - Entity Resolution Pipeline

## Problem Overview

This project solves an entity matching / record linkage problem across three business data sources:

**Source 1 (S1):**
- entity_id
- business_name
- business_address
- country

**Source 2 (S2):**
- entity_id
- business_name
- business_address
- country

**Source 3 (S3):**
- entity_id
- business_name
- business_address
- country

**Goal:** Identify which S2/S3 entities correspond to each S1 entity.

**Evaluation Metric:** Macro-averaged F0.5 at the Source-1 entity level. Precision is weighted more heavily than recall, so false merges are particularly costly.

## Planned Pipeline

```
Load Data
    ↓
Normalize Text
    ↓
Multi-Pass Blocking
    ↓
Pairwise Feature Engineering
    ↓
XGBoost Classification
    ↓
Probability Calibration
    ↓
Precision-First Decision Engine
    ↓
Entity-Level Aggregation
    ↓
Evaluation & Submission
```

### Pipeline Stages (To Be Implemented)

1. **Normalization** (`src/preprocessing/`)
   - Text standardization
   - Address parsing
   - Unicode normalization
   - Abbreviation expansion

2. **Blocking** (`src/blocking/`)
   - Exact match blocking
   - Token overlap blocking
   - TF-IDF similarity blocking
   - FAISS nearest neighbor blocking
   - Candidate generation only (no final decisions)

3. **Features** (`src/features/`)
   - Name similarity features (Jaccard, Levenshtein, token sort ratio, etc.)
   - Address similarity features (numeric match, street type, city match)
   - Cross-field features (country match, combined similarity)

4. **Models** (`src/models/`)
   - XGBoost binary classifier
   - Probability calibration (isotonic regression or Platt scaling)

5. **Decision Engine** (`src/decision/`)
   - Threshold-based matching (precision-first)
   - Margin logic for ambiguous cases
   - Singleton detection for no-match handling

6. **Evaluation** (`src/evaluation/`)
   - Entity-level F0.5 computation
   - Confusion matrix analysis
   - Per-entity result logging

## Project Structure

```
amazon-ml-challenge/
├── data/                    # Data storage (not committed to Git)
│   ├── train/
│   └── test/
├── src/                     # Main source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main orchestration
│   ├── preprocessing/      # Text normalization and address parsing
│   ├── blocking/           # Candidate generation strategies
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and prediction
│   ├── decision/           # Threshold and decision logic
│   └── evaluation/         # F0.5 evaluation
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── experiments/            # Experiment tracking and results
├── output/                 # Generated outputs
├── configs/                # Configuration files
│   └── config.yaml
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git exclusions
├── .env.example            # Environment variables template
└── PROJECT_RULES.md        # Competition constraints
```

## Local Setup

### Requirements

- Python 3.11+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd amazon-ml-challenge
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (optional for local development)
   ```

5. **Place data files:**
   - Training data in `data/train/`
   - Test data in `data/test/`
   - **Do not commit data files to Git**

### Running the Pipeline

```bash
# Run with default configuration
python -m src.pipeline

# Or run as module
python src/pipeline.py
```

## AWS Architecture (Future)

The project is designed to support AWS deployment:

```
Local Development
    ↓
S3 Data Storage
    ↓
SageMaker Training
    ↓
S3 Model Artifacts
    ↓
SageMaker Batch Transform
    ↓
S3 Outputs
```

**Phase 0:** Local development only - no AWS resources created or used.

**Future Phases:**
- S3 for data and model storage
- SageMaker for training and inference
- CloudWatch for logging
- IAM for access control

## Configuration

Configuration is managed via:
- `configs/config.yaml` - Main configuration file
- Environment variables - For secrets and deployment-specific settings
- `src/config.py` - Configuration loader and management

Key configuration sections:
- Data paths (local and S3)
- Blocking parameters
- Feature engineering settings
- XGBoost hyperparameters
- Decision thresholds
- Logging level

## Experiment Methodology

All experiments are tracked in `experiments/` directory. Each experiment should record:

- Experiment ID and date
- Code version / Git commit
- Preprocessing configuration
- Blocking configuration
- Feature configuration
- Model configuration
- Threshold and margin settings
- Validation F0.5, precision, recall
- Candidate recall
- Runtime
- Notes and observations

See `experiments/README.md` for detailed tracking guidelines.

## Git Workflow

### Branches

- `main` - Stable production code only
- `experiment/baseline` - Baseline model experiments
- `experiment/tfidf` - TF-IDF blocking experiments
- `experiment/hard-negative` - Hard negative mining experiments
- `experiment/threshold` - Threshold optimization experiments

### Workflow

1. Create feature branch from `main`
2. Implement and test changes
3. Submit pull request
4. Review and merge to `main`
5. Delete feature branch

**Important:** Never commit data, credentials, or generated outputs to Git.

## Competition Constraints

See `PROJECT_RULES.md` for complete list of competition constraints. Key rules:

1. **Data Usage:** Use only competition-provided data. No external business lookup.
2. **No External APIs:** No Google Maps, geocoding, or internet-based enrichment.
3. **Country Handling:** Countries are open-set. Do not hard-code for specific countries.
4. **Evaluation Metric:** Optimize for F0.5 (precision-weighted) at entity level.
5. **Precision First:** False merges are costly - prioritize precision.
6. **Singleton Handling:** Correct no-match handling is critical.
7. **Reproducibility:** All experiments must be reproducible.
8. **Security:** Never commit credentials or secrets.

## Current Status

**Phase 0 Complete:**
- [x] Project structure created
- [x] Configuration system implemented
- [x] Placeholder modules for all pipeline stages
- [x] Documentation and project rules
- [x] Git repository initialized
- [x] Testing infrastructure

**Phase 1 (Next Steps):**
- [ ] Implement text normalization
- [ ] Implement blocking strategies
- [ ] Implement feature engineering
- [ ] Implement XGBoost training
- [ ] Implement decision engine
- [ ] Implement evaluation
- [ ] Generate first submission

## License

This project is for the Amazon ML Challenge 2026 competition only.

## Team

- Lead ML Engineer: [Your Name]
- Team Members: [Team Member Names]

---

**Note:** This project is under active development. The ML pipeline components are not yet implemented - only the architectural framework exists in Phase 0.
