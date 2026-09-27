# Project Rules - Amazon ML Challenge 2026

This document outlines all competition constraints and engineering conventions for the entity resolution project.

---

## Competition Requirements

The following rules are mandated by the competition and **must not be violated**:

### 1. Data Usage Constraints

**Rule:** Use only the competition-provided training and test data.

- **No external business lookup databases** (e.g., Crunchbase, LinkedIn, Yelp)
- **No external entity databases** (e.g., OpenCorporates, Wikidata)
- **No purchasing or acquiring third-party business data**

**Rationale:** Ensures fair competition and tests pure ML capabilities.

### 2. External API Restrictions

**Rule:** Do not use external APIs for entity enrichment.

- **No Google Maps API**
- **No geocoding APIs** (e.g., Nominatim, MapBox, Here)
- **No reverse geocoding**
- **No address validation APIs**

**Rationale:** Prevents data enrichment that wouldn't be available in real-world scenarios without cost/API access.

### 3. Internet-Based Enrichment

**Rule:** Do not use internet-based entity enrichment.

- **No web scraping** of business websites
- **No search engine queries** for business verification
- **No social media lookup**
- **No WHOIS domain lookups**

**Rationale:** Competition evaluates pure ML-based entity resolution, not data augmentation.

### 4. Country Handling

**Rule:** Country must remain open-set.

- **Do not hard-code assumptions for only US/India**
- **Test data may contain countries beyond training distribution**
- **Do not create country-specific rules or dictionaries**
- **Handle unknown countries gracefully**

**Rationale:** Real-world entity resolution must work across diverse geographies.

### 5. Validation Methodology

**Rule:** Validation must be performed at Source-1 entity level.

- **Do not optimize for pair-level metrics**
- **Do not optimize for accuracy** (use F0.5)
- **Every S1 entity must be evaluated equally**

**Rationale:** Competition metric is macro-averaged F0.5 at entity level.

### 6. Evaluation Metric

**Rule:** Optimize for F0.5, not accuracy or pair-level F1.

- **F0.5 weights precision more heavily than recall**
- **False merges are particularly costly**
- **Missing a match is less costly than incorrect merge**

**Implications:**
- Use precision-first thresholds (higher thresholds)
- Implement margin logic to avoid ambiguous matches
- Carefully handle borderline cases

### 7. Match Coverage

**Rule:** Every S1 entity must appear exactly once in final output.

- **No duplicate S1 entities**
- **No missing S1 entities**
- **Correctly handle singletons** (S1 entities with no match)

**Implications:**
- Must track all S1 entities through pipeline
- Must handle no-match cases explicitly
- Final output validation is mandatory

### 8. Candidate Pair Constraint

**Rule:** Candidate matches must be a subset of candidate_pairs.

- **Final matches cannot appear outside candidate set**
- **Blocking must generate all potential matches**
- **False negatives in blocking cannot be recovered**

**Implications:**
- Blocking recall is critical
- Prefer over-generating candidates over under-generating
- Validate candidate coverage

### 9. Reproducibility

**Rule:** Every experiment must be reproducible.

- **Record all random seeds**
- **Version control all code changes**
- **Document all hyperparameters**
- **Track Git commit for each experiment**

**Implications:**
- Use fixed random seeds
- Log all configuration
- Save experiment metadata

### 10. Submission Validation

**Rule:** Every submission must pass the competition validator.

- **Validate format before submission**
- **Check for missing/duplicate entities**
- **Verify file structure and encoding**

**Implications:**
- Implement pre-submission validation
- Test submission format early

---

## Engineering Conventions

The following rules are engineering best practices for this project:

### 1. No Hard-Coded Paths

**Convention:** Use configuration files and pathlib.Path for all paths.

- **Do not hard-code absolute paths**
- **Use config.yaml for path configuration**
- **Use pathlib.Path for cross-platform compatibility**

**Rationale:** Enables portability and AWS deployment.

### 2. No Credentials in Code

**Convention:** Never commit credentials or secrets.

- **No AWS keys in code**
- **No API keys in code**
- **Use environment variables or AWS credential files**
- **.env must be in .gitignore**

**Rationale:** Security and compliance.

### 3. Modular Design

**Convention:** Each pipeline stage must remain independent.

- **Preprocessing:** Text normalization only, no match decisions
- **Blocking:** Candidate generation only, no final decisions
- **Features:** Feature extraction only, no thresholding
- **Models:** Prediction only, no business logic
- **Decision:** Threshold and aggregation logic only

**Rationale:** Enables independent testing and debugging.

### 4. Configuration-Driven

**Convention:** All parameters must be configurable.

- **No hard-coded hyperparameters**
- **No hard-coded thresholds**
- **Use config.yaml for all settings**

**Rationale:** Enables systematic experimentation.

### 5. Logging Standards

**Convention:** Use Python logging module, not print statements.

- **Log pipeline stages**
- **Log record counts**
- **Log timing information**
- **Log errors and warnings**

**Rationale:** Enables debugging and monitoring.

### 6. Type Hints

**Convention:** Use Python 3.11+ type hints.

- **Function signatures with type hints**
- **Docstrings for all public functions**
- **PEP 8 compliance**

**Rationale:** Code readability and maintainability.

### 7. Git Workflow

**Convention:** Use Git branches for experiments.

- **main branch is always stable**
- **Create feature/experiment branches**
- **Review before merging to main**
- **Delete merged branches**

**Rationale:** Code quality and traceability.

### 8. Data Integrity

**Convention:** Do not silently modify raw competition data.

- **Keep raw data read-only**
- **Create processed copies**
- **Log all transformations**
- **Preserve raw data for debugging**

**Rationale:** Enables reproducibility and debugging.

### 9. No GPU Dependencies

**Convention:** Project must run on CPU-only environments.

- **No CUDA-specific code**
- **Use faiss-cpu, not faiss-gpu**
- **Ensure compatibility with SageMaker CPU instances**

**Rationale:** AWS credits may not support GPU instances.

### 10. AWS Readiness

**Convention:** Design for AWS deployment from the start.

- **Use S3-compatible paths**
- **Support environment-based configuration**
- **No local-only assumptions**

**Rationale:** Competition provides AWS credits for production runs.

---

## Constraint Checklist

Before each submission, verify:

- [ ] Using only competition-provided data
- [ ] No external business lookup APIs
- [ ] No geocoding or mapping APIs
- [ ] No internet-based enrichment
- [ ] Country handling is open-set
- [ ] Validation at S1 entity level
- [ ] Optimizing for F0.5 metric
- [ ] Every S1 entity appears exactly once
- [ ] Candidates generated before matching
- [ ] Experiment is reproducible
- [ ] Submission passes validator
- [ ] No credentials committed
- [ ] All paths configurable
- [ ] Code is modular and documented

---

**Last Updated:** 2026-09-27
**Version:** 1.0
