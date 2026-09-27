# Test Data Audit

## S1 Dataset
- **Total S1 Rows:** `1,732,544`
- **Unique S1 IDs:** `1,732,544`
- **Duplicate S1 IDs:** `0`
- **Null / Missing IDs:** `0`
- **Malformed IDs:** `0`
- **EXPECTED_UNIQUE_S1_ENTITIES:** `1,732,544`

## Candidate / Test Dataset
- **Total Candidates Universe:** `9,969,589`
- **Source-2 (S2) Count:** `4,887,273`
- **Source-3 (S3) Count:** `5,082,316`
- **Duplicate Candidates:** `0`
- **Invalid / Malformed Candidates:** `0`

## Candidate Distribution

| Candidate Count | Number of S1 | Percentage |
|---|---:|---:|
| 0 | `1,725,208` | `99.58%` |
| 1 | `759` | `0.04%` |
| 2 | `486` | `0.03%` |
| 3–5 | `750` | `0.04%` |
| >5 | `5,341` | `0.31%` |

- **Minimum Candidates per S1:** `0`
- **Maximum Candidates per S1:** `716`
- **Mean Candidates per S1:** `0.4679`
- **Median Candidates per S1:** `0.0`
- **Zero Candidate S1 Entities:** `1,725,208` (`99.58%`)

## Blocking Validation
- **candidate_pairs.tsv Rows:** `1,732,544`
- **Unique S1:** `1,732,544`
- **Duplicate Pairs:** `0`
- **Invalid Candidate IDs:** `0`

## Blocking Recall
- **Validation Blocking Recall (Phase 6/7 split):** `0.940034` (94.00%)
- **Test Set Note:** Test blocking recall cannot be directly measured because test ground-truth labels are unavailable.
