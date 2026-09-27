# Experiment E09 — S2/S3 Conflict Analysis Report

| rule                        |   s2_positives |   s3_positives |   precision |   recall |      f05 |   tp |   fp |   fn |
|:----------------------------|---------------:|---------------:|------------:|---------:|---------:|-----:|-----:|-----:|
| Pure Probability (Baseline) |            775 |            871 |           1 | 0.974484 | 0.99479  | 1604 |    0 |   42 |
| S2 Priority Penalty         |            775 |            871 |           1 | 0.464156 | 0.81242  |  764 |    0 |  882 |
| S3 Priority Penalty         |            775 |            871 |           1 | 0.510328 | 0.838993 |  840 |    0 |  806 |

- **Optimal Strategy**: Pure Model Probability (No artificial source bias)
- **F0.5 Score**: `0.99479`
