| dataset       | A                      | B                     |   Gen. diff | 95% CI           |   p bootstrap |   p t-test | sig (p<0.05)   |
|:--------------|:-----------------------|:----------------------|------------:|:-----------------|--------------:|-----------:|:---------------|
| fiqa_test     | QuDAR-confidence       | OS                    |      -0.014 | [-0.058, 0.030]  |        0.565  |     0.5382 |                |
| fiqa_test     | QuDAR-llm              | OS                    |       0     | [-0.044, 0.044]  |        1      |     1      |                |
| fiqa_test     | QuDAR-confidence       | Retriever-Type Fusion |      -0.026 | [-0.072, 0.020]  |        0.2798 |     0.2601 |                |
| fiqa_test     | QuDAR-llm              | QuDAR-confidence      |       0.014 | [-0.026, 0.054]  |        0.5246 |     0.4951 |                |
| fiqa_test     | Static-train (ours)    | QuDAR-confidence      |       0.04  | [0.002, 0.078]   |        0.044  |     0.0369 | *              |
| fiqa_test     | Static-train (ours)    | QuDAR-llm             |       0.026 | [-0.012, 0.062]  |        0.1918 |     0.1779 |                |
| fiqa_test     | Model B (ours)         | Static-train (ours)   |       0.018 | [-0.022, 0.058]  |        0.4242 |     0.3803 |                |
| fiqa_test     | Query-wise UB (oracle) | QuDAR-confidence      |       0.106 | [0.064, 0.148]   |        0      |     0      | *              |
| hotpotqa_test | QuDAR-confidence       | OS                    |       0.078 | [0.040, 0.116]   |        0      |     0.0001 | *              |
| hotpotqa_test | QuDAR-llm              | OS                    |       0.092 | [0.054, 0.130]   |        0      |     0      | *              |
| hotpotqa_test | QuDAR-confidence       | Retriever-Type Fusion |       0.06  | [0.022, 0.098]   |        0.0032 |     0.0019 | *              |
| hotpotqa_test | QuDAR-llm              | QuDAR-confidence      |       0.014 | [-0.016, 0.046]  |        0.4058 |     0.3706 |                |
| hotpotqa_test | Static-train (ours)    | QuDAR-confidence      |       0.008 | [-0.024, 0.040]  |        0.6614 |     0.6229 |                |
| hotpotqa_test | Static-train (ours)    | QuDAR-llm             |      -0.006 | [-0.038, 0.026]  |        0.7422 |     0.7102 |                |
| hotpotqa_test | Model B (ours)         | Static-train (ours)   |      -0.032 | [-0.062, -0.002] |        0.0372 |     0.0293 | *              |
| hotpotqa_test | Query-wise UB (oracle) | QuDAR-confidence      |       0.056 | [0.022, 0.090]   |        0.0014 |     0.0015 | *              |