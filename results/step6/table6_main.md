| dataset       | method                 |   Recall@10 |   nDCG@10 |   Precision@1 |   gap closed % (Recall) |
|:--------------|:-----------------------|------------:|----------:|--------------:|------------------------:|
| FiQA          | QuDAR-simple (Equal)   |      0.3333 |    0.2512 |        0.2099 |                    -1.3 |
| FiQA          | QuDAR-confidence       |      0.3354 |    0.2525 |        0.213  |                     0   |
| FiQA          | Static-train [both]    |      0.3555 |    0.2754 |        0.2269 |                    12.6 |
| FiQA          | QuDAR-llm              |      0.3281 |    0.2539 |        0.2099 |                    -4.6 |
| FiQA          | QuDAR-llm-debiased     |      0.3347 |    0.2585 |        0.2222 |                    -0.5 |
| FiQA          | Direct-A [both]        |      0.3605 |    0.2761 |        0.2238 |                    15.7 |
| FiQA          | Model A [both]         |      0.3524 |    0.2789 |        0.2438 |                    10.7 |
| FiQA          | Direct-B [both]        |      0.3587 |    0.2803 |        0.2392 |                    14.7 |
| FiQA          | Model B [both]         |      0.3515 |    0.286  |        0.2731 |                    10.1 |
| FiQA          | Static UB (oracle)     |      0.3598 |    0.2855 |        0.2485 |                    15.3 |
| FiQA          | Query-wise UB (oracle) |      0.4944 |    0.4321 |        0.4491 |                   100   |
| SciDocs       | QuDAR-simple (Equal)   |      0.1901 |    0.1793 |        0.203  |                    -0.7 |
| SciDocs       | QuDAR-confidence       |      0.1907 |    0.1798 |        0.204  |                     0   |
| SciDocs       | Static-train [both]    |      0.1917 |    0.1808 |        0.203  |                     1.2 |
| SciDocs       | QuDAR-llm              |      0.1899 |    0.1799 |        0.206  |                    -1   |
| SciDocs       | QuDAR-llm-debiased     |      0.1903 |    0.1802 |        0.205  |                    -0.5 |
| SciDocs       | Direct-A [both]        |      0.1929 |    0.1821 |        0.21   |                     2.6 |
| SciDocs       | Model A [both]         |      0.1855 |    0.1753 |        0.198  |                    -6.2 |
| SciDocs       | Direct-B [both]        |      0.1939 |    0.1823 |        0.203  |                     3.9 |
| SciDocs       | Model B [both]         |      0.186  |    0.1759 |        0.201  |                    -5.6 |
| SciDocs       | Static UB (oracle)     |      0.1923 |    0.1823 |        0.206  |                     1.9 |
| SciDocs       | Query-wise UB (oracle) |      0.2749 |    0.281  |        0.393  |                   100   |
| Climate-FEVER | QuDAR-simple (Equal)   |      0.2504 |    0.2047 |        0.1909 |                    -0.3 |
| Climate-FEVER | QuDAR-confidence       |      0.2507 |    0.2043 |        0.1896 |                     0   |
| Climate-FEVER | Static-train [both]    |      0.2473 |    0.2012 |        0.1902 |                    -2.8 |
| Climate-FEVER | QuDAR-llm              |      0.2567 |    0.2093 |        0.2013 |                     4.8 |
| Climate-FEVER | QuDAR-llm-debiased     |      0.2572 |    0.2089 |        0.1993 |                     5.3 |
| Climate-FEVER | Direct-A [both]        |      0.2433 |    0.201  |        0.1935 |                    -6.1 |
| Climate-FEVER | Model A [both]         |      0.2207 |    0.1843 |        0.1792 |                   -24.5 |
| Climate-FEVER | Direct-B [both]        |      0.2462 |    0.2026 |        0.1909 |                    -3.7 |
| Climate-FEVER | Model B [both]         |      0.2305 |    0.1878 |        0.1739 |                   -16.5 |
| Climate-FEVER | Static UB (oracle)     |      0.291  |    0.2352 |        0.2091 |                    32.8 |
| Climate-FEVER | Query-wise UB (oracle) |      0.3734 |    0.3471 |        0.4059 |                   100   |
| HotpotQA      | QuDAR-simple (Equal)   |      0.6827 |    0.6318 |        0.6782 |                    -3   |
| HotpotQA      | QuDAR-confidence       |      0.6853 |    0.6366 |        0.6887 |                     0   |
| HotpotQA      | Static-train [both]    |      0.7039 |    0.6621 |        0.7354 |                    21.8 |
| HotpotQA      | QuDAR-llm              |      0.6704 |    0.6355 |        0.7168 |                   -17.4 |
| HotpotQA      | QuDAR-llm-debiased     |      0.6714 |    0.6371 |        0.7187 |                   -16.2 |
| HotpotQA      | Direct-A [both]        |      0.7047 |    0.6664 |        0.7438 |                    22.7 |
| HotpotQA      | Model A [both]         |      0.6959 |    0.6598 |        0.7422 |                    12.4 |
| HotpotQA      | Direct-B [both]        |      0.7074 |    0.6715 |        0.7552 |                    25.9 |
| HotpotQA      | Model B [both]         |      0.699  |    0.6661 |        0.7531 |                    16   |
| HotpotQA      | Static UB (oracle)     |      0.7041 |    0.6634 |        0.7377 |                    22   |
| HotpotQA      | Query-wise UB (oracle) |      0.7708 |    0.7637 |        0.8806 |                   100   |