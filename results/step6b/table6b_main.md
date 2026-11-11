| dataset       | method                 |   Recall@10 |   nDCG@10 |   Precision@1 |   gap closed % (Recall) |
|:--------------|:-----------------------|------------:|----------:|--------------:|------------------------:|
| FiQA          | QuDAR-simple (Equal)   |      0.3333 |    0.2512 |        0.2099 |                    -1.3 |
| FiQA          | QuDAR-confidence       |      0.3354 |    0.2525 |        0.213  |                     0   |
| FiQA          | QuDAR-llm              |      0.3281 |    0.2539 |        0.2099 |                    -4.6 |
| FiQA          | QuDAR-llm-debiased     |      0.3347 |    0.2585 |        0.2222 |                    -0.5 |
| FiQA          | Static-train [both]    |      0.3555 |    0.2754 |        0.2269 |                    12.6 |
| FiQA          | Static-train [fiqa]    |      0.3584 |    0.2843 |        0.2515 |                    14.4 |
| FiQA          | Direct-A [both]        |      0.3605 |    0.2761 |        0.2238 |                    15.7 |
| FiQA          | Direct-B [both]        |      0.3587 |    0.2803 |        0.2392 |                    14.7 |
| FiQA          | Model A [both]         |      0.3579 |    0.288  |        0.2562 |                    14.2 |
| FiQA          | Gated A [both]         |      0.3578 |    0.2835 |        0.2407 |                    14.1 |
| FiQA          | Model A-sf [both]      |      0.358  |    0.2823 |        0.25   |                    14.2 |
| FiQA          | Gated A-sf [both]      |      0.3566 |    0.2815 |        0.2469 |                    13.3 |
| FiQA          | Model B [both]         |      0.3517 |    0.2849 |        0.2577 |                    10.2 |
| FiQA          | Gated B [both]         |      0.3529 |    0.285  |        0.2577 |                    11   |
| FiQA          | Model B-sf [both]      |      0.3465 |    0.2786 |        0.2485 |                     6.9 |
| FiQA          | Gated B-sf [both]      |      0.3552 |    0.2812 |        0.2438 |                    12.4 |
| FiQA          | Static UB (oracle)     |      0.3598 |    0.2855 |        0.2485 |                    15.3 |
| FiQA          | Query-wise UB (oracle) |      0.4944 |    0.4321 |        0.4491 |                   100   |
| SciDocs       | QuDAR-simple (Equal)   |      0.1901 |    0.1793 |        0.203  |                    -0.7 |
| SciDocs       | QuDAR-confidence       |      0.1907 |    0.1798 |        0.204  |                     0   |
| SciDocs       | QuDAR-llm              |      0.1899 |    0.1799 |        0.206  |                    -1   |
| SciDocs       | QuDAR-llm-debiased     |      0.1903 |    0.1802 |        0.205  |                    -0.5 |
| SciDocs       | Static-train [both]    |      0.1917 |    0.1808 |        0.203  |                     1.2 |
| SciDocs       | Static-train [fiqa]    |      0.1897 |    0.179  |        0.203  |                    -1.2 |
| SciDocs       | Direct-A [both]        |      0.1929 |    0.1821 |        0.21   |                     2.6 |
| SciDocs       | Direct-B [both]        |      0.1939 |    0.1823 |        0.203  |                     3.9 |
| SciDocs       | Model A [both]         |      0.1879 |    0.1768 |        0.196  |                    -3.3 |
| SciDocs       | Gated A [both]         |      0.19   |    0.1776 |        0.191  |                    -0.8 |
| SciDocs       | Model A-sf [both]      |      0.184  |    0.1735 |        0.186  |                    -8   |
| SciDocs       | Gated A-sf [both]      |      0.1846 |    0.1737 |        0.187  |                    -7.3 |
| SciDocs       | Model B [both]         |      0.1876 |    0.1754 |        0.194  |                    -3.7 |
| SciDocs       | Gated B [both]         |      0.1899 |    0.1766 |        0.191  |                    -1   |
| SciDocs       | Model B-sf [both]      |      0.1833 |    0.1739 |        0.196  |                    -8.8 |
| SciDocs       | Gated B-sf [both]      |      0.1885 |    0.1774 |        0.198  |                    -2.6 |
| SciDocs       | Static UB (oracle)     |      0.1923 |    0.1823 |        0.206  |                     1.9 |
| SciDocs       | Query-wise UB (oracle) |      0.2749 |    0.281  |        0.393  |                   100   |
| Climate-FEVER | QuDAR-simple (Equal)   |      0.2504 |    0.2047 |        0.1909 |                    -0.3 |
| Climate-FEVER | QuDAR-confidence       |      0.2507 |    0.2043 |        0.1896 |                     0   |
| Climate-FEVER | QuDAR-llm              |      0.2567 |    0.2093 |        0.2013 |                     4.8 |
| Climate-FEVER | QuDAR-llm-debiased     |      0.2572 |    0.2089 |        0.1993 |                     5.3 |
| Climate-FEVER | Static-train [both]    |      0.2473 |    0.2012 |        0.1902 |                    -2.8 |
| Climate-FEVER | Static-train [fiqa]    |      0.2584 |    0.2121 |        0.2    |                     6.3 |
| Climate-FEVER | Direct-A [both]        |      0.2433 |    0.201  |        0.1935 |                    -6.1 |
| Climate-FEVER | Direct-B [both]        |      0.2462 |    0.2026 |        0.1909 |                    -3.7 |
| Climate-FEVER | Model A [both]         |      0.2232 |    0.1845 |        0.1798 |                   -22.5 |
| Climate-FEVER | Gated A [both]         |      0.2324 |    0.1902 |        0.1818 |                   -14.9 |
| Climate-FEVER | Model A-sf [both]      |      0.2325 |    0.1925 |        0.185  |                   -14.9 |
| Climate-FEVER | Gated A-sf [both]      |      0.2314 |    0.1907 |        0.1824 |                   -15.8 |
| Climate-FEVER | Model B [both]         |      0.2272 |    0.1865 |        0.1772 |                   -19.2 |
| Climate-FEVER | Gated B [both]         |      0.2334 |    0.1894 |        0.1805 |                   -14.1 |
| Climate-FEVER | Model B-sf [both]      |      0.2391 |    0.1971 |        0.1909 |                    -9.5 |
| Climate-FEVER | Gated B-sf [both]      |      0.2392 |    0.196  |        0.187  |                    -9.4 |
| Climate-FEVER | Static UB (oracle)     |      0.291  |    0.2352 |        0.2091 |                    32.8 |
| Climate-FEVER | Query-wise UB (oracle) |      0.3734 |    0.3471 |        0.4059 |                   100   |
| HotpotQA      | QuDAR-simple (Equal)   |      0.6827 |    0.6318 |        0.6782 |                    -3   |
| HotpotQA      | QuDAR-confidence       |      0.6853 |    0.6366 |        0.6887 |                     0   |
| HotpotQA      | QuDAR-llm              |      0.6704 |    0.6355 |        0.7168 |                   -17.4 |
| HotpotQA      | QuDAR-llm-debiased     |      0.6714 |    0.6371 |        0.7187 |                   -16.2 |
| HotpotQA      | Static-train [both]    |      0.7039 |    0.6621 |        0.7354 |                    21.8 |
| HotpotQA      | Static-train [fiqa]    |      0.6943 |    0.6546 |        0.7332 |                    10.5 |
| HotpotQA      | Direct-A [both]        |      0.7047 |    0.6664 |        0.7438 |                    22.7 |
| HotpotQA      | Direct-B [both]        |      0.7074 |    0.6715 |        0.7552 |                    25.9 |
| HotpotQA      | Model A [both]         |      0.6951 |    0.6595 |        0.7403 |                    11.5 |
| HotpotQA      | Gated A [both]         |      0.7005 |    0.6618 |        0.741  |                    17.8 |
| HotpotQA      | Model A-sf [both]      |      0.6945 |    0.6589 |        0.7409 |                    10.8 |
| HotpotQA      | Gated A-sf [both]      |      0.6989 |    0.661  |        0.7402 |                    15.9 |
| HotpotQA      | Model B [both]         |      0.6966 |    0.6653 |        0.7554 |                    13.3 |
| HotpotQA      | Gated B [both]         |      0.7014 |    0.6656 |        0.7495 |                    18.8 |
| HotpotQA      | Model B-sf [both]      |      0.6991 |    0.6662 |        0.7546 |                    16.2 |
| HotpotQA      | Gated B-sf [both]      |      0.7026 |    0.6653 |        0.7479 |                    20.3 |
| HotpotQA      | Static UB (oracle)     |      0.7041 |    0.6634 |        0.7377 |                    22   |
| HotpotQA      | Query-wise UB (oracle) |      0.7708 |    0.7637 |        0.8806 |                   100   |