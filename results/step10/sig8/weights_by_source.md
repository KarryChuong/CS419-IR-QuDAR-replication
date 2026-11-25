| judge       | set                   | mode                    |   queries |   weight OS |   weight OD |   weight ES |   weight ED |   score OS |   score OD |   score ES |   score ED |
|:------------|:----------------------|:------------------------|----------:|------------:|------------:|------------:|------------:|-----------:|-----------:|-----------:|-----------:|
| gpt-4o-mini | FiQA                  | paper prompt + shuffled |       648 |      0.2687 |      0.2219 |      0.2783 |      0.2311 |      3.08  |      2.494 |      3.225 |      2.653 |
| gpt-4o-mini | FiQA                  | paper prompt + fixed    |       648 |      0.2919 |      0.1707 |      0.3205 |      0.217  |      3.233 |      1.565 |      3.62  |      2.369 |
| gpt-4o-mini | FiQA                  | code prompt + shuffled  |       648 |      0.2789 |      0.2183 |      0.2801 |      0.2227 |      3.593 |      2.83  |      3.63  |      2.914 |
| gpt-4o-mini | FiQA                  | code prompt + fixed     |       648 |      0.2785 |      0.2041 |      0.2983 |      0.2191 |      3.432 |      2.469 |      3.812 |      2.701 |
| Qwen2.5-7B  | FiQA                  | paper prompt + shuffled |       648 |      0.2499 |      0.2341 |      0.2721 |      0.2439 |      2.37  |      2.184 |      2.648 |      2.301 |
| Qwen2.5-7B  | FiQA                  | paper prompt + fixed    |       648 |      0.2649 |      0.1916 |      0.2487 |      0.2949 |      2.443 |      1.315 |      2.275 |      2.736 |
| Qwen2.5-7B  | FiQA                  | code prompt + shuffled  |       648 |      0.271  |      0.2166 |      0.2804 |      0.232  |      2.378 |      1.725 |      2.514 |      1.943 |
| Qwen2.5-7B  | FiQA                  | code prompt + fixed     |       648 |      0.284  |      0.2031 |      0.2475 |      0.2654 |      2.321 |      1.231 |      1.926 |      2.06  |
| gpt-4o-mini | SciDocs               | paper prompt + shuffled |      1000 |      0.2772 |      0.2188 |      0.2641 |      0.24   |      3.705 |      2.851 |      3.556 |      3.246 |
| gpt-4o-mini | SciDocs               | paper prompt + fixed    |      1000 |      0.3029 |      0.1416 |      0.3279 |      0.2276 |      3.849 |      1.357 |      4.112 |      2.947 |
| gpt-4o-mini | SciDocs               | code prompt + shuffled  |      1000 |      0.2963 |      0.1683 |      0.2905 |      0.2449 |      4.114 |      2.077 |      4.054 |      3.488 |
| gpt-4o-mini | SciDocs               | code prompt + fixed     |      1000 |      0.3084 |      0.1506 |      0.3046 |      0.2364 |      4.119 |      1.631 |      4.162 |      3.236 |
| Qwen2.5-7B  | SciDocs               | paper prompt + shuffled |      1000 |      0.2576 |      0.2343 |      0.2569 |      0.2512 |      2.847 |      2.506 |      2.815 |      2.741 |
| Qwen2.5-7B  | SciDocs               | paper prompt + fixed    |      1000 |      0.2478 |      0.1639 |      0.2603 |      0.328  |      2.649 |      1.233 |      2.854 |      3.46  |
| Qwen2.5-7B  | SciDocs               | code prompt + shuffled  |      1000 |      0.2829 |      0.1913 |      0.2775 |      0.2483 |      2.773 |      1.484 |      2.715 |      2.346 |
| Qwen2.5-7B  | SciDocs               | code prompt + fixed     |      1000 |      0.2756 |      0.1731 |      0.285  |      0.2664 |      2.57  |      1.012 |      2.703 |      2.368 |
| gpt-4o-mini | Climate-FEVER         | paper prompt + shuffled |      1535 |      0.2521 |      0.2041 |      0.2957 |      0.2481 |      2.545 |      1.873 |      3.109 |      2.564 |
| gpt-4o-mini | Climate-FEVER         | paper prompt + fixed    |      1535 |      0.2559 |      0.1684 |      0.3058 |      0.2699 |      2.324 |      1.051 |      3.021 |      2.576 |
| gpt-4o-mini | Climate-FEVER         | code prompt + shuffled  |      1535 |      0.2573 |      0.1743 |      0.3129 |      0.2555 |      3.001 |      1.743 |      3.723 |      3.065 |
| gpt-4o-mini | Climate-FEVER         | code prompt + fixed     |      1535 |      0.2446 |      0.1635 |      0.3204 |      0.2716 |      2.577 |      1.354 |      3.646 |      2.988 |
| Qwen2.5-7B  | Climate-FEVER         | paper prompt + shuffled |      1535 |      0.2503 |      0.2264 |      0.2695 |      0.2538 |      2.44  |      2.128 |      2.724 |      2.522 |
| Qwen2.5-7B  | Climate-FEVER         | paper prompt + fixed    |      1535 |      0.25   |      0.1704 |      0.2352 |      0.3444 |      2.61  |      1.315 |      2.461 |      3.577 |
| Qwen2.5-7B  | Climate-FEVER         | code prompt + shuffled  |      1535 |      0.2566 |      0.1965 |      0.2845 |      0.2624 |      2.007 |      1.192 |      2.364 |      2.114 |
| Qwen2.5-7B  | Climate-FEVER         | code prompt + fixed     |      1535 |      0.2785 |      0.1808 |      0.2151 |      0.3256 |      2.197 |      0.822 |      1.457 |      2.665 |
| gpt-4o-mini | HotpotQA (first 1500) | paper prompt + shuffled |      1500 |      0.298  |      0.2145 |      0.2378 |      0.2496 |      3.194 |      2.117 |      2.479 |      2.675 |
| gpt-4o-mini | HotpotQA (first 1500) | paper prompt + fixed    |      1500 |      0.3025 |      0.1792 |      0.2728 |      0.2455 |      3.223 |      1.549 |      3.019 |      2.662 |
| gpt-4o-mini | HotpotQA (first 1500) | code prompt + shuffled  |      1500 |      0.3098 |      0.2041 |      0.2341 |      0.2521 |      3.614 |      2.202 |      2.725 |      2.997 |
| gpt-4o-mini | HotpotQA (first 1500) | code prompt + fixed     |      1500 |      0.2971 |      0.1913 |      0.2511 |      0.2604 |      3.541 |      2.043 |      3.114 |      3.142 |
| Qwen2.5-7B  | HotpotQA (first 1500) | paper prompt + shuffled |      1500 |      0.2738 |      0.2265 |      0.2464 |      0.2533 |      3.161 |      2.525 |      2.803 |      2.914 |
| Qwen2.5-7B  | HotpotQA (first 1500) | paper prompt + fixed    |      1500 |      0.2839 |      0.168  |      0.2351 |      0.3131 |      3.439 |      1.611 |      2.825 |      3.66  |
| Qwen2.5-7B  | HotpotQA (first 1500) | code prompt + shuffled  |      1500 |      0.2957 |      0.2141 |      0.2358 |      0.2544 |      3.328 |      2.202 |      2.559 |      2.849 |
| Qwen2.5-7B  | HotpotQA (first 1500) | code prompt + fixed     |      1500 |      0.3098 |      0.1951 |      0.2149 |      0.2803 |      3.505 |      1.851 |      2.347 |      3.109 |
| gpt-4o-mini | SciFact               | paper prompt + shuffled |       300 |      0.2845 |      0.2398 |      0.251  |      0.2247 |      3.973 |      3.36  |      3.527 |      3.163 |
| gpt-4o-mini | SciFact               | paper prompt + fixed    |       300 |      0.2892 |      0.2262 |      0.2638 |      0.2208 |      3.92  |      3.09  |      3.653 |      3.01  |
| gpt-4o-mini | SciFact               | code prompt + shuffled  |       300 |      0.2832 |      0.2408 |      0.2521 |      0.2239 |      4.15  |      3.577 |      3.75  |      3.357 |
| gpt-4o-mini | SciFact               | code prompt + fixed     |       300 |      0.2829 |      0.2381 |      0.2654 |      0.2136 |      4.12  |      3.497 |      3.963 |      3.163 |
| Qwen2.5-7B  | SciFact               | paper prompt + shuffled |       300 |      0.2636 |      0.2422 |      0.2544 |      0.2398 |      3.437 |      3.15  |      3.3   |      3.1   |
| Qwen2.5-7B  | SciFact               | paper prompt + fixed    |       300 |      0.2698 |      0.195  |      0.2507 |      0.2844 |      3.523 |      2.357 |      3.32  |      3.577 |
| Qwen2.5-7B  | SciFact               | code prompt + shuffled  |       300 |      0.2782 |      0.24   |      0.2576 |      0.2242 |      3.41  |      2.937 |      3.153 |      2.703 |
| Qwen2.5-7B  | SciFact               | code prompt + fixed     |       300 |      0.2813 |      0.2404 |      0.2365 |      0.2418 |      3.29  |      2.727 |      2.723 |      2.723 |
| gpt-4o-mini | NFCorpus              | paper prompt + shuffled |       323 |      0.2969 |      0.2404 |      0.2521 |      0.2106 |      3.598 |      2.941 |      3.087 |      2.523 |
| gpt-4o-mini | NFCorpus              | paper prompt + fixed    |       323 |      0.3068 |      0.2036 |      0.2849 |      0.2047 |      3.647 |      2.353 |      3.477 |      2.375 |
| gpt-4o-mini | NFCorpus              | code prompt + shuffled  |       323 |      0.2941 |      0.2416 |      0.2585 |      0.2059 |      3.929 |      3.316 |      3.514 |      2.805 |
| gpt-4o-mini | NFCorpus              | code prompt + fixed     |       323 |      0.3013 |      0.2318 |      0.267  |      0.1998 |      3.991 |      3.146 |      3.669 |      2.659 |
| Qwen2.5-7B  | NFCorpus              | paper prompt + shuffled |       323 |      0.2683 |      0.2437 |      0.2506 |      0.2375 |      2.851 |      2.539 |      2.632 |      2.471 |
| Qwen2.5-7B  | NFCorpus              | paper prompt + fixed    |       323 |      0.2679 |      0.2029 |      0.2487 |      0.2805 |      2.889 |      1.932 |      2.706 |      2.935 |
| Qwen2.5-7B  | NFCorpus              | code prompt + shuffled  |       323 |      0.2849 |      0.2421 |      0.2481 |      0.2249 |      2.814 |      2.316 |      2.362 |      2.068 |
| Qwen2.5-7B  | NFCorpus              | code prompt + fixed     |       323 |      0.2708 |      0.2296 |      0.2401 |      0.2595 |      2.644 |      2.034 |      2.235 |      2.461 |
| gpt-4o-mini | DBPedia-entity        | paper prompt + shuffled |       400 |      0.234  |      0.2234 |      0.2687 |      0.2738 |      2.575 |      2.467 |      3.075 |      3.16  |
| gpt-4o-mini | DBPedia-entity        | paper prompt + fixed    |       400 |      0.2384 |      0.1898 |      0.3026 |      0.2692 |      2.645 |      1.955 |      3.565 |      3.172 |
| gpt-4o-mini | DBPedia-entity        | code prompt + shuffled  |       400 |      0.2356 |      0.2193 |      0.2678 |      0.2774 |      2.965 |      2.712 |      3.428 |      3.587 |
| gpt-4o-mini | DBPedia-entity        | code prompt + fixed     |       400 |      0.2345 |      0.2084 |      0.2785 |      0.2786 |      2.978 |      2.587 |      3.683 |      3.612 |
| Qwen2.5-7B  | DBPedia-entity        | paper prompt + shuffled |       400 |      0.2419 |      0.2376 |      0.2569 |      0.2635 |      2.84  |      2.805 |      3.062 |      3.212 |
| Qwen2.5-7B  | DBPedia-entity        | paper prompt + fixed    |       400 |      0.2466 |      0.1824 |      0.2467 |      0.3243 |      3.12  |      2.047 |      3.127 |      3.967 |
| Qwen2.5-7B  | DBPedia-entity        | code prompt + shuffled  |       400 |      0.2355 |      0.2324 |      0.2552 |      0.2768 |      2.65  |      2.63  |      3     |      3.28  |
| Qwen2.5-7B  | DBPedia-entity        | code prompt + fixed     |       400 |      0.2631 |      0.2033 |      0.2419 |      0.2918 |      3.123 |      2.205 |      2.857 |      3.47  |
| gpt-4o-mini | FEVER (first 1500)    | paper prompt + shuffled |      1500 |      0.2558 |      0.2218 |      0.277  |      0.2454 |      2.817 |      2.335 |      3.079 |      2.743 |
| gpt-4o-mini | FEVER (first 1500)    | paper prompt + fixed    |      1500 |      0.253  |      0.1972 |      0.3021 |      0.2477 |      2.721 |      1.895 |      3.281 |      2.698 |
| gpt-4o-mini | FEVER (first 1500)    | code prompt + shuffled  |      1500 |      0.2668 |      0.21   |      0.2748 |      0.2483 |      3.277 |      2.442 |      3.355 |      3.099 |
| gpt-4o-mini | FEVER (first 1500)    | code prompt + fixed     |      1500 |      0.2608 |      0.2022 |      0.2904 |      0.2466 |      3.139 |      2.255 |      3.541 |      2.985 |
| Qwen2.5-7B  | FEVER (first 1500)    | paper prompt + shuffled |      1500 |      0.2495 |      0.2364 |      0.2653 |      0.2488 |      2.97  |      2.781 |      3.157 |      2.989 |
| Qwen2.5-7B  | FEVER (first 1500)    | paper prompt + fixed    |      1500 |      0.2706 |      0.172  |      0.2419 |      0.3155 |      3.322 |      1.724 |      2.966 |      3.757 |
| Qwen2.5-7B  | FEVER (first 1500)    | code prompt + shuffled  |      1500 |      0.2586 |      0.2215 |      0.2695 |      0.2504 |      2.759 |      2.252 |      2.887 |      2.692 |
| Qwen2.5-7B  | FEVER (first 1500)    | code prompt + fixed     |      1500 |      0.2701 |      0.2136 |      0.2297 |      0.2866 |      2.867 |      1.977 |      2.299 |      3.017 |