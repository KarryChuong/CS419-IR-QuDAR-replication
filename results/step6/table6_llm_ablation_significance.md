| dataset               | effect                                 | metric    |   mean diff | 95% CI             |   p bootstrap |   p t-test | sig (p<0.05)   |
|:----------------------|:---------------------------------------|:----------|------------:|:-------------------|--------------:|-----------:|:---------------|
| FiQA                  | order (paper prompt): fixed - shuffled | Recall@10 |      0.0142 | [0.0005, 0.0276]   |        0.0402 |     0.041  | *              |
| FiQA                  | order (paper prompt): fixed - shuffled | nDCG@10   |      0.0173 | [0.0081, 0.0265]   |        0      |     0.0003 | *              |
| FiQA                  | order (code prompt): fixed - shuffled  | Recall@10 |      0.0136 | [0.0058, 0.0221]   |        0.0004 |     0.0012 | *              |
| FiQA                  | order (code prompt): fixed - shuffled  | nDCG@10   |      0.0124 | [0.0065, 0.0184]   |        0      |     0.0001 | *              |
| FiQA                  | prompt (shuffled): code - paper        | Recall@10 |      0.0106 | [-0.0008, 0.0217]  |        0.0712 |     0.0685 |                |
| FiQA                  | prompt (shuffled): code - paper        | nDCG@10   |      0.0085 | [0.0008, 0.0162]   |        0.0316 |     0.0299 | *              |
| FiQA                  | prompt (fixed): code - paper           | Recall@10 |      0.0099 | [0.0017, 0.0186]   |        0.0182 |     0.0245 | *              |
| FiQA                  | prompt (fixed): code - paper           | nDCG@10   |      0.0036 | [-0.0028, 0.0103]  |        0.2786 |     0.2894 |                |
| FiQA                  | debias (paper prompt, shuffled)        | Recall@10 |      0.0066 | [0.0009, 0.0128]   |        0.0188 |     0.0289 | *              |
| FiQA                  | debias (paper prompt, shuffled)        | nDCG@10   |      0.0046 | [0.0008, 0.0086]   |        0.016  |     0.0241 | *              |
| SciDocs               | order (paper prompt): fixed - shuffled | Recall@10 |      0.0007 | [-0.0039, 0.0052]  |        0.7644 |     0.7637 |                |
| SciDocs               | order (paper prompt): fixed - shuffled | nDCG@10   |      0.001  | [-0.0025, 0.0045]  |        0.5614 |     0.5589 |                |
| SciDocs               | order (code prompt): fixed - shuffled  | Recall@10 |     -0.0031 | [-0.0062, -0.0001] |        0.0454 |     0.0438 | *              |
| SciDocs               | order (code prompt): fixed - shuffled  | nDCG@10   |     -0.0016 | [-0.0037, 0.0005]  |        0.1252 |     0.1262 |                |
| SciDocs               | prompt (shuffled): code - paper        | Recall@10 |      0.0022 | [-0.0018, 0.0061]  |        0.2736 |     0.2742 |                |
| SciDocs               | prompt (shuffled): code - paper        | nDCG@10   |      0.0024 | [-0.0006, 0.0052]  |        0.1172 |     0.1114 |                |
| SciDocs               | prompt (fixed): code - paper           | Recall@10 |     -0.0016 | [-0.0050, 0.0018]  |        0.358  |     0.3528 |                |
| SciDocs               | prompt (fixed): code - paper           | nDCG@10   |     -0.0003 | [-0.0028, 0.0022]  |        0.8214 |     0.8126 |                |
| SciDocs               | debias (paper prompt, shuffled)        | Recall@10 |      0.0004 | [-0.0021, 0.0030]  |        0.7282 |     0.7277 |                |
| SciDocs               | debias (paper prompt, shuffled)        | nDCG@10   |      0.0003 | [-0.0012, 0.0019]  |        0.6566 |     0.6526 |                |
| Climate-FEVER         | order (paper prompt): fixed - shuffled | Recall@10 |     -0.0015 | [-0.0076, 0.0047]  |        0.6324 |     0.6294 |                |
| Climate-FEVER         | order (paper prompt): fixed - shuffled | nDCG@10   |      0.0002 | [-0.0039, 0.0045]  |        0.9046 |     0.9103 |                |
| Climate-FEVER         | order (code prompt): fixed - shuffled  | Recall@10 |     -0.0018 | [-0.0066, 0.0031]  |        0.478  |     0.4773 |                |
| Climate-FEVER         | order (code prompt): fixed - shuffled  | nDCG@10   |     -0.0001 | [-0.0032, 0.0032]  |        0.956  |     0.9595 |                |
| Climate-FEVER         | prompt (shuffled): code - paper        | Recall@10 |      0.0042 | [-0.0010, 0.0095]  |        0.1144 |     0.1162 |                |
| Climate-FEVER         | prompt (shuffled): code - paper        | nDCG@10   |      0.0053 | [0.0016, 0.0090]   |        0.0046 |     0.0049 | *              |
| Climate-FEVER         | prompt (fixed): code - paper           | Recall@10 |      0.0039 | [-0.0011, 0.0090]  |        0.1246 |     0.1346 |                |
| Climate-FEVER         | prompt (fixed): code - paper           | nDCG@10   |      0.005  | [0.0017, 0.0083]   |        0.0028 |     0.0029 | *              |
| Climate-FEVER         | debias (paper prompt, shuffled)        | Recall@10 |      0.0005 | [-0.0026, 0.0036]  |        0.718  |     0.7275 |                |
| Climate-FEVER         | debias (paper prompt, shuffled)        | nDCG@10   |     -0.0004 | [-0.0022, 0.0015]  |        0.6818 |     0.6751 |                |
| HotpotQA (first 1500) | order (paper prompt): fixed - shuffled | Recall@10 |      0.0087 | [0.0010, 0.0167]   |        0.0324 |     0.0302 | *              |
| HotpotQA (first 1500) | order (paper prompt): fixed - shuffled | nDCG@10   |      0.0075 | [0.0005, 0.0145]   |        0.0352 |     0.0372 | *              |
| HotpotQA (first 1500) | order (code prompt): fixed - shuffled  | Recall@10 |     -0.0043 | [-0.0097, 0.0010]  |        0.1268 |     0.1123 |                |
| HotpotQA (first 1500) | order (code prompt): fixed - shuffled  | nDCG@10   |     -0.0032 | [-0.0076, 0.0011]  |        0.1488 |     0.1563 |                |
| HotpotQA (first 1500) | prompt (shuffled): code - paper        | Recall@10 |      0.019  | [0.0120, 0.0263]   |        0      |     0      | *              |
| HotpotQA (first 1500) | prompt (shuffled): code - paper        | nDCG@10   |      0.0191 | [0.0129, 0.0255]   |        0      |     0      | *              |
| HotpotQA (first 1500) | prompt (fixed): code - paper           | Recall@10 |      0.006  | [-0.0010, 0.0130]  |        0.102  |     0.0975 |                |
| HotpotQA (first 1500) | prompt (fixed): code - paper           | nDCG@10   |      0.0085 | [0.0023, 0.0146]   |        0.0066 |     0.0079 | *              |
| HotpotQA (first 1500) | debias (paper prompt, shuffled)        | Recall@10 |      0.0007 | [-0.0033, 0.0047]  |        0.8084 |     0.7519 |                |
| HotpotQA (first 1500) | debias (paper prompt, shuffled)        | nDCG@10   |      0.0014 | [-0.0012, 0.0041]  |        0.279  |     0.2833 |                |