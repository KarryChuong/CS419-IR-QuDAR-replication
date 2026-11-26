# Phát hiện khi tái hiện QuDAR

Ghi lại những chỗ code công bố (`kaist-dmlab/QuDAR`) khác mô tả trong paper, các thí nghiệm đã chạy để kiểm chứng, quyết định của đồ án, kết quả các phần mở rộng trong đồ án (mục 9–14), và phần mở rộng sau đồ án lên 8 bộ dữ liệu với 2 LLM chấm (Step 10, mục 15–22). Mọi số liệu là trên tập test, dạng **Recall@10 / nDCG@10 / P@1** nếu không ghi khác.

## Tóm tắt

| # | Thành phần | Code công bố | Paper / bằng chứng | Quyết định của đồ án |
|---|---|---|---|---|
| 1 | BM25 | `rank_bm25`, tách từ bằng `.split()`, không bỏ stopword | Kết quả thấp hơn paper khoảng 30%; bm25s mặc định khớp gần như tuyệt đối | Dùng bm25s |
| 2 | Contriever | Tiền tố `query: ` / `passage: ` kiểu E5 | OD thấp hơn paper 4–19%; bỏ tiền tố cũng không khớp | Giữ code tác giả |
| 3 | ED (dense + query mở rộng) | Encode đoạn mở rộng như passage | FiQA thấp hơn paper 31%; 4 cách encode đều không khớp nhất quán | Giữ code tác giả |
| 4 | QuDAR-llm | Thứ tự passage cố định (ES, OS, ED, OD), prompt rút gọn | Phụ lục D: xáo trộn, Prompt 1 đầy đủ; LLM thiên lệch vị trí mạnh | Chạy theo paper, báo cáo thêm ablation 2×2 |
| 5 | DAT | Không có trong repo | Prompt dựng lại từ mô tả Phụ lục A bài DAT | Khớp paper ±4% |

Kết quả Step 10 (8 bộ, 2 LLM chấm, Holm trong từng họ kiểm định đặt trước):

| # | Câu hỏi | Kết quả | Mục |
|---|---|---|---|
| 6 | QuDAR-llm theo paper có vượt chia đều (Equal) không? | Không, ở 7/8 bộ với cả gpt-4o-mini lẫn Qwen2.5-7B; ngoại lệ là FEVER | 17 |
| 7 | LLM chấm có thiên lệch vị trí không? | Có, 32/32 kiểm định hoán vị; hình dạng khác nhau giữa hai LLM (gpt-4o-mini ưu ái vị trí 1, Qwen vị trí 3, cả hai chấm vị trí 4 thấp nhất) | 18 |
| 8 | Thứ tự cố định của code có trung tính không? | Không: dồn trọng số vào vị trí LLM ưu ái; có lợi với gpt-4o-mini (23/32 phép so), có hại với Qwen (22/32) | 18 |
| 9 | Prompt code hay prompt paper? | Prompt code tốt hơn ở 56/64 phép so, 11 phép có ý nghĩa, đều nghiêng về prompt code; thu hẹp thiên lệch vị trí ở 15/16 trường hợp | 19 |
| 10 | Trọng số tĩnh chọn trên train cùng bộ so với QuDAR? | Cao hơn cả ba biến thể QuDAR ở 30/30 phép so trên 5 bộ có train, 18 có ý nghĩa | 20 |
| 11 | Cần bao nhiêu nhãn? | Khoảng 200 query có nhãn: vượt QuDAR-confidence trong 96–100% số lần rút ở cả 5 bộ | 20 |
| 12 | Không có nhãn cùng bộ thì sao? | Mang trọng số từ bộ khác sang có lợi ở 3 bộ, có hại ở 2 bộ; mô hình học chỉ có lợi ở 2 bộ có 10.000 query train | 21 |

## 1. BM25: code công bố không cho ra số trong paper

Repo dùng `rank_bm25` (BM25Okapi) với tách từ `text.lower().split()` và stemming, không bỏ stopword và không bỏ dấu câu, nên `"cheque?"` và `"cheque"` thành hai từ khác nhau. Thư viện này cũng phải quét từng tài liệu bằng Python cho mỗi query, nên không chạy được với corpus 5 triệu tài liệu.

| Cấu hình | FiQA | SciDocs |
|---|---|---|
| `rank_bm25` của repo | 0.2156 / 0.1678 / 0.1481 | 0.1412 / 0.1390 / 0.1750 |
| bm25s lucene + stopword | **0.3178 / 0.2514 / 0.2299** | **0.1653 / 0.1578 / 0.1830** |
| bm25s robertson + stopword | 0.3192 / 0.2528 / 0.2346 | 0.1659 / 0.1581 / 0.1830 |
| bm25s lucene, không stopword | 0.3173 / 0.2517 / 0.2315 | 0.1637 / 0.1561 / 0.1780 |
| Paper (Table 1) | 0.3135 / 0.2500 / 0.2299 | 0.1655 / 0.1578 / 0.1810 |

bm25s mặc định (lucene, stopword tiếng Anh, PyStemmer) trùng tuyệt đối P@1 trên FiQA và nDCG@10 trên SciDocs. Repo cũng khai báo phụ thuộc `PyStemmer`, stemmer mà bm25s khuyên dùng. Nhiều khả năng thí nghiệm của paper chạy bằng bm25s, còn code công bố đã đổi sang `rank_bm25`.

`scripts/bm25_fast.py` có cả hai lớp: `FastBM25SparseRetriever` (cho điểm giống hệt `rank_bm25`, sai khác dưới 1e-5, dùng để đối chứng) và `BM25sSparseRetriever` (cấu hình khớp paper, dùng cho đồ án).

## 2. Contriever với tiền tố kiểu E5

`DenseEncoder` mặc định thêm `"query: "` / `"passage: "`, vốn là cách dùng của E5, không phải của Contriever. Trên FiQA:

| Cấu hình | OD |
|---|---|
| Có tiền tố (code tác giả) | 0.1492 / 0.1188 / 0.1096 |
| Không tiền tố | 0.1525 / 0.1241 / 0.1235 |
| Paper | 0.1649 / 0.1299 / 0.1173 |

Không cấu hình nào khớp: bỏ tiền tố gần paper hơn ở Recall và nDCG nhưng vượt paper ở P@1. Đồ án giữ code tác giả, vì 2 corpus lớn đã được encode theo cách này và thay đổi không có căn cứ chắc chắn. Độ lệch OD trên 4 bộ: FiQA −10%, SciDocs −15%, Climate-FEVER −19%, HotpotQA −4% (Recall@10).

## 3. Cách encode query mở rộng cho dense (ED)

ED trên FiQA thấp hơn paper 31% trong khi ES (cùng nội dung query mở rộng, qua BM25) chỉ lệch 2%. Đã thử 4 cách encode trên index có sẵn:

| Cách | FiQA | SciDocs |
|---|---|---|
| A – `passage: ` + mở rộng (code tác giả) | 0.1261 / 0.0939 / 0.0741 | 0.1361 / 0.1283 / 0.1460 |
| B – `query: ` + mở rộng | 0.1753 / 0.1198 / 0.0772 | 0.1496 / 0.1428 / 0.1710 |
| C – `passage: ` + câu hỏi + mở rộng | 0.1512 / 0.1180 / 0.1049 | 0.1403 / 0.1333 / 0.1540 |
| D – `query: ` + câu hỏi + mở rộng | 0.2036 / 0.1532 / 0.1281 | 0.1573 / 0.1469 / 0.1640 |
| Paper | 0.1835 / 0.1363 / 0.1049 | 0.1487 / 0.1390 / 0.1570 |

Vì dense của đồ án vốn yếu hơn paper (OD bằng khoảng 0,90 lần ở FiQA và 0,86 lần ở SciDocs), cách encode đúng phải cho ED có cùng tỷ lệ đó. Tỷ lệ Recall@10 so với paper: A 0.69 (FiQA) / 0.92 (SciDocs), B 0.96 / 1.01, C 0.82 / 0.94, D 1.11 / 1.06. Không cách nào nhất quán trên cả hai bộ, nên đồ án giữ cách A. Ở HotpotQA, ED lệch −6%, tương đương OD, nên khoảng hụt lớn ở FiQA và Climate-FEVER (−37%) nhiều khả năng đến từ nội dung query mở rộng (dense nhạy với văn phong hơn BM25), không phải từ cách encode.

## 4. Fusion bền vững với retriever dense yếu

Dù OD và ED thấp hơn paper tới 10–37%, các biến thể QuDAR vẫn lệch paper trong khoảng ±3%, ví dụ Climate-FEVER: QuDAR-confidence 0.2507 so với 0.2473, dù ED chỉ đạt 0.1234 so với 0.1960. Trộn có trọng số tự giảm ảnh hưởng của tín hiệu yếu.

## 5. QuDAR-llm: prompt, thứ tự passage và thiên lệch vị trí

**Khác biệt.** Phụ lục D mô tả 4 passage top-1 được **xáo trộn ngẫu nhiên** rồi chấm bằng Prompt 1 (thang điểm đầy đủ, trả về mảng JSON). `qudar/llm.py` luôn đưa theo thứ tự **cố định ES, OS, ED, OD** với prompt rút gọn trả về 4 số. Ngoài ra, code chỉ gửi trường `text` của passage: trung bình 858 token mỗi query trên FiQA, so với 1.271 trong Table 4 của paper.

**Kết quả theo paper (Prompt 1 + xáo trộn, seed = qid):**

| | FiQA | SciDocs | Climate-FEVER | HotpotQA |
|---|---|---|---|---|
| QuDAR-llm | 0.3281 / 0.2539 / 0.2099 | 0.1899 / 0.1799 / 0.2060 | 0.2567 / 0.2093 / 0.2013 | 0.6704 / 0.6355 / 0.7169 |
| Paper | 0.3639 / 0.2837 / 0.2454 | 0.1920 / 0.1816 / 0.2050 | 0.2583 / 0.2138 / 0.2085 | 0.6883 / 0.6579 / 0.7515 |

**Ablation 2×2 trên FiQA (Recall@10):**

| | Xáo trộn | Thứ tự cố định |
|---|---|---|
| Prompt paper | 0.3281 | 0.3423 |
| Prompt code | 0.3386 | **0.3522** |

Hai yếu tố cộng dồn: thứ tự cố định +0.014, prompt code +0.010. Bản theo code gần paper nhất, gợi ý paper có thể đã chạy theo code công bố thay vì theo mô tả ở Phụ lục D.

**Thiên lệch vị trí.** Điểm trung bình gpt-4o-mini chấm theo vị trí passage trong prompt (chế độ paper, đã xáo trộn nên mỗi nguồn xuất hiện đều ở mọi vị trí):

| Run | Vị trí 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| FiQA test | 3.52 | 3.09 | 2.74 | 2.10 |
| SciDocs test | 3.98 | 3.68 | 3.20 | 2.49 |
| Climate-FEVER test | 2.87 | 2.65 | 2.51 | 2.07 |
| HotpotQA test | 3.12 | 2.56 | 2.48 | 2.27 |

Passage đứng đầu được chấm cao hơn passage đứng cuối 0,8–1,5 điểm trên thang 0–5. Khi xáo trộn, thiên lệch thành nhiễu trong trọng số. Khi cố định, hai passage BM25 luôn đứng đầu và OD luôn đứng cuối (điểm TB của OD trên FiQA: 1.57 khi cố định so với 2.49 khi xáo trộn). Ở FiQA, BM25 lại là tín hiệu mạnh nhất, nên thứ tự cố định vô tình có lợi. Giả thuyết: lợi thế này phụ thuộc dataset; đang kiểm chứng trên 3 bộ còn lại (Step 6.1).

## 6. DAT

Repo không có DAT. Prompt được dựng lại theo mô tả Phụ lục A của bài DAT (Hsu & Tzeng, 2025): chấm top-1 dense và top-1 BM25 của query gốc trên thang 0–5; α = 0.5 nếu cả hai bằng 0, = 1 hoặc 0 nếu chỉ một bên đạt 5, còn lại α = s_dense / (s_dense + s_bm25) làm tròn 0.1. Kết quả khớp paper trong khoảng ±4% (FiQA 0.2894 so với 0.2842, HotpotQA 0.6009 so với 0.5781).

## 7. Ghi chú phương pháp

- **Static UB** theo code tác giả lấy giá trị tốt nhất riêng cho từng chỉ số (Recall tốt nhất và nDCG tốt nhất có thể ở hai trọng số khác nhau). `grid_sweep.py` và `step5_analysis.py` dùng cùng định nghĩa.
- **Static Avg (4 danh sách)** là trung bình trên lưới đầy đủ 11 × 11 × 11 điểm (α, β1, β2).
- **Trọng số tối ưu theo query (Figure 4a, 5):** Recall@10 và nDCG@10 chỉ nhận ít giá trị rời rạc nên nhiều trọng số hòa điểm. Ép chọn một giá trị (ví dụ gần 0.5) tạo đỉnh giả; các hình chia đều phần đếm của mỗi query cho mọi trọng số tối ưu và bỏ các query mà mọi trọng số cho cùng điểm.
- **Đánh giá RAG (Step 7):** cột "Gen." của Table 3 dùng thước đo QA-Eval (Wang et al., 2023a, *Evaluating Open-QA Evaluation*), cần đáp án tham chiếu. HotpotQA có sẵn đáp án trong `metadata.answer` của BEIR; FiQA không có, sẽ dùng passage liên quan trong qrels. Paper không công bố prompt chấm, nên Step 7 là xấp xỉ.

## 8. Ghi chú triển khai

- `run_retrieval.py` gọi lại hàm của tác giả (encoder, chuẩn hóa, fusion, đánh giá), chỉ đổi thứ tự chạy: BM25 xong giải phóng RAM rồi mới chạy dense, dense tìm trên GPU từ shard. Trên dữ liệu giả lập, `eval_results.csv` trùng tuyệt đối với `python -m qudar.cli` (FAISS) và điểm số trùng đến 1e-6 (shard).
- Kaggle/Colab cài sẵn JAX; bm25s tự dùng JAX cho top-k, làm JAX chiếm 75% bộ nhớ GPU và Contriever bị hết bộ nhớ. Đã ép bm25s dùng numpy và đặt `JAX_PLATFORMS=cpu`.
- `run_retrieval.py` kiểm tra mã tài liệu của shard so với corpus trước khi tìm dense, sau một lần chạy Climate-FEVER đọc nhầm shard HotpotQA (OD, ED bằng 0).
- Chỉ số tính bằng numpy trong `grid_sweep.py` trùng với `qudar.fusion` + `pytrec_eval` (sai khác dưới 1e-4, do làm tròn).

## 9. Ablation QuDAR-llm 2×2 trên 4 dataset (Step 6.1)

Recall@10; HotpotQA dùng 1.500 query đầu.

| Dataset | Prompt paper + xáo trộn | Prompt paper + cố định | Prompt code + xáo trộn | Prompt code + cố định |
|---|---|---|---|---|
| FiQA | 0.3281 | 0.3423 | 0.3386 | 0.3522 |
| SciDocs | 0.1899 | 0.1906 | 0.1920 | 0.1889 |
| Climate-FEVER | 0.2567 | 0.2552 | 0.2609 | 0.2591 |
| HotpotQA | 0.6737 | 0.6823 | 0.6927 | 0.6883 |

- Thứ tự cố định có lợi có ý nghĩa ở FiQA (cả hai prompt) và HotpotQA (prompt paper); không có ý nghĩa ở Climate-FEVER; kém hơn ở SciDocs (prompt code). Hiệu ứng phụ thuộc dataset, nên kết quả QuDAR-llm cao trên FiQA của paper có thể gắn với đặc thù của FiQA.
- Giả thuyết "thiên lệch vị trí trùng với tín hiệu mạnh" không được xác nhận ở Climate-FEVER (ES mạnh nhất, đứng đầu khi cố định, nhưng không có lợi).
- Prompt code tốt hơn prompt paper có ý nghĩa ở 5/8 ô theo nDCG@10, rõ nhất ở HotpotQA (+0.019): cách diễn đạt prompt ảnh hưởng nhất quán hơn thứ tự passage.
- Trong bản tái hiện, QuDAR-llm không vượt QuDAR-confidence ở FiQA và SciDocs, và kém hơn về Recall ở HotpotQA (−0.015, có ý nghĩa), khác với paper.

## 10. Khử thiên lệch vị trí

Độ lệch theo vị trí được ước lượng trên tập train (chế độ paper) rồi trừ khỏi điểm tập test trước khi tính trọng số. Kết quả tốt hơn hoặc ngang bằng ở cả 4 bộ, nhưng nhỏ: sau hiệu chỉnh Holm chỉ còn HotpotQA nDCG@10 +0.0016 có ý nghĩa; FiQA +0.0066 Recall@10 (p = 0.019 trước hiệu chỉnh). Khử lệch không đạt mức của thứ tự cố định trên FiQA (0.3347 so với 0.3423).

## 11. Một trọng số tĩnh chọn trên tập train vượt QuDAR

Paper so QuDAR với Static Avg và Static UB (chọn trên tập test, không dùng được trong thực tế), nhưng không so với một trọng số tĩnh chọn trên dữ liệu train. Static-train (điểm lưới có nDCG@10 trung bình cao nhất trên FiQA train + HotpotQA train):

| | FiQA | HotpotQA |
|---|---|---|
| Static-train − QuDAR-confidence | R +0.020, nDCG +0.023 * | R +0.019, nDCG +0.026 * |
| Static-train − QuDAR-llm | R +0.028, nDCG +0.022 * | R +0.034, nDCG +0.027 * |

\* Holm p < 0.05. Trên SciDocs không khác biệt; trên Climate-FEVER, Static-train kém QuDAR-llm (nDCG −0.008, có ý nghĩa). Kết luận: luật theo query của QuDAR có giá trị khi không có dữ liệu train cùng miền; khi có, tinh chỉnh một trọng số tĩnh đơn giản và tốt hơn.

## 12. Học trọng số theo từng query (Step 6, 6.3b)

- **Thiết kế:** LightGBM dự đoán nDCG@10 (đã trừ trung bình theo query) của từng điểm lưới (α, β1, β2) bước 0.2 từ đặc trưng của 4 danh sách và của query; Model A không dùng LLM, Model B thêm 4 điểm LLM và vị trí của chúng. Direct-A/B hồi quy thẳng (α, β1, β2). Bản 6.3b (một lần sửa, thiết kế trước): mô hình có cổng chỉ rời Static-train khi lợi ích dự đoán vượt ngưỡng chọn trên 20% validation tách từ train; bộ đặc trưng không phụ thuộc thang đo (A-sf, B-sf); p-value hiệu chỉnh Holm.
- **Cùng miền:** mô hình học vượt QuDAR-confidence và QuDAR-llm có ý nghĩa, nhưng phần lớn mức tăng đã có ở Static-train. So với Static-train, chỉ HotpotQA có cải thiện có ý nghĩa: Direct-B nDCG +0.0095 và Recall +0.0035, Gated B nDCG +0.0035. FiQA không có khác biệt có ý nghĩa sau Holm.
- **Zero-shot:** mọi mô hình học ngang hoặc kém Static-train; ở Climate-FEVER kém có ý nghĩa (Gated A nDCG −0.011). Đặc trưng không phụ thuộc thang đo giảm thiệt hại (Gated B-sf −0.005) nhưng không khắc phục được. Ngoài miền, nhiều query vượt ngưỡng hơn (Gated A-sf: 83% ở SciDocs so với 55% ở FiQA), tức lợi ích dự đoán bị thổi phồng khi dữ liệu khác phân phối.
- **Validation:** cổng chỉ cải thiện nDCG@10 validation 0.001–0.004 so với Static-train; với mô hình scale-free huấn luyện trên FiQA, validation chọn ngưỡng vô cùng (không rời trọng số tĩnh).
- **Feature importance:** đặc trưng quan trọng nhất là chính trọng số được chấm (w_OS khoảng 35% gain); đặc trưng của query hàng đầu là độ đồng thuận giữa các danh sách (Jaccard top-10 OS–ES, OS top-1 có trong danh sách khác).
- **Kết luận:** khoảng cách tới Query-wise UB (ví dụ FiQA 0.4944) gần như chưa được khai thác bằng đặc trưng rẻ; đây là giới hạn chính và hướng mở cho nghiên cứu tiếp.

## 13. Đánh giá RAG end-to-end (Step 7)

500 query ngẫu nhiên mỗi bộ (seed 0); Qwen2.5-7B-Instruct sinh câu trả lời từ top-5 passage (tối đa 180 từ mỗi passage, giải mã tham lam) và tự chấm Yes/No theo prompt QA-Eval (Wang et al., 2023a). Đáp án chuẩn: HotpotQA `metadata.answer`; FiQA tối đa 3 passage liên quan trong qrels. Prompt chấm, số passage và cách chọn 500 query không được paper công bố, nên điểm Gen tuyệt đối cao hơn paper (FiQA khoảng 0.5 so với 0.14–0.18) và chỉ so sánh thứ tự giữa các phương pháp.

- Recall@10 và Gen tương quan mạnh trên 13 phương pháp (Pearson 0.91 FiQA, 0.98 HotpotQA).
- HotpotQA (Holm p < 0.05): QuDAR-confidence hơn OS +0.078 và hơn Retriever-Type Fusion +0.060; QuDAR-llm hơn OS +0.092; QuDAR-llm và QuDAR-confidence không khác biệt. Khớp kết luận của paper (+17% so với retriever đơn tốt nhất, paper khoảng 20%).
- FiQA: không có khác biệt có ý nghĩa giữa các phương pháp thực tế (0.49–0.55); chỉ các phương pháp thuần dense thấp rõ (0.36–0.42).
- Static-train và Model B không khác biệt có ý nghĩa so với QuDAR sau hiệu chỉnh Holm.
- Query-wise UB đạt Gen 0.600 / 0.588, so với khoảng 0.50–0.53: khoảng trống của retrieval cũng là khoảng trống của RAG.

## 14. Demo trên laptop tái lập đúng thí nghiệm (Step 8)

Demo (`demo/`) chạy lại toàn bộ pipeline cho FiQA trên CPU của laptop: bm25s dựng lại chỉ mục từ corpus, Contriever encode query, tìm dense bằng tích vô hướng trên ma trận embedding lấy từ FAISS index của Step 1, rồi tính trọng số của mọi phương pháp bằng cùng công thức. `demo/selftest.py` chạy 648 query test (query mở rộng của Step 1, điểm LLM của Step 4) theo hai cách, truy xuất trực tiếp và dùng danh sách đã lưu của Step 2:

| | Kết quả |
|---|---|
| Top-10 trùng với Step 2 (trung bình mỗi query) | OS 10/10, OD 10/10, ES 10/10, ED 9,96/10 |
| Model A / Model B chọn cùng điểm lưới như Step 6 | 648/648, 648/648 |
| Recall@10 / nDCG@10 của 13 phương pháp, trực tiếp so với đã lưu | trùng đến 4 chữ số |
| Thời gian | 385 giây cho 648 query × 2 cách (CPU laptop) |

Một số giá trị đối chiếu (Recall@10 / nDCG@10): QuDAR-confidence 0.3354 / 0.2525 và QuDAR-llm 0.3281 / 0.2539 (trùng Step 4–5), QuDAR-llm khử thiên lệch 0.3347 / 0.2585 (trùng mục 10), Model A 0.3524 / 0.2789, Model B 0.3515 / 0.2860, Query-wise UB 0.4944 / 0.4320. Static-train cho 0.3556 so với 0.3555 trong bảng Step 6; mức chênh 0.0001 nhiều khả năng do hai đoạn code xếp các tài liệu hòa điểm theo thứ tự khác nhau. Vài tài liệu lệch ở ED nhiều khả năng cũng do hòa điểm (tích vô hướng tính lại trên CPU có thể đổi thứ tự hai tài liệu điểm gần bằng nhau so với lần tìm ở Step 2), và không làm đổi chỉ số trung bình.

Kết luận: phần xếp hạng của đồ án là tất định và tái lập được trên máy khác với cùng dữ liệu đầu vào. Phần không tất định nằm ở LLM: query mở rộng sinh trực tiếp trong demo (gpt-4o-mini, hoặc LLaMA 4-bit qua Ollama) khác cấu hình của Step 1, nên câu tự nhập không có số đối chiếu.

## 15. Mở rộng lên 8 bộ và 2 LLM chấm (Step 10a–10c)

Mục tiêu: kiểm tra các kết luận của đồ án có đứng vững ngoài 4 bộ của paper và ngoài một LLM chấm không, và tăng số bộ có dữ liệu train từ 2 lên 5 để kiểm chứng mục 11. Bốn bộ mới giữ cơ cấu của paper (2 bộ nhỏ, 2 bộ hàng triệu tài liệu); paper không báo cáo các bộ này nên không có số để so.

| Bộ mới | Tài liệu (từ / tài liệu) | Query test (tài liệu liên quan / query) | Query train dùng | Ghi chú |
|---|---|---|---|---|
| SciFact | 5.183 (215) | 300 (1,13) | 809 | kiểm chứng khẳng định khoa học |
| NFCorpus | 3.633 (234) | 323 (38,2) | 2.590 | y khoa, nhiều tài liệu liên quan mỗi query |
| DBPedia-entity | 4.635.922 (50) | 400 (38,2, nhãn 0–2) | không có | truy xuất thực thể |
| FEVER | 5.416.568 (98) | 6.666 (1,19) | 10.000 đầu trong 109.810 | kiểm chứng khẳng định trên Wikipedia |

- **FEVER và Climate-FEVER dùng chung corpus.** So từng tài liệu: 5.416.536 tài liệu trùng cả id lẫn văn bản, 32 tài liệu chỉ có ở FEVER (10 ví dụ được ghi lại đều là tên có dấu như `Eugène_Ionesco`, gợi ý khác cách chuẩn hóa Unicode), 57 chỉ có ở Climate-FEVER, không tài liệu nào trùng id mà khác văn bản (`results/step10/fever_vs_cf_summary.json`). `build_fever_shards.py` dựng shard FEVER từ shard Climate-FEVER và chỉ encode thêm 32 tài liệu, tiết kiệm khoảng 2,7 giờ GPU.
- **Query mở rộng** cho 21.088 query (test và train) cùng thiết lập Step 1C; lần này lưu thêm log-prob từng token (`expand_queries.py --logprobs 5`), văn bản sinh ra không đổi.
- **LLM chấm thứ hai: Qwen2.5-7B-Instruct** (vLLM, T4 ×2), cùng prompt, cùng định dạng cache với gpt-4o-mini; passage bị cắt ở `--passage-tokens`, câu trả lời sai định dạng được hỏi lại. Mỗi LLM chấm ghi vào thư mục riêng; không gắn hai bộ điểm vào cùng một notebook vì tên file cache trùng nhau. Lý do thêm: OpenAI tắt `gpt-4o-mini` ngày 14/01/2027, sau đó chỉ bản mô hình mở chạy lại được. Toàn bộ lượt chấm gpt-4o-mini của Step 10 tốn 4,42 USD và đã được lưu (`results/step10/llm_gpt/`).
- Ablation prompt × thứ tự của HotpotQA và FEVER chạy trên 1.500 query đầu, như mục 9.

**Chất lượng từng danh sách** (Recall@10), dùng để giải thích các mục sau:

| Bộ | OS | OD | ES | ED | ES so với OS | Xếp hạng |
|---|---|---|---|---|---|---|
| FiQA | 0.3178 | 0.1492 | 0.2047 | 0.1261 | −36% | OS > ES > OD > ED |
| SciDocs | 0.1653 | 0.0950 | 0.1459 | 0.1361 | −12% | OS > ES > ED > OD |
| Climate-FEVER | 0.1733 | 0.0999 | 0.2576 | 0.1234 | +49% | ES > OS > ED > OD |
| HotpotQA | 0.5995 | 0.3849 | 0.4047 | 0.4526 | −32% | OS > ED > ES > OD |
| SciFact | 0.8187 | 0.7176 | 0.7826 | 0.6256 | −4% | OS > ES > OD > ED |
| NFCorpus | 0.1471 | 0.1244 | 0.1394 | 0.0881 | −5% | OS > ES > OD > ED |
| DBPedia-entity | 0.1924 | 0.1789 | 0.1432 | 0.1793 | −26% | OS > ED > OD > ES |
| FEVER | 0.6538 | 0.5015 | 0.7051 | 0.5400 | +8% | ES > OS > ED > OD |

- Danh sách tốt nhất luôn là một danh sách BM25 (OS ở 6 bộ, ES ở Climate-FEVER và FEVER). Hai danh sách Contriever là hai danh sách yếu nhất ở 6/8 bộ (trừ HotpotQA, DBPedia-entity).
- Đoạn mở rộng thay query gốc (ES so với OS) chỉ có lợi ở 2/8 bộ, đều là kiểm chứng khẳng định trên Wikipedia; hại nặng nhất ở FiQA và hai bộ hỏi về thực thể (HotpotQA, DBPedia-entity), nơi đoạn mở rộng làm mất tên riêng của query.
- ES tốt hơn OD ở 7/8 bộ (trừ DBPedia-entity); ED tốt hơn ES chỉ ở 2/8 bộ (HotpotQA, DBPedia-entity). Hai so sánh này dùng ở mục 18.

## 16. Các họ kiểm định trên 8 bộ (Step 10d)

Mỗi phép so: bootstrap ghép cặp 10.000 lần và t-test ghép cặp trên giá trị theo từng query, lấy p lớn hơn; Holm trong từng họ. Các họ được chốt trong script (`judge_significance.py`, `step10d2.py`) trước khi xem kết quả. Holm tính trên cả 8 bộ nên chặt hơn bản 4 bộ (`results/step10/sig/`, Step 10-B2).

| Họ | Phép so | Số phép | Có ý nghĩa | File |
|---|---|---|---|---|
| F1 | QuDAR-llm − Equal (2 prompt × 8 bộ × 2 chỉ số) | 32 mỗi LLM chấm | gpt-4o-mini 6 (5 dương), Qwen 4 (2 dương) | `sig8/sig_tests.md` |
| F2 | Qwen − gpt-4o-mini, prompt paper | 16 | 3, đều nghiêng về gpt-4o-mini | như trên |
| F3 | Ablation prompt × thứ tự (4 phép so × 8 bộ × 2 chỉ số) | 64 mỗi LLM chấm | gpt-4o-mini 5, Qwen 11 | như trên |
| F4 | Thiên lệch vị trí (kiểm định hoán vị 10.000 lần) | 16 mỗi LLM chấm | 32/32 | `sig8/position_tests.md` |
| S0 | QuDAR-confidence − Equal | 16 | 2 | `static8/sig_tests.md` |
| S1 | Static-train cùng bộ − Equal / QuDAR-confidence / QuDAR-llm, 5 bộ có train | 30 | 18 | như trên |
| S2 | Static-train chọn trên các bộ khác − QuDAR-confidence | 16 | 8 (5 dương) | như trên |
| S3 | Model A / Model B − Static-train, cùng học trên 5 bộ train gộp | 32 | nDCG@10: 5 (3 dương); Recall@10: xem file | như trên |

Chỉ số tính lại theo từng query khớp `eval_results.csv` / `llm_eval.csv`: lệch tối đa 0.0007 (họ F) và 0.0004 (họ S). Nguyên nhân là cách phá hòa điểm: khi chia đều 0,25, nhiều tài liệu có cùng điểm trộn; `pytrec_eval` xếp theo mã tài liệu, hàm numpy của đồ án theo thứ tự danh sách. Mỗi phép so dùng cùng một hàm cho cả hai vế nên kết luận không đổi. Static-train cùng bộ ở Step 10d-2 cho Recall@10 FiQA 0.3584, trùng `Static-train [fiqa]` của Step 6b; HotpotQA 0.7041, bằng Static UB của bộ này.

## 17. QuDAR-llm so với Equal trên 8 bộ (họ F1, F2)

Chênh lệch nDCG@10 so với Equal; thứ tự passage được xáo như paper. \* có ý nghĩa sau Holm; † 1.500 query đầu, so với Equal trên cùng tập.

| | FiQA | SciDocs | C-FEVER | HotpotQA | SciFact | NFCorpus | DBPedia | FEVER |
|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini, prompt paper | +0.003 | +0.001 | +0.005 | +0.004 | −0.001 | +0.002 | −0.002 | +0.016\* |
| Qwen2.5-7B, prompt paper | −0.004 | +0.002 | −0.002 | −0.005\* | −0.006 | −0.007 | +0.003 | +0.006\* |
| gpt-4o-mini, prompt code | +0.011 | +0.003 | +0.010\* | +0.020\* † | +0.006 | +0.007 | +0.007 | +0.029\* † |
| Qwen2.5-7B, prompt code | +0.009 | +0.002 | +0.001 | +0.010 † | +0.010 | +0.005 | +0.009 | +0.022\* † |

- **Prompt paper** (làm đúng theo Phụ lục D): trong 16 phép so mỗi LLM, gpt-4o-mini thắng 1 (FEVER nDCG +0.0164), thua 1 (HotpotQA Recall −0.0123); Qwen thắng 1 (FEVER nDCG +0.0063), thua 2 (HotpotQA Recall −0.0124, nDCG −0.0049). Bảy bộ còn lại với gpt-4o-mini chỉ chênh −0.002 đến +0.005 nDCG@10. Kết quả "không vượt Equal" của mục 9 (4 bộ) vì vậy mở rộng được sang 7/8 bộ.
- **FEVER là ngoại lệ**: bộ duy nhất mà cách làm theo paper có ích với cả hai LLM chấm. Paper không đánh giá FEVER; đây là kết quả mới, cần báo cáo cùng với 7 bộ không có lợi.
- **Prompt code**: gpt-4o-mini thắng 4, không thua (Climate-FEVER nDCG +0.0099 và Recall +0.0105, HotpotQA nDCG +0.0200, FEVER nDCG +0.0293); Qwen thắng 1 (FEVER nDCG +0.0216).
- **Qwen so với gpt-4o-mini** (F2, prompt paper): 3/16 có ý nghĩa, cả 3 nghiêng về gpt-4o-mini: Climate-FEVER Recall −0.0127, HotpotQA nDCG −0.0086, FEVER nDCG −0.0101. Không phép nào nghiêng về Qwen. Qwen2.5-7B kém một chút, đủ dùng làm bản chạy lại sau khi gpt-4o-mini bị tắt.

## 18. Thiên lệch vị trí và thứ tự cố định (họ F3 phần thứ tự, F4)

**Điểm trung bình theo vị trí trong prompt** (prompt paper, thứ tự đã xáo theo seed = qid, nên mỗi nguồn xuất hiện đều ở mọi vị trí; vị trí 1 / 2 / 3 / 4):

| Bộ | gpt-4o-mini | Qwen2.5-7B |
|---|---|---|
| FiQA | 3.52 / 3.09 / 2.74 / 2.10 | 2.14 / 2.48 / 3.12 / 1.77 |
| SciDocs | 3.98 / 3.68 / 3.20 / 2.49 | 2.73 / 2.75 / 3.58 / 1.85 |
| Climate-FEVER | 2.87 / 2.65 / 2.51 / 2.07 | 2.22 / 2.52 / 3.37 / 1.71 |
| HotpotQA | 3.12 / 2.56 / 2.48 / 2.27 | 2.80 / 3.19 / 3.43 / 2.03 |
| SciFact | 3.73 / 3.50 / 3.56 / 3.23 | 3.25 / 3.51 / 3.71 / 2.51 |
| NFCorpus | 3.54 / 2.96 / 2.91 / 2.74 | 2.70 / 2.80 / 3.08 / 1.90 |
| DBPedia-entity | 3.39 / 2.93 / 2.70 / 2.27 | 2.98 / 3.14 / 3.60 / 2.20 |
| FEVER | 3.05 / 2.69 / 2.72 / 2.51 | 2.78 / 3.25 / 3.65 / 2.24 |
| Trung bình 8 bộ | 3.40 / 3.01 / 2.85 / 2.46 | 2.70 / 2.95 / 3.44 / 2.03 |

- Hình dạng ổn định theo LLM, không theo bộ dữ liệu: gpt-4o-mini cao nhất ở vị trí 1 ở cả 8 bộ; Qwen cao nhất ở vị trí 3 ở cả 8 bộ; với prompt paper, cả hai chấm vị trí 4 thấp nhất ở cả 8 bộ.
- Kiểm định hoán vị (xáo bốn điểm giữa bốn vị trí trong từng query, 10.000 lần; hợp lệ vì thứ tự vốn được xáo ngẫu nhiên): 32/32 có ý nghĩa sau Holm (2 LLM × 2 prompt × 8 bộ). Với 10.000 lần hoán vị, p nhỏ nhất là khoảng 0.0001, nên p Holm của hầu hết các ô bằng 0.0016.

**Thứ tự cố định dồn trọng số.** Code luôn đưa ES ở vị trí 1, OS ở 2, ED ở 3, OD ở 4. Trọng số LLM trung bình khi cố định trừ khi xáo, prompt paper (`sig8/weights_by_source.md`):

| Bộ | gpt: ES | gpt: OD | Qwen: ED | Qwen: OD |
|---|---|---|---|---|
| FiQA | +0.042 | −0.051 | +0.051 | −0.043 |
| SciDocs | +0.064 | −0.077 | +0.077 | −0.070 |
| Climate-FEVER | +0.010 | −0.036 | +0.091 | −0.056 |
| HotpotQA † | +0.035 | −0.035 | +0.060 | −0.058 |
| SciFact | +0.013 | −0.014 | +0.045 | −0.047 |
| NFCorpus | +0.033 | −0.037 | +0.043 | −0.041 |
| DBPedia-entity | +0.034 | −0.034 | +0.061 | −0.055 |
| FEVER † | +0.025 | −0.025 | +0.067 | −0.064 |

† 1.500 query đầu. Với prompt code, ES của gpt-4o-mini và ED của Qwen vẫn tăng ở cả 8 bộ, mức nhỏ hơn. Với Qwen, ES thường cũng mất trọng số khi cố định (ví dụ Climate-FEVER 0.270 → 0.235).

**Hệ quả lên truy xuất** (cố định − xáo, cả hai prompt, 32 phép so mỗi LLM):

- gpt-4o-mini: thứ tự cố định tốt hơn ở 23/32; 2 phép có ý nghĩa, đều ở FiQA (nDCG +0.0173 với prompt paper, +0.0124 với prompt code).
- Qwen: thứ tự cố định kém hơn ở 22/32; 3 phép có ý nghĩa: Climate-FEVER với prompt code (Recall −0.0124, nDCG −0.0082) và FEVER với prompt paper (nDCG −0.0173).
- So với mục 9 (4 bộ, chỉ gpt-4o-mini, chưa hiệu chỉnh Holm): lợi thế của thứ tự cố định ở HotpotQA với prompt paper không còn ý nghĩa sau Holm; chỉ FiQA còn lại.

**Giải thích (chưa kiểm định riêng).** Dồn trọng số có lợi hay có hại tùy danh sách được dồn vào tốt đến đâu (mục 15): ES tốt hơn OD ở 7/8 bộ, nên gpt-4o-mini dồn từ OD sang ES thường có lợi; ED tốt hơn ES chỉ ở 2/8 bộ, nên Qwen dồn sang ED (và lấy bớt của ES) thường có hại. Giả thuyết ở mục 5 và 9 ("lợi thế của thứ tự cố định phụ thuộc dataset") vì vậy được làm rõ thêm: nó phụ thuộc vào việc thiên lệch vị trí riêng của LLM chấm có trùng với danh sách mạnh của bộ đó hay không. Thứ tự cố định trong code không phải lựa chọn trung tính, và paper không nêu chi tiết này.

## 19. Prompt code so với prompt paper (họ F3 phần prompt)

- Cùng thứ tự xáo: 32/32 chênh lệch dương (2 LLM × 8 bộ × 2 chỉ số). Tính cả thứ tự cố định: 56/64.
- 11 phép có ý nghĩa sau Holm, đều nghiêng về prompt code: gpt-4o-mini 3 (HotpotQA Recall +0.0190 và nDCG +0.0191, FEVER nDCG +0.0130); Qwen 8 (FiQA, HotpotQA, NFCorpus, FEVER). Mức chênh lớn nhất đều ở Qwen với thứ tự cố định: nDCG@10 HotpotQA +0.0247 và FiQA +0.0219, Recall@10 FiQA +0.0303.
- Prompt code thu hẹp khoảng chênh giữa vị trí được chấm cao nhất và thấp nhất ở 15/16 trường hợp (ngoại lệ: gpt-4o-mini, Climate-FEVER 0.79 → 0.92):

| Bộ | gpt-4o-mini: paper → code | Qwen: paper → code |
|---|---|---|
| FiQA | 1.42 → 0.65 | 1.36 → 0.48 |
| SciDocs | 1.49 → 0.71 | 1.74 → 0.36 |
| Climate-FEVER | 0.79 → 0.92 | 1.66 → 1.17 |
| HotpotQA | 0.85 → 0.36 | 1.40 → 0.47 |
| SciFact | 0.50 → 0.39 | 1.21 → 0.55 |
| NFCorpus | 0.79 → 0.49 | 1.18 → 0.39 |
| DBPedia-entity | 1.12 → 0.57 | 1.40 → 0.65 |
| FEVER | 0.54 → 0.45 | 1.41 → 0.72 |

  Với prompt code, thiên lệch vẫn có ý nghĩa ở cả 16 trường hợp, chỉ nhỏ hơn. Đây là quan sát mô tả: prompt code ít thiên lệch hơn có thể là một lý do nó tốt hơn, nhưng chưa được kiểm chứng là nguyên nhân duy nhất. Paper không báo cáo kết quả với prompt rút gọn và không nói bảng kết quả dùng prompt nào.

## 20. Static-train trên 5 bộ có train và số nhãn cần thiết (họ S0, S1)

Static-train cùng bộ: thử cả 1.331 điểm lưới trên train của chính bộ đó, chọn điểm có nDCG@10 trung bình cao nhất, dùng chung cho mọi query test. nDCG@10:

| Phương pháp | FiQA | HotpotQA | SciFact | NFCorpus | FEVER |
|---|---|---|---|---|---|
| Equal | 0.2512 | 0.6318 | 0.7231 | 0.3531 | 0.6355 |
| QuDAR-confidence | 0.2525 | 0.6366 | 0.7240 | 0.3529 | 0.6368 |
| QuDAR-llm (gpt-4o-mini, prompt paper) | 0.2539 | 0.6355 | 0.7226 | 0.3551 | 0.6520 |
| Static-train cùng bộ | 0.2843\* | 0.6634\* | 0.7323 | 0.3603 | 0.6982\* |
| Static UB (chọn trên test) | 0.2855 | 0.6634 | 0.7418 | 0.3655 | 0.6982 |

\* vượt cả ba phương pháp QuDAR có ý nghĩa sau Holm (S1), ở cả nDCG@10 và Recall@10.

- 30/30 chênh lệch dương; 18/30 có ý nghĩa, đúng là 18 phép so ở FiQA, HotpotQA và FEVER. SciFact và NFCorpus cao hơn 0.005–0.01 nhưng chỉ có 300 và 323 query test.
- Ở HotpotQA và FEVER, điểm chọn trên train trùng điểm tốt nhất trên test; FiQA đạt 99,6% Static UB. Trọng số tối ưu của một bộ dữ liệu ổn định giữa train và test.
- **Trọng số được chọn** (`static8/static_points.md`) dồn 60–80% vào hai danh sách BM25; ED nhận 0–0.09, trừ HotpotQA 0.24:

| Bộ | OS | OD | ES | ED | OS + ES |
|---|---|---|---|---|---|
| FiQA | 0.42 | 0.24 | 0.28 | 0.06 | 70% |
| HotpotQA | 0.42 | 0.16 | 0.18 | 0.24 | 60% |
| SciFact | 0.56 | 0.27 | 0.14 | 0.03 | 70% |
| NFCorpus | 0.42 | 0.21 | 0.28 | 0.09 | 70% |
| FEVER | 0.32 | 0.20 | 0.48 | 0.00 | 80% |

  Khớp với mục 15: danh sách tốt nhất luôn là BM25. Equal luôn dành 50% cho Contriever.
- **QuDAR-confidence gần như là Equal** (S0): chỉ 2/16 phép so có ý nghĩa, cùng ở HotpotQA và đều dưới 0.005 (Recall +0.0026, nDCG +0.0048). Khớp với DESA (arXiv 2608.15851). Điều cần học không phải là điều chỉnh nhỏ theo từng query mà là việc bộ dữ liệu này nên tin BM25 nhiều hơn.

**Đường cong số nhãn** (mô tả, không kiểm định). Chọn trọng số tĩnh từ n query train rút ngẫu nhiên (100 lần rút mỗi n), đo nDCG@10 trên test; mỗi ô là tỉ lệ lần rút vượt QuDAR-confidence (`static8/label_budget.md`):

| Bộ | n = 10 | 20 | 50 | 100 | 200 | 500 | n = 200: trung vị / QuDAR-confidence |
|---|---|---|---|---|---|---|---|
| FiQA | 69% | 82% | 94% | 100% | 100% | 100% | 0.2823 / 0.2525 |
| HotpotQA | 64% | 86% | 100% | 100% | 100% | 100% | 0.6592 / 0.6366 |
| SciFact | 22% | 38% | 60% | 77% | 96% | 100% | 0.7308 / 0.7240 |
| NFCorpus | 41% | 61% | 78% | 91% | 99% | 100% | 0.3587 / 0.3529 |
| FEVER | 77% | 98% | 98% | 100% | 100% | 100% | 0.6915 / 0.6368 |

Khoảng 200 query có nhãn là đủ để vượt QuDAR-confidence trong 96–100% số lần rút ở cả 5 bộ; 50 query đủ cho FiQA, HotpotQA, FEVER nhưng chưa đủ cho SciFact, NFCorpus. Với n = 200, trung vị đã gần mức của toàn bộ train (FEVER 0.6915 so với 0.6982). Đây là so điểm trung bình trên test của từng lần rút; chưa kiểm định riêng cho từng lần.

## 21. Không có nhãn cùng bộ: mang trọng số sang và mô hình học (họ S2, S3)

nDCG@10, \* có ý nghĩa sau Holm:

| Bộ | Static-train từ các bộ khác − QuDAR-confidence (S2) | Recall@10 (S2) | Model A − Static-train (S3) | Model B − Static-train (S3) |
|---|---|---|---|---|
| FiQA | +0.027\* | +0.015 | +0.005 | +0.004 |
| SciDocs | −0.004 | −0.005 | −0.007\* | −0.003 |
| Climate-FEVER | +0.021\* | +0.024\* | −0.006 | −0.007 |
| HotpotQA | −0.029\* | −0.026\* | +0.016\* | +0.023\* |
| SciFact | −0.001 | −0.003 | −0.001 | +0.007 |
| NFCorpus | +0.009 | +0.004 | −0.007 | −0.005 |
| DBPedia-entity | −0.010 | −0.015\* | −0.029\* | −0.012 |
| FEVER | +0.024\* | +0.023\* | +0.005 | +0.017\* |

- **S2, mang trọng số sang:** trọng số chọn trên train của các bộ khác (không dùng nhãn của chính bộ đó). Có lợi có ý nghĩa ở FiQA, Climate-FEVER, FEVER; có hại có ý nghĩa ở HotpotQA (cả hai chỉ số) và DBPedia-entity (Recall). Trọng số mang sang dồn khoảng 40–48% vào ES vì các bộ train khác, nhất là FEVER, có ES rất tốt; ở HotpotQA và DBPedia-entity, ES lại kém OS 32% và 26% (mục 15).
- So với mục 11 (Static-train chọn trên FiQA + HotpotQA, kém QuDAR-llm ở Climate-FEVER): giờ trọng số mang sang có lợi ở Climate-FEVER, nhiều khả năng một phần nhờ train của FEVER (chung corpus, cùng dạng câu khẳng định). Kết luận về Climate-FEVER vì vậy phụ thuộc vào việc có bộ train "họ hàng" hay không.
- **S3, mô hình học** (LightGBM như mục 12; Model A 55 đặc trưng về query và 4 danh sách, Model B thêm 8 đặc trưng từ điểm gpt-4o-mini, tổng 63; mô hình và Static-train cùng học trên 5 bộ train gộp): có lợi có ý nghĩa chỉ ở HotpotQA và FEVER, hai bộ có 10.000 query train; có hại có ý nghĩa ở SciDocs và DBPedia-entity (Model A), hai bộ không có train. Ở FEVER, Model B đạt nDCG@10 0.7088, cao hơn Static UB 0.6982, tức cao hơn mọi trọng số cố định: chỗ duy nhất học theo query thật sự vượt được trọng số cố định (quan sát, chưa nằm trong họ kiểm định).
- **Tổng hợp:** có khoảng 200 nhãn cùng bộ thì dùng trọng số cố định; có hàng chục nghìn nhãn thì mô hình học theo query mới bắt đầu có lợi; không có nhãn nào thì cả hai đều rủi ro, và QuDAR-confidence cũng chỉ ngang Equal.

## 22. Giới hạn của Step 10

- SciFact và NFCorpus chỉ có khoảng 300 query test: các chênh lệch dương ở đây chưa đủ để khẳng định.
- Ablation prompt × thứ tự của HotpotQA và FEVER chỉ chạy trên 1.500 query đầu; so sánh khoảng chênh vị trí ở hai bộ này (mục 19) là so giữa toàn bộ test (prompt paper) và 1.500 query đầu (prompt code).
- Đường cong số nhãn (mục 20) và kết quả Model B vượt Static UB ở FEVER (mục 21) là mô tả, chưa kiểm định.
- Giải thích lợi/hại của thứ tự cố định theo chất lượng danh sách (mục 18) phù hợp với phần lớn các bộ nhưng chưa có phép kiểm định riêng.
- QuDAR-llm khử thiên lệch (mục 10) chưa chạy lại trong các họ kiểm định 8 bộ.
- Mọi kết luận đúng trong thiết lập của paper: Contriever không giám sát với tiền tố kiểu E5 (mục 2). Với retriever dense mạnh hơn, trọng số tối ưu có thể dời về phía dense; câu hỏi "trọng số cố định hay thích ứng" vẫn cần kiểm tra lại trong thiết lập đó.
