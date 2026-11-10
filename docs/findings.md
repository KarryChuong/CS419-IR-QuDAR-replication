# Phát hiện khi tái hiện QuDAR

Ghi lại những chỗ code công bố (`kaist-dmlab/QuDAR`) khác mô tả trong paper, các thí nghiệm đã chạy để kiểm chứng, và quyết định của đồ án. Mọi số liệu là trên tập test, dạng **Recall@10 / nDCG@10 / P@1** nếu không ghi khác.

## Tóm tắt

| # | Thành phần | Code công bố | Paper / bằng chứng | Quyết định của đồ án |
|---|---|---|---|---|
| 1 | BM25 | `rank_bm25`, tách từ bằng `.split()`, không bỏ stopword | Kết quả thấp hơn paper khoảng 30%; bm25s mặc định khớp gần như tuyệt đối | Dùng bm25s |
| 2 | Contriever | Tiền tố `query: ` / `passage: ` kiểu E5 | OD thấp hơn paper 4–19%; bỏ tiền tố cũng không khớp | Giữ code tác giả |
| 3 | ED (dense + query mở rộng) | Encode đoạn mở rộng như passage | FiQA thấp hơn paper 31%; 4 cách encode đều không khớp nhất quán | Giữ code tác giả |
| 4 | QuDAR-llm | Thứ tự passage cố định (ES, OS, ED, OD), prompt rút gọn | Phụ lục D: xáo trộn, Prompt 1 đầy đủ; LLM thiên lệch vị trí mạnh | Chạy theo paper, báo cáo thêm ablation 2×2 |
| 5 | DAT | Không có trong repo | Prompt dựng lại từ mô tả Phụ lục A bài DAT | Khớp paper ±4% |

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
