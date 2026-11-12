# QuDAR Replication – Truy xuất thông tin (CS419)

Tái hiện và mở rộng bài báo **QuDAR: Query-Wise Dual-Perspective Adaptive Retrieval** (ACL 2026, KAIST) cho đồ án môn Truy xuất thông tin (CS419), UIT. Đồ án thuộc bài toán **"Chọn công thức tính giá trị xếp hạng tốt nhất"**: QuDAR trộn 4 danh sách kết quả (BM25 / Contriever × query gốc / query mở rộng) bằng trọng số riêng cho từng câu truy vấn.

- Paper: QuDAR, ACL 2026 (Long Papers)
- Code gốc của tác giả: <https://github.com/kaist-dmlab/QuDAR> (không chép vào repo này; notebook tự `git clone` khi chạy)
- Thực hiện: `Bùi Huy Chương` – `24520230`

## Trạng thái

| Step | Nội dung | Trạng thái |
|---|---|---|
| 0 | Chuẩn bị dữ liệu, môi trường | ✅ |
| 1 | BM25 (bm25s), encode Contriever, sinh query mở rộng bằng LLaMA 3.1-8B-Instruct | ✅ |
| 2 | 4 danh sách truy xuất OS / OD / ES / ED (top-1000) | ✅ |
| 3 | Fusion và baseline: Static Avg/UB, RRF, Equal, Confidence, DAT | ✅ |
| 4 | QuDAR-llm (gpt-4o-mini) + ablation prompt × thứ tự | ✅ |
| 5 | Phân tích: Table 8, Figure 2–5 | ✅ |
| 6 | Mở rộng: học trọng số theo query, khử thiên lệch vị trí, kiểm định thống kê | ✅ |
| 7 | Đánh giá RAG end-to-end (Table 3) | ✅ |
| 8 | Demo chạy trên laptop (FiQA) | ✅ |
| 9 | Báo cáo | ⏳ |

## Quy trình truy xuất trong đồ án

| Khâu | Sparse (BM25) | Dense (Contriever) |
|---|---|---|
| Tài liệu | 4 corpus BEIR, ghép `title + text` | như bên cạnh |
| Phân tích tài liệu | chữ thường, tách từ regex, bỏ stopword tiếng Anh, stemming Snowball (PyStemmer) | WordPiece, tối đa 512 token, tiền tố `passage: `, mean pooling, chuẩn hóa L2 → vector 768 chiều |
| Lập chỉ mục | inverted index của bm25s (ma trận thưa term × tài liệu, trọng số BM25 tính sẵn, k1 = 1.5, b = 0.75) | FAISS IndexFlatIP (FiQA, SciDocs); shard fp16 (Climate-FEVER, HotpotQA) |
| Phân tích câu truy vấn | cùng bộ tách từ + **mở rộng query bằng LLaMA 3.1-8B-Instruct** | tiền tố `query: ` (gốc), `passage: ` (mở rộng) |
| Truy xuất chỉ mục | posting list → điểm BM25, top-1000 | tích vô hướng toàn bộ, top-1000 |
| **Xếp hạng (trọng tâm)** | chuẩn hóa min-max rồi trộn 4 danh sách: Static, RRF, Equal, Confidence, LLM, DAT, trọng số học được (Step 6) | |

## Dữ liệu và mô hình

| Thành phần | Lựa chọn |
|---|---|
| Dataset | FiQA (57.638 tài liệu / 648 query test), SciDocs (25.657 / 1.000), Climate-FEVER (5.416.593 / 1.535), HotpotQA (5.233.329 / 7.405); train: FiQA 5.500, HotpotQA 10.000 query đầu |
| Sparse | bm25s, phương pháp lucene, stopword tiếng Anh, PyStemmer |
| Dense | `facebook/contriever` |
| Mở rộng query | `meta-llama/Llama-3.1-8B-Instruct`, prompt Phụ lục G, giải mã tham lam (greedy), tối đa 256 token |
| LLM chấm điểm | `gpt-4o-mini`, temperature 0 |
| Học trọng số (Step 6) | LightGBM |
| RAG (Step 7) | `Qwen/Qwen2.5-7B-Instruct` sinh câu trả lời và tự chấm theo QA-Eval |
| Chỉ số | Recall@10, nDCG@10, Precision@1 (pytrec_eval) |

## Cấu trúc repo

```
scripts/      phần bổ sung cho pipeline của tác giả (xem bảng dưới)
notebooks/    notebook Kaggle/Colab, đánh số theo thứ tự chạy
results/      kết quả nhỏ: CSV đánh giá, điểm LLM, bảng và hình cuối
demo/         demo Gradio chạy trên laptop (xem demo/README.md)
docs/         findings.md: các chỗ code công bố lệch paper và kết quả ablation
```

| Script | Step | Việc |
|---|---|---|
| `bm25_fast.py`, `apply_patch.py` | 1–2 | Thay `rank_bm25` của repo gốc bằng bm25s (cấu hình khớp paper, chạy được 5 triệu tài liệu) |
| `encode_shards.py`, `build_index_from_shards.py` | 1 | Encode corpus lớn thành shard fp16, chạy tiếp được khi bị ngắt |
| `expand_queries.py` | 1 | Sinh query mở rộng (vLLM, dự phòng transformers) |
| `check_datasets.py` | 2 | Kiểm tra cấu trúc và số lượng các dataset Kaggle |
| `run_retrieval.py` | 2 | 4 danh sách truy xuất + mọi kết quả không cần LLM; cho `eval_results.csv` trùng với `qudar.cli`, tiết kiệm RAM, tìm dense trên GPU |
| `grid_sweep.py` | 3, 5 | Lưới 1.331 tổ hợp (α, β1, β2) cho từng query: Static Avg/UB 4 danh sách, cận trên theo query |
| `llm_score.py` | 3–4 | gpt-4o-mini chấm top-1: `paper`, `code`, `paper_fixed`, `code_shuffle`, `dat` |
| `step5_analysis.py` | 5 | Bảng so với paper, Figure 2–5, thống kê thiên lệch vị trí |
| `step6.py` | 6 | Đặc trưng, mô hình học trọng số A/B (LightGBM), QuDAR-llm khử thiên lệch, ablation LLM, kiểm định thống kê |
| `step6b.py` | 6 | Bản sửa duy nhất (đăng ký trước): mô hình có cổng quanh Static-train, đặc trưng không phụ thuộc thang đo, hiệu chỉnh Holm |
| `rag_eval.py` | 7 | RAG end-to-end: Qwen2.5-7B sinh câu trả lời từ top-5 và tự chấm theo QA-Eval, kiểm định cặp |
| `pack_demo_data.py` | 8 | Gom dữ liệu FiQA, embedding, danh sách Step 2, điểm LLM, mô hình Step 6 thành `demo_data.zip` cho demo |

## Dataset trên Kaggle

File lớn (index, shard, `retrieval_result.json`, `grid_metrics.npz`) không nằm trong repo mà ở các dataset sau:

| Dataset | Nội dung |
|---|---|
| [qudar-beir-data](https://www.kaggle.com/datasets/karrychuong/qudar-beir-data) | 4 bộ BEIR (corpus, query, qrels) |
| [qudar-patches](https://www.kaggle.com/datasets/karrychuong/qudar-patches) | thư mục `scripts/` |
| [qudar-index-small](https://www.kaggle.com/datasets/karrychuong/qudar-index-small) | FAISS index FiQA, SciDocs |
| [qudar-shards-climate-fever](https://www.kaggle.com/datasets/karrychuong/qudar-shards-climate-fever), [qudar-shards-hotpotqa](https://www.kaggle.com/datasets/karrychuong/qudar-shards-hotpotqa) | 11 shard embedding fp16 mỗi bộ |
| [qudar-expansions](https://www.kaggle.com/datasets/karrychuong/qudar-expansions) | query mở rộng (test 4 bộ, train FiQA + HotpotQA) |
| [qudar-runs](https://www.kaggle.com/datasets/karrychuong/qudar-runs) | output Step 2 |
| [qudar-grid](https://www.kaggle.com/datasets/karrychuong/qudar-grid) | output `grid_sweep.py` |
| [qudar-llm](https://www.kaggle.com/datasets/karrychuong/qudar-llm) | điểm LLM, QuDAR-llm, DAT |
| [qudar-llm-ablation](https://www.kaggle.com/datasets/karrychuong/qudar-llm-ablation) | ablation prompt × thứ tự trên 4 bộ (Step 6.1) |
| [qudar-step6](https://www.kaggle.com/datasets/karrychuong/qudar-step6) | output `step6.py`: mô hình LightGBM, trọng số đã chọn, độ lệch vị trí |

## Cách chạy lại

Mọi notebook chạy trên Kaggle (Colab với `01a`), cần bật Internet. Secret: `HF_TOKEN` (LLaMA), `OPENAI_API_KEY` (Step 4, 6.1).

| Notebook | Máy | Ghi chú |
|---|---|---|
| `00_prepare_data` | CPU | tải BEIR, tạo `qudar-beir-data` |
| `01a_bm25_dense_check_colab` | Colab T4 | index FiQA, SciDocs; chẩn đoán BM25 / OD / ED |
| `01b_encode_climate-fever`, `01b_encode_hotpotqa` | T4 ×2 | chạy song song trên 2 tài khoản |
| `01c_query_expansion_test`, `01c_query_expansion_train` | T4 ×2 | vLLM, LLaMA 3.1-8B-Instruct fp16 |
| `02_retrieval_*` | T4 ×1 | FiQA + SciDocs / Climate-FEVER / HotpotQA |
| `03_grid_sweep` | CPU |  |
| `04_llm_scoring_test`, `04_llm_scoring_train`, `04_llm_ablation_fiqa` | CPU | tổng chi phí OpenAI khoảng 5 USD |
| `05_analysis` | CPU | |
| `06a_llm_ablation_3datasets` | CPU | ablation 2×2 trên SciDocs, Climate-FEVER, HotpotQA (1.500 query) |
| `06b_step6_learned_weighting`, `06c_step6b_gated_scalefree` | CPU | học trọng số, khử thiên lệch, kiểm định |
| `07_rag_eval` | T4 ×2 | vLLM, Qwen2.5-7B-Instruct, 500 query × 2 bộ × 13 phương pháp |
| `08_pack_demo_data` | CPU | tạo `demo_data.zip` cho demo |

## Kết quả chính

Recall@10 trên tập test, **của đồ án / paper** (bảng đầy đủ với nDCG@10 và P@1: [`results/step5_analysis/table_main.md`](results/step5_analysis/table_main.md)).

| Phương pháp | FiQA | SciDocs | Climate-FEVER | HotpotQA |
|---|---|---|---|---|
| OS (BM25, query gốc) | 0.3178 / 0.3135 | 0.1653 / 0.1655 | 0.1733 / 0.1750 | 0.5995 / 0.6039 |
| OD (Contriever, query gốc) | 0.1492 / 0.1649 | 0.0950 / 0.1112 | 0.0999 / 0.1235 | 0.3849 / 0.4011 |
| ES (BM25, query mở rộng) | 0.2047 / 0.2085 | 0.1459 / 0.1497 | 0.2576 / 0.2598 | 0.4047 / 0.4151 |
| ED (Contriever, query mở rộng) | 0.1261 / 0.1835 | 0.1361 / 0.1487 | 0.1234 / 0.1960 | 0.4526 / 0.4838 |
| DAT | 0.2894 / 0.2842 | 0.1624 / 0.1614 | 0.1679 / 0.1699 | 0.6009 / 0.5781 |
| Static UB (4 danh sách) | 0.3631 / 0.3715 | 0.1925 / 0.1917 | 0.2929 / 0.2972 | 0.7041 / 0.7110 |
| QuDAR-simple (RRF) | 0.3145 / 0.3219 | 0.1877 / 0.1893 | 0.2422 / 0.2411 | 0.6700 / 0.6741 |
| QuDAR-simple (Equal) | 0.3333 / 0.3402 | 0.1901 / 0.1906 | 0.2504 / 0.2464 | 0.6828 / 0.6924 |
| QuDAR-confidence | 0.3354 / 0.3438 | 0.1907 / 0.1920 | 0.2507 / 0.2473 | 0.6853 / 0.6962 |
| QuDAR-llm | 0.3281 / 0.3639 | 0.1899 / 0.1920 | 0.2567 / 0.2583 | 0.6704 / 0.6883 |
| Query-wise UB (4 danh sách) | 0.4944 / 0.5011 | 0.2749 / 0.2745 | 0.3734 / 0.3859 | 0.7708 / 0.7891 |

Tóm tắt:
- Các biến thể QuDAR không dùng LLM và các cận trên lệch paper trong khoảng ±3%; độ lệch lớn tập trung ở retriever dense (OD, ED).
- Cận trên khi chọn trọng số riêng cho từng query cao hơn trọng số tĩnh tốt nhất 9–43% (Recall@10), xác nhận động lực của paper.
- Code công bố và mô tả trong paper khác nhau ở nhiều chỗ, chi tiết trong [`docs/findings.md`](docs/findings.md).


![Figure 2–3](results/step5_analysis/fig2_3_static_sweeps.png)
![Figure 4b](results/step5_analysis/fig4b_upper_bounds.png) 
![Figure 4a](results/step5_analysis/fig4a_optimal_hist.png)  
![Figure 5](results/step5_analysis/fig5_optimal_heatmap.png) 

## Kết quả mở rộng (Step 6)

Recall@10 / nDCG@10 trên tập test. Static-train = một bộ trọng số cố định chọn trên tập train (FiQA + HotpotQA). Chi tiết: [`results/step6b/`](results/step6b/), [`docs/findings.md`](docs/findings.md) mục 9–12.

| Phương pháp | FiQA | SciDocs (zero-shot) | Climate-FEVER (zero-shot) | HotpotQA |
|---|---|---|---|---|
| QuDAR-confidence | 0.3354 / 0.2525 | 0.1907 / 0.1798 | 0.2507 / 0.2043 | 0.6853 / 0.6366 |
| QuDAR-llm | 0.3281 / 0.2539 | 0.1899 / 0.1799 | 0.2567 / 0.2093 | 0.6704 / 0.6355 |
| **Static-train** | **0.3555 / 0.2754** | 0.1917 / 0.1808 | 0.2473 / 0.2012 | **0.7039 / 0.6621** |
| Direct-B (học, có LLM) | 0.3587 / 0.2803 | 0.1939 / 0.1823 | 0.2462 / 0.2026 | **0.7074 / 0.6715** |
| Gated B (học, có LLM) | 0.3529 / 0.2850 | 0.1899 / 0.1766 | 0.2334 / 0.1894 | 0.7014 / 0.6656 |
| Query-wise UB (oracle) | 0.4944 / 0.4321 | 0.2749 / 0.2810 | 0.3734 / 0.3471 | 0.7708 / 0.7637 |

- Khi có dữ liệu train cùng miền, một bộ trọng số tĩnh chọn trên train vượt QuDAR-confidence và QuDAR-llm có ý nghĩa thống kê (FiQA, HotpotQA; Holm p < 0.05).
- Học trọng số theo từng query chỉ thêm được một chút trên HotpotQA (Direct-B so với Static-train: nDCG +0.0095, Recall +0.0035, có ý nghĩa) và không tổng quát hóa sang SciDocs, Climate-FEVER.
- gpt-4o-mini thiên lệch vị trí mạnh ở cả 6 run; thứ tự passage cố định chỉ có lợi trên FiQA, còn prompt của code công bố tốt hơn prompt trong paper khá nhất quán.

![Thiên lệch vị trí](results/step6/fig6_position_bias.png)

## Đánh giá RAG (Step 7)

Gen. = tỷ lệ câu trả lời đúng của Qwen2.5-7B-Instruct trên 500 query ngẫu nhiên mỗi bộ, khi đưa top-5 của từng phương pháp. Prompt chấm và cách chọn query của paper không được công bố nên **chỉ so thứ tự giữa các phương pháp**, không so số tuyệt đối với paper. Bảng đầy đủ: [`results/step7/`](results/step7/), [`docs/findings.md`](docs/findings.md) mục 13.

| Phương pháp | FiQA Gen. | HotpotQA Gen. |
|---|---|---|
| OS | 0.508 | 0.454 |
| QuDAR-confidence | 0.494 | 0.532 |
| QuDAR-llm | 0.508 | 0.546 |
| Static-train | 0.534 | 0.540 |
| Model B (học, có LLM) | 0.552 | 0.508 |
| Query-wise UB (oracle) | 0.600 | 0.588 |

- Recall@10 và Gen. tương quan mạnh trên 13 phương pháp (Pearson 0.91 FiQA, 0.98 HotpotQA).
- HotpotQA tái hiện kết luận của paper: QuDAR hơn retriever đơn tốt nhất +17% và hơn trộn một chiều +13% (Holm p < 0.05). FiQA không có khác biệt có ý nghĩa giữa các phương pháp thực tế.

## Demo (Step 8)

Trang Gradio chạy trên laptop (CPU) cho FiQA: chọn câu hỏi test hoặc tự nhập, xem 4 danh sách, trọng số mà 16 phương pháp gán, kết quả sau khi trộn, thời gian từng khâu và câu trả lời RAG. Câu hỏi test chạy hoàn toàn offline. Hướng dẫn cài: [`demo/README.md`](demo/README.md).

`demo/selftest.py` truy xuất trực tiếp trên laptop toàn bộ 648 query test và khớp với Step 2 và Step 6: top-10 trùng 10/10 ở OS, OD, ES và 9,96/10 ở ED; Recall@10 / nDCG@10 của mọi phương pháp trùng đến 4 chữ số; Model A, B chọn đúng điểm lưới như Step 6 ở 648/648 query ([`results/step8/selftest.csv`](results/step8/selftest.csv)).

## Ghi nhận

Dữ liệu từ BEIR; code pipeline gốc, prompt và số liệu so sánh từ paper QuDAR và repo của tác giả. Các script trong `scripts/` là phần đồ án bổ sung.
