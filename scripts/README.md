# qudar-patches

Phần bổ sung cho pipeline của tác giả (`kaist-dmlab/QuDAR`), dùng cho cả đồ án (Step 1–8) và phần mở rộng lên 8 bộ dữ liệu với 2 LLM chấm (Step 10). Trên Kaggle, cả thư mục được tải lên dataset [qudar-patches](https://www.kaggle.com/datasets/karrychuong/qudar-patches); notebook tìm script bằng `glob`, nên không phụ thuộc đường dẫn cố định. Các notebook Step 10d clone thẳng repo này thay vì gắn `qudar-patches`.

## Truy xuất và chuẩn bị dữ liệu

| File | Dùng ở | Việc |
|---|---|---|
| `bm25_fast.py` + `apply_patch.py` | Step 1A, 2, 10c-1 | Thay `rank_bm25` của repo bằng `BM25sSparseRetriever` (bm25s mặc định: lucene, stopword tiếng Anh, PyStemmer), cấu hình khớp Table 1 của paper; cần `pip install bm25s`. Lớp `FastBM25SparseRetriever` (giống hệt rank_bm25) giữ lại để đối chứng |
| `encode_shards.py` | Step 1B, 10b | Encode corpus bằng Contriever thành shard fp16 (`shard_XXX.npy` + `shard_XXX.ids.json`), chạy tiếp được nếu bị ngắt. Step 10b dùng cho cả SciFact, NFCorpus và DBPedia-entity |
| `build_index_from_shards.py` | tùy chọn | Ghép shard thành FAISS index cùng định dạng với `qudar.build_index`. Các notebook hiện tại tìm thẳng trên shard bằng `run_retrieval.py --dense-shards` |
| `expand_queries.py` | Step 1C, 10b | Sinh query mở rộng bằng LLaMA 3.1-8B-Instruct (vLLM, dự phòng transformers), prompt Phụ lục G, giải mã tham lam, tối đa 256 token, output `{qid: passage}`; `--max-queries N` lấy N query đầu (train HotpotQA, FEVER: 10.000). Từ Step 10b: đọc query không cần nạp corpus (cùng thứ tự với `run_retrieval`), và `--logprobs K` lưu log-prob của từng token sinh ra kèm top-K vào `logprobs/<dataset>_<split>/part_NNN.npz`; văn bản sinh ra không đổi |
| `check_datasets.py` | trước Step 2 | Kiểm tra cấu trúc và số lượng trong các dataset Kaggle |
| `run_retrieval.py` | Step 2, 10c-1 | Chạy 4 danh sách OS/OD/ES/ED (top-1000) + mọi kết quả không cần LLM, giống hệt `qudar.cli` nhưng tiết kiệm RAM; dense tìm bằng FAISS (`--dense-faiss`, FiQA và SciDocs) hoặc trên GPU từ shard (`--dense-shards`, 6 bộ còn lại); kiểm tra id của shard khớp corpus trước khi tìm |
| `grid_sweep.py` | Step 3, 5, 10c-1 | Lưới trọng số (α, β1, β2) bước 0.1, tức 1.331 điểm, cho từng query: Static Avg/UB 4 danh sách, cận trên theo query, `grid_metrics.npz`. Hàm `query_metrics` được `judge_significance.py` dùng lại để tính chỉ số từng query |
| `prepare_new_datasets.py` | Step 10a | Tải SciFact, NFCorpus, FEVER, DBPedia-entity từ bản sao BEIR trên Hugging Face, chuyển sang bố cục BEIR; thống kê corpus/qrels (id trùng, số query, mức nhãn, số tài liệu liên quan mỗi query, id trong qrels không có trong corpus) và đối chiếu với Table 1 của bài BEIR; so corpus FEVER với Climate-FEVER từng tài liệu (`--cf-corpus`) để quyết định có dùng lại shard Contriever không; ước tính thời gian mở rộng query |
| `build_fever_shards.py` | Step 10b | Dựng shard Contriever của FEVER từ 11 shard Climate-FEVER: bỏ 57 id chỉ có ở Climate-FEVER, encode thêm 32 tài liệu chỉ có ở FEVER bằng đúng định dạng văn bản và pooling của `encode_shards.py`, kiểm tra id của shard khớp corpus FEVER, báo cáo lệch chuẩn hóa Unicode (NFC/NFD). Tiết kiệm khoảng 2,7 giờ GPU so với encode lại |

## LLM chấm điểm

| File | Dùng ở | Việc |
|---|---|---|
| `llm_score.py` | Step 4, 6.1, 10-B1, 10c-2, 10c-3 | LLM chấm top-1 của 4 danh sách: `paper` (Prompt 1 + xáo trộn theo seed = qid), `code` (bản công bố: thứ tự cố định ES, OS, ED, OD, prompt rút gọn), `paper_fixed`, `code_shuffle` (ablation 2×2), `dat` (baseline DAT); `--max-queries` để chạy một phần (ablation HotpotQA, FEVER: 1.500 query đầu). Cache từng dòng trong `llm_<mode>_<name>.jsonl` (chạy tiếp được), rồi ghi `llm_scores_<mode>_<name>.csv` và chỉ số vào `llm_eval.csv`. Từ Step 10: `--mode` nhận nhiều chế độ trong một lần chạy; `--backend vllm --model Qwen/Qwen2.5-7B-Instruct` chấm bằng LLM mã nguồn mở (cùng prompt, cùng định dạng cache, cắt mỗi passage ở `--passage-tokens`, hỏi lại khi trả lời sai định dạng, `llm_eval.csv` có thêm cột `judge`). **Mỗi LLM chấm ghi vào một thư mục `--out` riêng**: các script phân tích tìm file theo tên, nên hai bộ điểm không được nằm chung một input |

## Phân tích và kiểm định

| File | Dùng ở | Việc |
|---|---|---|
| `step5_analysis.py` | Step 5, 10-B1 | Bảng so với paper (Table 1–2, 8), Figure 2–5 (chia đều các điểm hòa), thống kê thiên lệch vị trí |
| `step6.py` | Step 6 | 63 đặc trưng theo query, mô hình học trọng số A/B (LightGBM hồi quy bề mặt nDCG) và Direct, Static-train, QuDAR-llm khử thiên lệch, bảng ablation LLM, kiểm định bootstrap + t-test. Các hàm đọc run, chọn điểm tĩnh, huấn luyện và dự đoán bề mặt, ước lượng thiên lệch vị trí được `step10d2.py` dùng lại |
| `step6b.py` | Step 6.3b | Bản sửa duy nhất (thiết kế trước): mô hình có cổng quanh Static-train (ngưỡng chọn trên 20% validation), đặc trưng không phụ thuộc thang đo (A-sf, B-sf), hiệu chỉnh Holm |
| `judge_significance.py` | Step 10-B2 (4 bộ), 10d-1 (8 bộ) | Kiểm định ghép cặp cho hai LLM chấm (gpt-4o-mini, Qwen2.5-7B): bootstrap 10.000 lần và t-test ghép cặp (lấy p lớn hơn, như Step 6), hiệu chỉnh Holm trong từng họ kiểm định đặt trước: F1 QuDAR-llm so với Equal, F2 so hai LLM chấm, F3 ablation 2×2 (thứ tự × prompt), F4 thiên lệch vị trí (kiểm định hoán vị 10.000 lần, hợp lệ vì `llm_score.py` xáo thứ tự theo seed = qid). Danh sách bộ truyền qua `--full` / `--ablation`; với 8 bộ, Holm tính trên cả 8 bộ trong mỗi họ. Thêm bảng mô tả `weights_by_source` (trọng số LLM và điểm trung bình mỗi danh sách nhận được theo LLM chấm × bộ × chế độ). Tính lại chỉ số từng query bằng `grid_sweep.query_metrics` và đối chiếu với `llm_eval.csv` / `eval_results.csv` (`check_means.csv`) |
| `step10d2.py` | Step 10d-2 | Trọng số tĩnh chọn trên train so với QuDAR trên 8 bộ test và 5 bộ train (FiQA, HotpotQA, SciFact, NFCorpus, FEVER). Static-train cùng bộ / chọn trên các bộ khác / chọn trên 5 bộ gộp; Model A, B của Step 6 học trên 5 bộ gộp. Họ kiểm định đặt trước: S0 QuDAR-confidence − Equal (16), S1 Static-train cùng bộ − Equal / QuDAR-confidence / QuDAR-llm (30), S2 Static-train từ bộ khác − QuDAR-confidence (16), S3 Model A, B − Static-train gộp (32); kiểm định và Holm dùng hàm của `judge_significance.py`. Thêm đường cong số nhãn (mô tả): chọn trọng số tĩnh từ n query train ngẫu nhiên, n = 10 … toàn bộ, 100 lần rút. Chỉ gắn điểm gpt-4o-mini: file điểm Qwen trùng tên sẽ bị đọc nhầm. Output: `table_main`, `sig_tests`, `static_points`, `label_budget` (.csv/.md), `fig_label_budget.png`, `check.csv`, `summary.json` |
| `rag_eval.py` | Step 7 | RAG end-to-end: Qwen2.5-7B-Instruct (vLLM) sinh câu trả lời từ top-5 passage và tự chấm theo QA-Eval; 500 query × 13 phương pháp; kiểm định cặp (Table 3) |
| `pack_demo_data.py` | Step 8 | Gom dữ liệu cho demo: corpus và query FiQA, embedding lấy từ FAISS index, 4 danh sách Step 2, query mở rộng, điểm LLM, mô hình Step 6, điểm Static-train → `demo_data.zip` (khoảng 110 MB) |

## Thứ tự phụ thuộc

- **Đồ án:** `expand_queries` → `run_retrieval` → `grid_sweep` → `llm_score` → `step5_analysis` → `step6` → `step6b` → `rag_eval` → `pack_demo_data`. Bản thân demo nằm ở [`../demo/`](../demo/).
- **Step 10:** `prepare_new_datasets` → `encode_shards` / `build_fever_shards` và `expand_queries` → `run_retrieval` → `grid_sweep` → `llm_score` (gpt-4o-mini và Qwen, mỗi LLM một thư mục) → `judge_significance` (Step 10d-1) và `step10d2` (Step 10d-2). Hai script cuối chạy độc lập với nhau; `step10d2` cần thêm lưới trọng số (`grid_metrics.npz`) của cả 13 run.
- **Import giữa các script:** `judge_significance` dùng `grid_sweep.query_metrics`; `step10d2` dùng hàm của `step6` (đọc run, chọn điểm tĩnh, mô hình bề mặt, thiên lệch vị trí) và `judge_significance` (`paired_tests`, `holm`).

## Đã kiểm tra

- `encode_shards.py` + `build_index_from_shards.py` (CPU, mô hình nhỏ giả lập): top-50 trùng 100% và điểm giống hệt `qudar.build_index`.
- `run_retrieval.py` với `--dense-faiss` cho `eval_results.csv` trùng tuyệt đối với `python -m qudar.cli` (dữ liệu giả lập 3.000 tài liệu); với `--dense-shards` điểm số trùng đến 1e-6.
- `grid_sweep.py`: chỉ số tính bằng numpy trùng `qudar.fusion` + `pytrec_eval` (sai khác < 1e-4 do làm tròn).
- `llm_score.py`: đã test các chế độ bằng `--dry-run`; backend vllm đã chạy thật trên 8 bộ với Qwen2.5-7B.
- `prepare_new_datasets.py`: số tài liệu và số query test của 4 bộ mới khớp Table 1 của bài BEIR (`results/step10/step10a_report.json`).
- `build_fever_shards.py`: id của shard FEVER trùng id corpus FEVER (5.416.568 tài liệu); `run_retrieval.py` kiểm tra lại điều này trước khi tìm dense ở Step 10c-1.
- `judge_significance.py`: chạy hết các nhánh trên dữ liệu giả lập (2 bộ, 2 LLM chấm, có query thiếu điểm); điểm theo vị trí tính lại từ CSV khớp Step 5 (FiQA, gpt-4o-mini: 3,52 / 3,09 / 2,74 / 2,10). Trên 8 bộ, chỉ số tính lại lệch `llm_eval.csv` / `eval_results.csv` tối đa 0.0007, do cách phá hòa điểm khi trộn đều.
- `step10d2.py`: Equal và QuDAR-confidence tính lại lệch `eval_results.csv` tối đa 0.0004; Static-train cùng bộ cho Recall@10 FiQA 0.3584, trùng `Static-train [fiqa]` của Step 6b.
