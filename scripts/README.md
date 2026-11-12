# qudar-patches

Phần đồ án bổ sung cho pipeline của tác giả (`kaist-dmlab/QuDAR`). Trên Kaggle, cả thư mục được tải lên dataset [qudar-patches](https://www.kaggle.com/datasets/karrychuong/qudar-patches); notebook tìm script bằng `glob`, nên không phụ thuộc đường dẫn cố định.

| File | Dùng ở | Việc |
|---|---|---|
| `bm25_fast.py` + `apply_patch.py` | Step 1A, Step 2 | Thay `rank_bm25` của repo bằng `BM25sSparseRetriever` (bm25s mặc định: lucene, stopword tiếng Anh, PyStemmer), cấu hình khớp Table 1 của paper; cần `pip install bm25s`. Lớp `FastBM25SparseRetriever` (giống hệt rank_bm25) giữ lại để đối chứng |
| `encode_shards.py` | Step 1B | Encode corpus lớn bằng Contriever thành shard fp16, chạy tiếp được nếu bị ngắt |
| `expand_queries.py` | Step 1C | Sinh query mở rộng bằng LLaMA 3.1-8B-Instruct (vLLM, dự phòng transformers), prompt Phụ lục G, output `{qid: passage}` |
| `build_index_from_shards.py` | Step 2 | Ghép shard thành FAISS index cùng định dạng với `qudar.build_index` |
| `check_datasets.py` | Trước Step 2 | Kiểm tra cấu trúc và số lượng trong các dataset Kaggle |
| `run_retrieval.py` | Step 2 | Chạy 4 danh sách OS/OD/ES/ED + mọi kết quả không cần LLM, giống hệt `qudar.cli` nhưng tiết kiệm RAM; tìm dense trên GPU từ shard; kiểm tra shard khớp corpus |
| `grid_sweep.py` | Step 3, 5 | Lưới trọng số (α, β1, β2) bước 0.1 cho từng query: Static Avg/UB 4 danh sách, cận trên theo query |
| `llm_score.py` | Step 4, 6.1 | gpt-4o-mini chấm top-1: `paper` (Prompt 1 + xáo trộn), `code` (bản công bố), `paper_fixed`, `code_shuffle` (ablation 2×2), `dat` (baseline DAT); `--max-queries` để chạy một phần |
| `step5_analysis.py` | Step 5 | Bảng so với paper (Table 1–2, 8), Figure 2–5 (chia đều các điểm hòa), thống kê thiên lệch vị trí |
| `step6.py` | Step 6 | 63 đặc trưng theo query, mô hình học trọng số A/B (LightGBM hồi quy bề mặt nDCG) và Direct, Static-train, QuDAR-llm khử thiên lệch, bảng ablation LLM, kiểm định bootstrap + t-test |
| `step6b.py` | Step 6.3b | Bản sửa duy nhất (thiết kế trước): mô hình có cổng quanh Static-train (ngưỡng chọn trên 20% validation), đặc trưng không phụ thuộc thang đo (A-sf, B-sf), hiệu chỉnh Holm |
| `rag_eval.py` | Step 7 | RAG end-to-end: Qwen2.5-7B-Instruct (vLLM) sinh câu trả lời từ top-5 passage và tự chấm theo QA-Eval; 500 query × 13 phương pháp; kiểm định cặp (Table 3) |
| `pack_demo_data.py` | Step 8 | Gom dữ liệu cho demo: corpus và query FiQA, embedding lấy từ FAISS index, 4 danh sách Step 2, query mở rộng, điểm LLM, mô hình Step 6, điểm Static-train → `demo_data.zip` (khoảng 110 MB) |

Thứ tự phụ thuộc: `expand_queries` → `run_retrieval` → `grid_sweep` → `llm_score` → `step5_analysis` → `step6` → `step6b` → `rag_eval` → `pack_demo_data`. Bản thân demo nằm ở [`../demo/`](../demo/).

Đã kiểm tra (CPU, mô hình nhỏ giả lập): `encode_shards.py` + `build_index_from_shards.py` cho top-50 trùng 100% và điểm giống hệt `qudar.build_index`.

`run_retrieval.py` với `--dense-faiss` cho `eval_results.csv` trùng tuyệt đối với `python -m qudar.cli` (dữ liệu giả lập 3.000 tài liệu); với `--dense-shards` điểm số trùng đến 1e-6.
`grid_sweep.py`: chỉ số tính bằng numpy trùng `qudar.fusion` + `pytrec_eval` (sai khác < 1e-4 do làm tròn). `llm_score.py`: đã test cả 3 chế độ bằng `--dry-run`.
