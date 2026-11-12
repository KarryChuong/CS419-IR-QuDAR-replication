# qudar-patches

Bổ sung cho repo kaist-dmlab/QuDAR (đồ án CS419). Thư mục này đã được biến đổi thành Kaggle Dataset `qudar-patches`.

| File | Dùng ở | Việc |
|---|---|---|
| `bm25_fast.py` + `apply_patch.py` | Step 1A, Step 2 | Thay `rank_bm25` của repo bằng `BM25sSparseRetriever` (bm25s mặc định: lucene, stopword tiếng Anh, PyStemmer) — cấu hình khớp Table 1 của paper; cần `pip install bm25s`. Lớp `FastBM25SparseRetriever` (giống hệt rank_bm25) giữ lại để so sánh |
| `encode_shards.py` | Step 1B | Encode corpus lớn bằng Contriever thành shard fp16, chạy tiếp được nếu bị ngắt |
| `build_index_from_shards.py` | Step 2 | Ghép shard thành FAISS index cùng định dạng với `qudar.build_index` |
| `check_datasets.py` | Trước Step 2 | Kiểm tra cấu trúc và số lượng trong 6 dataset |
| `run_retrieval.py` | Step 2 | Chạy 4 danh sách OS/OD/ES/ED + mọi kết quả không cần LLM, giống hệt `qudar.cli` nhưng tiết kiệm RAM; tìm dense trên GPU từ shard |
| `grid_sweep.py` | Step 3 + 5 | Lưới trọng số 2 chiều (α, β1, β2) cho từng query: Static Avg/UB 4 danh sách, cận trên theo query, nhãn tối ưu cho hướng (c) |
| `llm_score.py` | Step 4 | gpt-4o-mini chấm top-1: `paper` (Prompt 1 + xáo trộn), `code` (bản công bố), `dat` (baseline DAT) |
| `step6.py` | Step 6 | Đặc trưng, mô hình học trọng số A/B (LightGBM), QuDAR-llm khử thiên lệch, bảng ablation LLM, kiểm định thống kê |
| `step6b.py` | Step 6 | Mô hình có cổng quanh Static-train, đặc trưng không phụ thuộc thang đo, hiệu chỉnh Holm |
| `rag_eval.py` | Step 7 | RAG end-to-end: Qwen2.5-7B sinh câu trả lời từ top-k passage và tự chấm theo QA-Eval (Table 3) |
| `expand_queries.py` | Step 1C | Sinh query mở rộng bằng LLaMA 3.1-8B-Instruct, prompt Phụ lục G, output `{qid: passage}` |

Đã kiểm tra (CPU, mô hình nhỏ giả lập): `encode_shards.py` + `build_index_from_shards.py` cho top-50 trùng 100% và điểm giống hệt `qudar.build_index`.

`run_retrieval.py` với `--dense-faiss` cho `eval_results.csv` trùng tuyệt đối với `python -m qudar.cli` (dữ liệu giả lập 3.000 tài liệu); với `--dense-shards` điểm số trùng đến 1e-6.
`grid_sweep.py`: chỉ số tính bằng numpy trùng `qudar.fusion` + `pytrec_eval` (sai khác < 1e-4 do làm tròn). `llm_score.py`: đã test cả 3 chế độ bằng `--dry-run`.
