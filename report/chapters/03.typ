= HỆ THỐNG VÀ HIỆN THỰC

Chương này mô tả cách đồ án hiện thực sáu khâu của quy trình truy xuất trên bốn bộ dữ liệu: dữ liệu, môi trường, cách hiện thực từng khâu, cách tổ chức mã nguồn và các bước kiểm tra tính đúng đắn của pipeline.

== Dữ liệu

Đồ án dùng đúng bốn bộ dữ liệu của bài báo, lấy từ BEIR @thakur2021beir (@tab:datasets). FiQA @maia2018fiqa gồm câu hỏi tư vấn tài chính; SciDocs @cohan2020specter là bài toán tìm bài báo khoa học được trích dẫn; Climate-FEVER @diggelmann2020climatefever là kiểm chứng phát biểu về khí hậu trên Wikipedia; HotpotQA @yang2018hotpotqa là hỏi đáp nhiều bước trên Wikipedia. Hai bộ sau có corpus hơn 5 triệu tài liệu.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    table.header([*Bộ dữ liệu*], [*Số tài liệu*], [*Câu truy vấn test*], [*Câu truy vấn train dùng*], [*Lĩnh vực*]),
    [FiQA], [57.638], [648], [5.500], [Tài chính, câu hỏi tư vấn],
    [SciDocs], [25.657], [1.000], [—], [Bài báo khoa học],
    [Climate-FEVER], [5.416.593], [1.535], [—], [Kiểm chứng phát biểu],
    [HotpotQA], [5.233.329], [7.405], [10.000 câu đầu], [Hỏi đáp nhiều bước],
  ),
  caption: [Bốn bộ dữ liệu thực nghiệm],
) <tab:datasets>

Tập train chỉ dùng cho các thí nghiệm mở rộng ở Chương 5 (chọn trọng số tĩnh, huấn luyện mô hình học trọng số). SciDocs và Climate-FEVER không có tập train trong BEIR nên được dùng làm kịch bản *zero-shot*.

== Môi trường và tài nguyên

Mã nguồn gốc của tác giả (`kaist-dmlab/QuDAR`) được `git clone` trực tiếp trong mỗi notebook. Phần đồ án bổ sung nằm trong thư mục `scripts/` của kho GitHub và được tải lên Kaggle dưới dạng dataset. @tab:runtime liệt kê các bước chính và phần cứng dùng để chạy.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.header([*Bước*], [*Máy*], [*Ghi chú*]),
    [Chuẩn bị dữ liệu], [CPU], [Tải BEIR],
    [Chỉ mục FiQA, SciDocs], [Colab T4], [Kèm chẩn đoán BM25 và dense],
    [Encode Climate-FEVER / HotpotQA], [Kaggle T4 ×2], [Chạy song song trên 2 tài khoản],
    [Sinh câu truy vấn mở rộng], [Kaggle T4 ×2], [vLLM @kwon2023vllm, LLaMA 3.1-8B fp16],
    [Truy xuất 4 danh sách], [Kaggle T4], [FiQA + SciDocs / Climate-FEVER / HotpotQA],
    [Quét lưới 1.331 trọng số], [CPU], [Mọi câu truy vấn, mọi bộ],
    [Chấm điểm bằng gpt-4o-mini], [CPU + API], [Tổng chi phí OpenAI khoảng 5 USD],
    [Mở rộng (Chương 5)], [CPU], [Học trọng số, kiểm định],
    [Đánh giá RAG], [Kaggle T4 ×2], [vLLM, Qwen2.5-7B-Instruct],
  ),
  caption: [Các bước thực nghiệm và phần cứng sử dụng ],
) <tab:runtime>

== Hiện thực sáu khâu

=== Tài liệu và phân tích tài liệu

Mỗi tài liệu được ghép tiêu đề và nội dung. Với BM25, văn bản đi qua bộ tách từ của bm25s: chữ thường, tách bằng biểu thức chính quy, bỏ stopword tiếng Anh, stemming Snowball (PyStemmer). Với Contriever, văn bản được thêm tiền tố `passage: ` và tách thành WordPiece, cắt ở 512 token.

=== Lập chỉ mục

*Chỉ mục BM25.* Mã nguồn gốc dùng thư viện `rank_bm25`, vốn phải duyệt từng tài liệu bằng Python cho mỗi câu truy vấn nên không chạy được với corpus 5 triệu tài liệu. Đồ án viết lớp `BM25sSparseRetriever` dùng bm25s (chỉ mục ma trận thưa, phương pháp Lucene) và một script vá để pipeline gọi lớp này. Như @sec:bm25 cho thấy, cấu hình này còn khớp bài báo tốt hơn mã nguồn gốc.

*Chỉ mục dense.* FiQA và SciDocs dùng chỉ mục FAISS phẳng của mã nguồn gốc. Climate-FEVER và HotpotQA quá lớn để encode trong một phiên Kaggle (giới hạn 12 giờ), nên đồ án encode thành 11 *shard* fp16, mỗi shard 500.000 tài liệu, chạy tiếp được khi phiên bị ngắt. Khi truy vấn, tích vô hướng được tính trên GPU lần lượt từng shard rồi gộp top-1000.

=== Phân tích câu truy vấn

Câu truy vấn gốc được xử lý giống tài liệu (tách từ cho BM25; tiền tố `query: ` cho Contriever). Câu truy vấn mở rộng được sinh bằng LLaMA 3.1-8B-Instruct qua vLLM (fp16, chia mô hình trên 2 GPU T4, giải mã tham lam, tối đa 256 token, dùng chat template) với prompt Phụ lục G, cho tập test của 4 bộ và tập train của FiQA, HotpotQA. Theo mã nguồn gốc, câu truy vấn mở rộng được encode như một tài liệu (tiền tố `passage: `).

=== Truy xuất chỉ mục

Script `run_retrieval.py` gọi lại các hàm của tác giả (encoder, chuẩn hóa, trộn, đánh giá) nhưng đổi thứ tự chạy để tiết kiệm bộ nhớ: chạy xong BM25 thì giải phóng RAM rồi mới chạy dense. Kết quả là bốn danh sách top-1000 (OS, OD, ES, ED) cho mỗi câu truy vấn. Trên dữ liệu giả lập, file đánh giá của script trùng tuyệt đối với lệnh `qudar.cli` của tác giả.


=== Xếp hạng

Với mỗi câu truy vấn, đồ án tính chỉ số của cả 1.331 điểm lưới $(alpha, beta_1, beta_2)$ bằng numpy (trùng với `pytrec_eval` đến $10^(-4)$). Từ kết quả lưới suy ra Static Avg, Static UB và Query-wise UB. Các phương pháp còn lại (RRF, Equal, QuDAR-confidence, QuDAR-llm, DAT) được tính bằng mã nguồn của tác giả; điểm LLM được lấy từ gpt-4o-mini (temperature 0).

== Tổ chức mã nguồn

Đồ án không sửa trực tiếp mã nguồn của tác giả mà bổ sung các script riêng, gọi lại các hàm gốc (encoder, chuẩn hóa, trộn, đánh giá) ở mọi chỗ có thể. @tab:scripts liệt kê các script theo thứ tự chạy.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    table.header([*Script*], [*Khâu*], [*Chức năng*]),
    [`bm25_fast.py`, `apply_patch.py`], [2–3], [Lớp `BM25sSparseRetriever` dùng bm25s và script vá để pipeline gốc gọi lớp này],
    [`encode_shards.py`, `build_index_from_shards.py`], [3], [Encode corpus lớn thành shard fp16, chạy tiếp khi bị ngắt; ghép shard thành chỉ mục FAISS],
    [`expand_queries.py`], [4], [Sinh câu truy vấn mở rộng bằng LLaMA 3.1-8B (vLLM, dự phòng transformers)],
    [`run_retrieval.py`], [5], [Bốn danh sách top-1000 và các phương pháp không cần LLM; tiết kiệm RAM, tìm dense trên GPU theo shard],
    [`grid_sweep.py`], [6], [Chỉ số của 1.331 điểm lưới cho từng câu truy vấn; Static Avg, Static UB, Query-wise UB],
    [`llm_score.py`], [6], [Chấm điểm top-1 bằng gpt-4o-mini theo 5 chế độ (bài báo, mã nguồn, hai biến thể ablation, DAT)],
    [`step5_analysis.py`], [—], [Bảng so sánh với bài báo, các hình, thống kê thiên lệch vị trí],
    [`step6.py`, `step6b.py`], [6], [Các hướng mở rộng ở Chương 5: đặc trưng, mô hình học trọng số, khử thiên lệch, kiểm định],
    [`rag_eval.py`], [—], [Đánh giá RAG end-to-end],
    [`pack_demo_data.py`], [—], [Đóng gói dữ liệu cho bản demo ở Chương 6],
  ),
  caption: [Các script đồ án bổ sung cho pipeline gốc],
) <tab:scripts>

Mỗi bước chạy trong một notebook Kaggle hoặc Colab riêng, đánh số theo thứ tự. Kết quả trung gian lớn (chỉ mục, shard, danh sách truy xuất, kết quả quét lưới) được lưu thành dataset Kaggle để bước sau gắn vào làm đầu vào; kết quả nhỏ (bảng, hình, điểm LLM) được lưu trong kho GitHub.

== Kiểm tra tính đúng đắn của pipeline

Vì đồ án thay đổi cách chạy của pipeline gốc (BM25, chia shard, chạy theo giai đoạn), mỗi thay đổi đều được kiểm tra để bảo đảm không làm sai kết quả:

- *Khớp với lệnh gốc.* Trên một corpus giả lập nhỏ, file đánh giá của `run_retrieval.py` trùng tuyệt đối với lệnh `qudar.cli` của tác giả.
- *Khớp giữa hai bộ BM25.* Lớp `FastBM25SparseRetriever` (cài lại `rank_bm25` bằng ma trận thưa) cho điểm trùng với `rank_bm25` gốc, nên khoảng chênh ở @tab:bm25 đến từ cấu hình chứ không phải từ cách cài.
- *Khớp chỉ số.* Chỉ số tính bằng numpy trong `grid_sweep.py` trùng với `pytrec_eval` đến $10^(-4)$.
- *Khớp dữ liệu.* Trước khi tìm dense, mã tài liệu của từng shard được so với corpus; bước này được thêm sau sự cố Climate-FEVER đọc nhầm shard của HotpotQA.
- *Tái lập trên máy khác.* Bản demo truy xuất lại toàn bộ 648 câu truy vấn test của FiQA trên CPU laptop và khớp với kết quả trên Kaggle (@tab:selftest).
