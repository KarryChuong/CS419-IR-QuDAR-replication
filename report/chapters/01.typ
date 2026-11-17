= TỔNG QUAN

== Bài toán xếp hạng trong truy xuất thông tin

Cho một tập tài liệu (corpus) và một câu truy vấn, hệ truy xuất thông tin (IR) phải trả về danh sách tài liệu được sắp theo mức độ liên quan giảm dần. Trong thực tế người dùng chỉ đọc vài kết quả đầu, nên chất lượng của hệ thống phụ thuộc phần lớn vào *khâu xếp hạng*: tính cho mỗi tài liệu một giá trị xếp hạng sao cho tài liệu liên quan đứng trên.

Mỗi họ phương pháp truy xuất có một cách tính giá trị xếp hạng riêng:

- *Truy xuất thưa (sparse)*, tiêu biểu là BM25 @robertson2009bm25, chấm điểm dựa trên tần suất từ khóa xuất hiện trong tài liệu và độ hiếm của từ trong corpus. Phương pháp này mạnh khi câu truy vấn chứa thuật ngữ, tên riêng, con số cần khớp chính xác.
- *Truy xuất dày (dense)*, ví dụ Contriever @izacard2022contriever, biểu diễn câu truy vấn và tài liệu thành vector rồi chấm điểm bằng tích vô hướng. Phương pháp này tìm được tài liệu cùng nghĩa dù dùng từ khác.
- *Mở rộng câu truy vấn bằng LLM*, như Query2doc @wang2023query2doc hay HyDE @gao2023hyde, cho LLM viết một đoạn văn giả định về chủ đề của câu truy vấn rồi dùng đoạn văn đó để truy xuất, nhờ vậy bổ sung từ vựng và ngữ cảnh mà câu truy vấn ngắn còn thiếu.

Không phương pháp nào thắng ở mọi câu truy vấn. Vì vậy các hệ thống thực tế thường dùng *truy xuất lai (hybrid)*: chạy nhiều bộ truy xuất rồi trộn điểm. Cách trộn phổ biến nhất là tổng có trọng số của các điểm đã chuẩn hóa, trong đó trọng số thường được cố định cho mọi câu truy vấn. Câu hỏi của bài toán là: trọng số nào là tốt nhất, và có nên thay đổi trọng số theo từng câu truy vấn hay không.

== Bài báo QuDAR

QuDAR (Query-Wise Dual-Perspective Adaptive Retrieval) @kim2026qudar do nhóm tác giả ở KAIST công bố tại ACL 2026. Bài báo kết hợp hai góc nhìn:

- *Góc nhìn bộ truy xuất:* sparse (BM25) hoặc dense (Contriever).
- *Góc nhìn câu truy vấn:* câu truy vấn gốc hoặc câu truy vấn mở rộng do LLaMA 3.1-8B-Instruct @grattafiori2024llama3 sinh ra.

Hai góc nhìn cho ra bốn danh sách kết quả, ký hiệu OS, OD, ES, ED (chữ đầu là câu truy vấn gốc/mở rộng, chữ sau là sparse/dense). QuDAR trộn bốn danh sách bằng tổng có trọng số, với trọng số được chọn riêng cho từng câu truy vấn theo một trong hai cách:

- *QuDAR-confidence:* dựa trên độ tự tin của từng danh sách, đo bằng khoảng cách điểm giữa kết quả thứ nhất và thứ hai.
- *QuDAR-llm:* cho một LLM (gpt-4o-mini) đọc tài liệu đứng đầu của mỗi danh sách và chấm điểm mức liên quan từ 0 đến 5.

Bài báo cho thấy cận trên khi chọn trọng số tối ưu cho từng câu truy vấn (Query-wise UB) cao hơn nhiều so với trọng số tĩnh tốt nhất, và QuDAR vượt các phương pháp trộn một chiều như DAT @hsu2025dat (chỉ chọn trọng số giữa sparse và dense). Kết quả được báo cáo trên bốn bộ dữ liệu của BEIR @thakur2021beir: FiQA, SciDocs, Climate-FEVER và HotpotQA, kèm một thí nghiệm RAG cho thấy tài liệu tốt hơn dẫn đến câu trả lời tốt hơn.

== Mục tiêu và phạm vi đồ án

Đồ án đặt ra bốn nhóm công việc:

+ *Tái hiện.* Chạy lại pipeline của tác giả trên đủ bốn bộ dữ liệu, kể cả hai corpus hơn 5 triệu tài liệu (Climate-FEVER, HotpotQA), và so sánh với các bảng kết quả của bài báo.
+ *Kiểm chứng.* Tìm và đo những điểm mã nguồn công bố khác với mô tả trong bài báo.
+ *Mở rộng.* Thử trọng số tĩnh chọn trên tập huấn luyện (Static-train), mô hình học trọng số theo câu truy vấn, khử thiên lệch vị trí của LLM chấm điểm, và đánh giá RAG end-to-end.
+ *Demo.* Một trang web chạy local để minh họa đủ các khâu cho một câu truy vấn.

Phạm vi: đồ án dùng đúng các mô hình của bài báo (Contriever, LLaMA 3.1-8B-Instruct, gpt-4o-mini) và không đề xuất mô hình truy xuất mới. Toàn bộ thí nghiệm chạy trên tài nguyên miễn phí (Kaggle, Google Colab) cùng khoảng dưới 10 USD phí gọi API OpenAI.

== Kết quả đạt được

- Các biến thể QuDAR không dùng LLM và các cận trên lệch bài báo trong khoảng ±3% (lớn nhất 3,2%) trên cả bốn bộ dữ liệu.
- Phát hiện bốn điểm mã nguồn khác bài báo. Đáng kể nhất là BM25 của mã nguồn cho kết quả thấp hơn bài báo khoảng 30%; đồ án thay bằng thư viện bm25s @lu2024bm25s và khớp được bài báo.
- LLM chấm điểm thiên lệch theo vị trí đoạn văn trong prompt, và ưu thế của QuDAR-llm trên FiQA phụ thuộc vào thứ tự đưa đoạn văn.
- Một trọng số tĩnh chọn trên tập huấn luyện vượt cả QuDAR-confidence lẫn QuDAR-llm có ý nghĩa thống kê trên FiQA và HotpotQA.
- Thí nghiệm RAG tái hiện được kết luận của bài báo trên HotpotQA.
- Bản demo trên laptop tái lập chính xác kết quả trên toàn bộ 648 câu truy vấn kiểm tra của FiQA.
