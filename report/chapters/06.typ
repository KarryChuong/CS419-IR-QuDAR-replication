= DEMO, KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

== Demo

=== Mục tiêu và kiến trúc

Bản demo là một trang web Gradio chạy trên laptop, chỉ cần CPU, cho bộ FiQA. Mục tiêu là minh họa trực quan cả sáu khâu của quy trình cho từng câu truy vấn, và cho thấy các công thức trộn khác nhau cho ra thứ hạng khác nhau thế nào. Demo gồm bốn thành phần:

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Thành phần*], [*Vai trò*]),
    [`pack_demo_data.py` (chạy trên Kaggle)], [Gom corpus, câu truy vấn, nhãn, ma trận embedding lấy từ chỉ mục FAISS, bốn danh sách đã lưu, câu truy vấn mở rộng, điểm LLM và mô hình LightGBM thành một gói khoảng 110 MB],
    [`qudar_core.py`], [Hiện thực sáu khâu trên laptop: dựng chỉ mục BM25 (lưu lại sau lần đầu), encode bằng Contriever trên CPU, tìm dense bằng tích vô hướng, chuẩn hóa và tính trọng số của mọi phương pháp bằng đúng công thức của thí nghiệm],
    [`app.py`], [Giao diện Gradio năm tab; gọi OpenAI hoặc Ollama khi cần LLM],
    [`selftest.py`], [Chạy lại toàn bộ câu truy vấn test và so với kết quả trên Kaggle],
  ),
  caption: [Các thành phần của bản demo],
) <tab:demo>

=== Giao diện và chế độ chạy

Người dùng chọn một câu hỏi test (có nhãn) hoặc tự nhập câu hỏi, và xem:

- top-10 sau khi trộn theo phương pháp đang chọn, với thanh màu cho biết mỗi danh sách đóng góp bao nhiêu điểm;
- trọng số mà 16 phương pháp gán cho câu truy vấn đó, kèm Recall\@10, nDCG\@10, P\@1 nếu câu có nhãn;
- top-10 của từng danh sách OS, OD, ES, ED, với điểm thô và điểm đã chuẩn hóa;
- sáu khâu của quy trình áp dụng cho câu này, cùng thời gian chạy từng khâu;
- câu trả lời RAG khi đưa top-5 của OS và của phương pháp đang chọn cho LLM.

Với câu hỏi test, demo chạy hoàn toàn offline nhờ dùng lại câu truy vấn mở rộng, điểm LLM và câu trả lời RAG đã lưu. Với câu tự nhập, câu truy vấn mở rộng được sinh bằng gpt-4o-mini hoặc bằng một mô hình chạy trên máy qua Ollama; ba thanh trượt $alpha$, $beta_1$, $beta_2$ cho phép tự chỉnh trọng số và xem thứ hạng thay đổi ngay.

=== Kiểm tra tái lập

Script `selftest.py` chạy lại toàn bộ 648 câu truy vấn test trên laptop theo hai cách: truy xuất trực tiếp (dựng lại chỉ mục BM25, encode bằng Contriever trên CPU) và dùng danh sách đã lưu. Kết quả ở @tab:selftest.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Kiểm tra*], [*Kết quả*]),
    [Top-10 trùng với danh sách đã lưu (trung bình mỗi câu)], [OS 10/10, OD 10/10, ES 10/10, ED 9,96/10],
    [Model A / Model B chọn cùng điểm lưới như thí nghiệm], [648/648 và 648/648],
    [Recall\@10, nDCG\@10 của 13 phương pháp, trực tiếp so với đã lưu], [Trùng đến 4 chữ số],
    [Thời gian], [385 giây cho 648 câu × 2 cách, trên CPU laptop],
  ),
  caption: [Kiểm tra tái lập của bản demo trên FiQA],
) <tab:selftest>

Phần xếp hạng của đồ án vì vậy là tất định và tái lập được trên máy khác. Phần không tất định nằm ở LLM: câu truy vấn mở rộng sinh trực tiếp trong demo khác cấu hình của thí nghiệm.

Một câu hỏi test được xử lý trong chưa tới một giây trên CPU laptop; câu tự nhập dùng gpt-4o-mini mất 3–6 giây, phần lớn là thời gian chờ API.

== Kết luận

*Kết quả đạt được.* Đồ án hiện thực đầy đủ sáu khâu của một hệ truy xuất thông tin trên bốn bộ dữ liệu, kể cả hai corpus hơn 5 triệu tài liệu, với tài nguyên miễn phí. Ở khâu xếp hạng, đồ án tái hiện được các phương pháp trộn của QuDAR trong khoảng ±3% so với bài báo, và xác nhận rằng chọn trọng số riêng cho từng câu truy vấn có dư địa lớn (Query-wise UB cao hơn Static UB 9–43%).

*Phát hiện chính:*

+ Mã nguồn công bố khác bài báo ở bốn điểm (BM25, tiền tố Contriever, cách encode câu truy vấn mở rộng, thứ tự và prompt của QuDAR-llm). Thay BM25 bằng bm25s khắc phục khoảng hụt 30%.
+ LLM chấm điểm thiên lệch vị trí rõ rệt; ưu thế của QuDAR-llm trên FiQA gắn với thứ tự đoạn văn cố định của mã nguồn và không lặp lại ổn định trên các bộ khác.
+ Khi có dữ liệu train cùng miền, một trọng số tĩnh chọn trên train vượt QuDAR có ý nghĩa thống kê; học trọng số theo câu truy vấn bằng đặc trưng rẻ chỉ thêm được rất ít.
+ Truy xuất tốt hơn dẫn tới câu trả lời RAG tốt hơn: tái hiện được trên HotpotQA, chưa rõ trên FiQA.

*Hạn chế.* Chưa giải thích được khoảng hụt của bộ truy xuất dense (ED thấp hơn bài báo tới 37%). Đồ án mới dùng một LLM chấm điểm (gpt-4o-mini), một bộ truy xuất dense (Contriever) và hai bộ dữ liệu có tập train. Đánh giá RAG là bản dựng lại, với mô hình tự chấm câu trả lời của chính nó.

== Hướng phát triển

Liên hệ tác giả để làm rõ các điểm khác biệt; thêm một LLM chấm điểm mã nguồn mở để kiểm tra thiên lệch vị trí; thêm các bộ dữ liệu có tập train (SciFact, NFCorpus) và đo số câu có nhãn cần để Static-train vượt QuDAR; thử một bộ truy xuất dense mạnh hơn (ví dụ E5, mô hình vốn dùng tiền tố `query:`/`passage:`). Những kết quả này có thể phát triển thành một bài báo tái hiện (reproducibility paper).
