= CÁC HƯỚNG MỞ RỘNG

Chương 4 cho thấy QuDAR tái hiện được, nhưng còn hai câu hỏi bài báo chưa trả lời: một trọng số tĩnh chọn trên dữ liệu có nhãn có đủ tốt không, và khoảng cách lớn tới Query-wise UB có thu hẹp được không. Chương này trình bày bốn hướng mở rộng: trọng số tĩnh chọn trên tập train, học trọng số theo từng câu truy vấn, khử thiên lệch vị trí, và đánh giá RAG end-to-end.

== Thiết kế thí nghiệm

Các thí nghiệm mở rộng tuân theo ba nguyên tắc để tránh kết luận lạc quan:

- *Tách dữ liệu nghiêm ngặt.* Mọi lựa chọn (điểm trọng số, siêu tham số, ngưỡng) chỉ dùng tập train của FiQA và HotpotQA; tập test chỉ dùng để báo cáo. Ngưỡng của mô hình có cổng được chọn trên 20% validation tách từ tập train.
- *Kịch bản zero-shot.* SciDocs và Climate-FEVER không có tập train, nên mọi phương pháp học được áp dụng nguyên trạng để đo khả năng tổng quát hóa sang miền khác.
- *Kiểm định và giới hạn số lần thử.* Mọi so sánh dùng kiểm định ghép cặp với hiệu chỉnh Holm. Sau lần chạy đầu, chỉ cho phép một lần sửa thiết kế (mô hình có cổng và bộ đặc trưng không phụ thuộc thang điểm), được cố định trước khi chạy.

== Trọng số tĩnh chọn trên tập train (Static-train)

Bài báo so QuDAR với Static Avg và Static UB. Static UB được chọn trên chính tập test nên không dùng được trong thực tế; bài báo không so với một trọng số tĩnh *chọn trên dữ liệu train*, là baseline tự nhiên nhất khi có câu truy vấn có nhãn. Đồ án định nghĩa *Static-train* là điểm lưới có nDCG\@10 trung bình cao nhất trên tập train của FiQA và HotpotQA gộp lại, rồi áp dụng nguyên điểm đó cho tập test của cả bốn bộ.

#figure(
  text(size: 9pt, table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header([*Phương pháp*], [*FiQA*], [*SciDocs (zero-shot)*], [*Climate-FEVER (zero-shot)*], [*HotpotQA*]),
    [QuDAR-confidence], [0.3354 / 0.2525], [0.1907 / 0.1798], [0.2507 / 0.2043], [0.6853 / 0.6366],
    [QuDAR-llm], [0.3281 / 0.2539], [0.1899 / 0.1799], [0.2567 / 0.2093], [0.6704 / 0.6355],
    [*Static-train*], [*0.3555 / 0.2754*], [0.1917 / 0.1808], [0.2473 / 0.2012], [*0.7039 / 0.6621*],
    [Direct-B (học, có LLM)], [0.3587 / 0.2803], [0.1939 / 0.1823], [0.2462 / 0.2026], [*0.7074 / 0.6715*],
    [Gated B (học, có LLM)], [0.3529 / 0.2850], [0.1899 / 0.1766], [0.2334 / 0.1894], [0.7014 / 0.6656],
    [Query-wise UB (oracle)], [0.4944 / 0.4321], [0.2749 / 0.2810], [0.3734 / 0.3471], [0.7708 / 0.7637],
  )),
  caption: [Static-train và các mô hình học trọng số, dạng Recall\@10 / nDCG\@10],
) <tab:ext>

Trên FiQA và HotpotQA (cùng miền với dữ liệu train), Static-train vượt QuDAR-confidence (Recall +0.020 và +0.019; nDCG +0.023 và +0.026) và QuDAR-llm (Recall +0.027 và +0.034), đều có ý nghĩa sau hiệu chỉnh Holm. Trên SciDocs không khác biệt; trên Climate-FEVER, Static-train kém QuDAR-llm (nDCG −0.008, có ý nghĩa). Kết luận: luật chọn trọng số theo câu truy vấn của QuDAR có giá trị khi *không* có dữ liệu train cùng miền; khi có, một trọng số tĩnh đơn giản lại tốt hơn.

== Học trọng số theo từng câu truy vấn

Khoảng cách giữa các phương pháp thực tế và Query-wise UB (ví dụ FiQA 0.3555 so với 0.4944) gợi ý thử *học* trọng số cho từng câu truy vấn.

*Đặc trưng.* Mỗi câu truy vấn được mô tả bằng 55 đặc trưng rẻ, tính từ bốn danh sách và từ câu truy vấn: điểm top-1, khoảng cách top-1/top-2, độ dốc điểm trong top-10 của từng danh sách; độ trùng top-10 (Jaccard) giữa các cặp danh sách; top-1 của danh sách này có nằm trong danh sách khác không; độ dài câu truy vấn và câu mở rộng. Model B thêm 8 đặc trưng từ LLM (4 điểm và vị trí của chúng trong prompt), tổng 63.

*Mô hình.* LightGBM @ke2017lightgbm dự đoán nDCG\@10 (đã trừ trung bình theo câu truy vấn) của từng điểm lưới bước 0.2 (216 điểm), rồi chọn điểm có dự đoán cao nhất (Model A, Model B). Biến thể Direct hồi quy thẳng $(alpha, beta_1, beta_2)$. Biến thể Gated chỉ rời Static-train khi lợi ích dự đoán vượt một ngưỡng chọn trên 20% validation tách từ tập train. Thiết kế của biến thể Gated được cố định trước khi chạy, và đây là lần sửa duy nhất sau thí nghiệm đầu.

*Kết quả* (@tab:ext):

- Cùng miền, các mô hình học vượt QuDAR có ý nghĩa, nhưng phần lớn mức tăng đã có ở Static-train. So với Static-train, chỉ HotpotQA có cải thiện có ý nghĩa: Direct-B nDCG +0.0095, Recall +0.0035; Gated B nDCG +0.0035.
- Zero-shot (SciDocs, Climate-FEVER), mọi mô hình học ngang hoặc kém Static-train; ở Climate-FEVER kém có ý nghĩa. Bộ đặc trưng không phụ thuộc thang điểm giảm được một phần thiệt hại nhưng không khắc phục được.
- Ngoài chính trọng số đang được chấm, các đặc trưng quan trọng nhất của câu truy vấn đo độ đồng thuận giữa các danh sách (độ trùng top-10 giữa OS và ES; top-1 của OS có nằm trong danh sách khác không).

Như vậy, với đặc trưng rẻ, khoảng cách tới Query-wise UB gần như chưa được khai thác. Đây là giới hạn chính và là hướng mở cho nghiên cứu tiếp.

== Khử thiên lệch vị trí

Từ @tab:position, độ lệch điểm theo vị trí được ước lượng trên tập train rồi trừ khỏi điểm LLM của tập test trước khi tính trọng số QuDAR-llm. Kết quả tốt hơn hoặc ngang bằng ở cả bốn bộ nhưng nhỏ: sau hiệu chỉnh Holm chỉ còn HotpotQA nDCG\@10 +0.0016 có ý nghĩa. Trên FiQA, bản khử thiên lệch đạt Recall\@10 0.3347 và nDCG\@10 0.2585, so với 0.3281 và 0.2539 của QuDAR-llm, nhưng vẫn thấp hơn khi dùng thứ tự cố định (0.3423).

== Đánh giá RAG end-to-end

Bài báo kiểm tra liệu tài liệu tốt hơn có cho câu trả lời tốt hơn hay không (Table 3 của bài báo). Đồ án dựng lại thí nghiệm theo kiến trúc RAG @lewis2020rag:

- *Dữ liệu:* 500 câu truy vấn ngẫu nhiên (seed 0) mỗi bộ, FiQA và HotpotQA, cho 13 phương pháp.
- *Sinh câu trả lời:* Qwen2.5-7B-Instruct @qwen2025qwen25 (vLLM, giải mã tham lam) đọc 5 tài liệu đầu (mỗi tài liệu tối đa 180 từ).
- *Chấm điểm:* chính Qwen chấm Yes/No theo prompt QA-Eval @wang2023qaeval. Đáp án tham chiếu của HotpotQA là đáp án ngắn có sẵn trong BEIR; FiQA không có đáp án ngắn nên dùng tối đa 3 đoạn văn liên quan.

Bài báo không công bố prompt chấm, số tài liệu đưa vào và cách chọn câu truy vấn. Vì vậy điểm tuyệt đối của đồ án cao hơn bài báo nhiều (FiQA khoảng 0.5 so với 0.14–0.18), và chỉ nên so *thứ tự giữa các phương pháp*.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Phương pháp*], [*FiQA Gen.*], [*HotpotQA Gen.*]),
    [OS], [0.508], [0.454],
    [QuDAR-confidence], [0.494], [0.532],
    [QuDAR-llm], [0.508], [0.546],
    [Static-train], [0.534], [0.540],
    [Model B (học, có LLM)], [0.552], [0.508],
    [Query-wise UB (oracle)], [0.600], [0.588],
  ),
  caption: [Tỷ lệ câu trả lời đúng (Gen.) trên 500 câu truy vấn mỗi bộ],
) <tab:rag>

- Recall\@10 và Gen. tương quan mạnh trên 13 phương pháp: Pearson 0.91 (FiQA) và 0.98 (HotpotQA).
- *HotpotQA tái hiện kết luận của bài báo:* QuDAR-confidence hơn OS +0.078 (+17%) và hơn trộn một chiều Retriever-Type Fusion +0.060 (+13%); QuDAR-llm hơn OS +0.092. Các chênh lệch này có ý nghĩa sau hiệu chỉnh Holm (8 so sánh mỗi bộ); bài báo báo cáo mức tăng khoảng 20% và 10%.
- *FiQA không có khác biệt có ý nghĩa* giữa các phương pháp thực tế (0.49–0.55); chỉ các phương pháp thuần dense thấp rõ (0.36–0.42). Câu hỏi FiQA mang tính tư vấn và đáp án tham chiếu dài nên cách chấm nhiễu hơn.
- Static-train và Model B không khác biệt có ý nghĩa so với QuDAR sau hiệu chỉnh Holm.
- Query-wise UB đạt 0.600 và 0.588, cao hơn rõ các phương pháp thực tế: khoảng trống của truy xuất cũng là khoảng trống của RAG.
