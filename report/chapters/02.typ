= CƠ SỞ LÝ THUYẾT

== Quy trình của một hệ truy xuất thông tin

Một hệ truy xuất thông tin gồm sáu khâu. Hai khâu đầu chạy *offline* một lần cho cả corpus; bốn khâu sau chạy *online* cho mỗi câu truy vấn (@fig:pipeline).

#let stage(title, body) = rect(
  width: 100%,
  inset: 6pt,
  radius: 4pt,
  stroke: 0.6pt,
)[#align(center)[#text(weight: "bold", size: 9pt)[#title] \ #text(size: 8pt)[#body]]]

#figure(
  grid(
    columns: (1fr, auto, 1fr, auto, 1fr),
    column-gutter: 4pt,
    row-gutter: 8pt,
    align: horizon,
    stage[1\. Tài liệu][corpus BEIR, ghép tiêu đề + nội dung], [→],
    stage[2\. Phân tích tài liệu][tách từ, stopword, stemming / WordPiece], [→],
    stage[3\. Lập chỉ mục][chỉ mục ngược BM25 / ma trận vector],
    [], [], [], [], align(center)[↓ #text(size: 8pt)[(offline)]],
    stage[6\. Xếp hạng][chuẩn hóa, gán trọng số, trộn 4 danh sách], [←],
    stage[5\. Truy xuất chỉ mục][4 danh sách top-1000: OS, OD, ES, ED], [←],
    stage[4\. Phân tích câu truy vấn][tách từ, mở rộng bằng LLM, encode],
  ),
  caption: [Sáu khâu của hệ truy xuất trong đồ án; khâu xếp hạng là trọng tâm],
) <fig:pipeline>

Các mục dưới đây trình bày lý thuyết của từng thành phần: hai bộ truy xuất (BM25, Contriever), mở rộng câu truy vấn, các công thức trộn điểm, và cách đánh giá.

== Truy xuất thưa: BM25

=== Phân tích tài liệu và lập chỉ mục

Với BM25, mỗi tài liệu được chuyển về chữ thường, tách thành từ bằng biểu thức chính quy, bỏ các stopword tiếng Anh và đưa về gốc từ bằng thuật toán Snowball. Từ các danh sách từ này, hệ thống xây *chỉ mục ngược* (inverted index): với mỗi từ $t$, lưu danh sách tài liệu chứa $t$ cùng số lần xuất hiện. Thư viện bm25s @lu2024bm25s tính sẵn điểm BM25 của từng cặp (từ, tài liệu) và lưu thành ma trận thưa, nên khi truy vấn chỉ cần cộng các dòng tương ứng với từ trong câu truy vấn.

=== Hàm xếp hạng

Điểm BM25 @robertson2009bm25 của tài liệu $d$ với câu truy vấn $q$ là:

$ "BM25"(q, d) = sum_(t in q) "IDF"(t) dot (f(t, d) dot (k_1 + 1)) / (f(t, d) + k_1 (1 - b + b dot (|d|) / "avgdl")) $

trong đó $f(t, d)$ là số lần từ $t$ xuất hiện trong $d$, $|d|$ là độ dài tài liệu, $"avgdl"$ là độ dài trung bình trong corpus, $k_1 = 1.5$ và $b = 0.75$. Biến thể Lucene mà đồ án dùng tính độ hiếm của từ bằng

$ "IDF"(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)) $

với $N$ là số tài liệu và $n_t$ là số tài liệu chứa $t$. Biến thể này bỏ hệ số hằng $(k_1 + 1)$ ở tử số; điều đó không làm thay đổi thứ hạng.

== Truy xuất dày: Contriever

Contriever @izacard2022contriever là bộ mã hóa BERT được huấn luyện tương phản không giám sát. Câu truy vấn và tài liệu được tách thành token WordPiece (tối đa 512 token), đi qua mạng, rồi lấy trung bình vector đầu ra của các token (mean pooling) và chuẩn hóa L2 để được vector 768 chiều $E(dot)$. 

Điểm liên quan là tích vô hướng:

$ s_"dense" (q, d) = ⟨ E(q), E(d) ⟩ $

Vector của toàn bộ corpus được tính trước và lưu trong chỉ mục FAISS @johnson2019faiss loại tìm kiếm chính xác (IndexFlatIP). Khi truy vấn, hệ thống tính tích vô hướng với mọi vector và lấy 1000 tài liệu điểm cao nhất. Mã nguồn của tác giả thêm tiền tố `query: ` trước câu truy vấn và `passage: ` trước tài liệu.

== Mở rộng câu truy vấn bằng LLM

Câu truy vấn thường ngắn và thiếu ngữ cảnh. Theo hướng của Query2doc @wang2023query2doc và HyDE @gao2023hyde, QuDAR cho LLaMA 3.1-8B-Instruct @grattafiori2024llama3 viết một đoạn văn mô tả chủ đề của câu truy vấn. Prompt (Phụ lục G của bài báo) yêu cầu mô hình đóng vai "người tổng hợp dữ kiện trung lập", viết một đoạn văn dày thông tin *nhưng không trả lời trực tiếp* câu hỏi. Đoạn văn này là *câu truy vấn mở rộng*, được dùng cho cả BM25 (danh sách ES) và Contriever (danh sách ED).

== Trộn điểm từ nhiều danh sách

=== Chuẩn hóa min-max

Điểm của các bộ truy xuất có thang đo khác nhau (BM25 không có cận trên, tích vô hướng nằm trong $[-1, 1]$). Trước khi trộn, điểm của mỗi danh sách $k$ được đưa về $[0, 1]$ trên 1000 kết quả của danh sách đó:

$ tilde(s)_k (d) = (s_k (d) - min_k) / (max_k - min_k) $

=== Tổng có trọng số

Điểm cuối cùng của tài liệu $d$ là tổng có trọng số của bốn điểm đã chuẩn hóa; tài liệu không có trong danh sách $k$ nhận $tilde(s)_k (d) = 0$:

$ S(d) = sum_(k in {"OS", "OD", "ES", "ED"}) w_k dot tilde(s)_k (d), quad sum_k w_k = 1 $

QuDAR tham số hóa bốn trọng số bằng ba hệ số trong $[0, 1]$: $alpha$ là phần dành cho dense, $beta_1$ là phần dành cho câu truy vấn mở rộng bên trong BM25, $beta_2$ là phần tương ứng bên trong Contriever:

$ w_"OS" = (1 - alpha)(1 - beta_1), quad w_"ES" = (1 - alpha) beta_1, quad w_"OD" = alpha (1 - beta_2), quad w_"ED" = alpha beta_2 $

Với bước 0.1, lưới $(alpha, beta_1, beta_2)$ có $11^3 = 1331$ điểm.

=== Trộn theo thứ hạng: RRF

Reciprocal Rank Fusion @cormack2009rrf không dùng điểm mà chỉ dùng thứ hạng $r_k (d)$ của tài liệu trong từng danh sách:

$ "RRF"(d) = sum_k 1 / (60 + r_k (d)) $

== Các chiến lược chọn trọng số

@tab:strategies tóm tắt các chiến lược được so sánh. Hai nhóm đầu dùng một bộ trọng số cho mọi câu truy vấn; QuDAR và DAT chọn trọng số riêng cho từng câu.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Chiến lược*], [*Cách chọn trọng số*]),
    [Static Avg / Static UB], [Trung bình chỉ số trên mọi điểm lưới / điểm lưới tốt nhất trên chính tập kiểm tra (cận trên của trọng số tĩnh, không dùng được trong thực tế)],
    [QuDAR-simple (Equal)], [$w_k = 0.25$ cho cả bốn danh sách],
    [QuDAR-simple (RRF)], [Công thức RRF ở trên],
    [QuDAR-confidence], [$w = "softmax"(m \/ tau)$, $tau = 2$, với $m_k = max(tilde(s)_k^((1)) - tilde(s)_k^((2)), 0)$ là khoảng cách điểm giữa kết quả thứ nhất và thứ hai của danh sách $k$],
    [QuDAR-llm], [gpt-4o-mini chấm tài liệu đứng đầu của mỗi danh sách bằng điểm nguyên $g_k in {0, ..., 5}$; $w = "softmax"(g \/ tau)$, $tau = 3$; nếu cả bốn bằng 0 thì chia đều],
    [DAT @hsu2025dat], [LLM chấm top-1 của BM25 và Contriever (câu truy vấn gốc); $alpha = 0.5$ nếu cả hai bằng 0, $alpha in {0, 1}$ nếu chỉ một bên đạt 5, ngược lại $alpha = g_"dense" \/ (g_"dense" + g_"BM25")$ làm tròn 0.1],
    [Query-wise UB], [Điểm lưới tốt nhất cho riêng từng câu truy vấn, biết trước nhãn (oracle); đo dư địa tối đa của việc chọn trọng số theo câu truy vấn],
  ),
  caption: [Các chiến lược chọn trọng số trộn],
) <tab:strategies>

== Đánh giá

=== Chỉ số

Với $"Rel"(q)$ là tập tài liệu liên quan của câu truy vấn $q$ và $L_10$ là 10 kết quả đầu:

$ "Recall@10" = (|L_10 inter "Rel"(q)|) / (|"Rel"(q)|), quad "nDCG@10" = 1 / "IDCG@10" sum_(i = 1)^(10) "rel"_i / (log_2 (i + 1)) $

trong đó $"rel"_i$ là mức liên quan của kết quả thứ $i$ và IDCG\@10 là giá trị của thứ hạng lý tưởng. Precision\@1 (P\@1) là tỷ lệ câu truy vấn có kết quả đầu tiên liên quan. Các chỉ số được tính bằng `pytrec_eval` @vangysel2018pytrec và lấy trung bình trên tập kiểm tra.

Với RAG, chỉ số *Gen.* là tỷ lệ câu trả lời được chấm đúng so với đáp án tham chiếu, theo cách chấm QA-Eval @wang2023qaeval.

=== Kiểm định thống kê

Khi so sánh hai phương pháp A và B trên cùng tập câu truy vấn, đồ án dùng *kiểm định ghép cặp*: paired bootstrap (10.000 lần lấy mẫu lại) và paired t-test trên hiệu chỉ số từng câu. Khi thực hiện nhiều so sánh cùng lúc, p-value được hiệu chỉnh bằng phương pháp Holm @holm1979 để kiểm soát xác suất kết luận sai ít nhất một lần.
