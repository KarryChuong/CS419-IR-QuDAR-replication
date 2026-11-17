#import "lib.typ": *
// 1. Cấu hình giao diện và typography
#show: report.with(
  theme: "academic",
  draft: false,
)

// 2. Trang bìa
#report-cover(
  course-name: "Truy xuất thông tin",
  course-id: "CS419.R11", 
  title: [
    Tái hiện và mở rộng QuDAR \
    #text(size: 0.7em, weight: "regular")[Trộn kết quả truy xuất thích ứng theo từng truy vấn]
  ],
  instructor: "TS. Nguyễn Trọng Chỉnh",
  group-name: "Nhóm CTB",
  members: (
    ("Bùi Huy Chương", "24520230"),
    ("Nguyễn Thái Bảo", "24520173"),
    ("Phạm Duy Tân", "24521584"),
  ),
  date: "Thành phố Hồ Chí Minh, Tháng 11, năm 2026",
  // Resolve the asset relative to main.typ, independent of the compile root.
  logo: read("assets/LOGO_UIT.PNG", encoding: none),
)

// 3. Phần mở đầu (Frontmatter)
#pagebreak()
#outline()

#list-of-tables()
#list-of-figures()
#list-of-abbreviations((
  IR: ("Information Retrieval", "Truy xuất thông tin"),
  BM25: ("Best Matching 25", "Hàm xếp hạng xác suất BM25"),
  LLM: ("Large Language Model", "Mô hình ngôn ngữ lớn"),
  RAG: ("Retrieval-Augmented Generation", "Sinh văn bản có tăng cường truy xuất"),
  QuDAR: ("Query-Wise Dual-Perspective Adaptive Retrieval", "Truy xuất thích ứng hai góc nhìn theo từng câu truy vấn"),
  DAT: ("Dynamic Alpha Tuning", "Điều chỉnh động hệ số alpha"),
  RRF: ("Reciprocal Rank Fusion", "Trộn theo nghịch đảo thứ hạng"),
  OS: ("Original query, Sparse retriever", "BM25 với câu truy vấn gốc"),
  OD: ("Original query, Dense retriever", "Contriever với câu truy vấn gốc"),
  ES: ("Expanded query, Sparse retriever", "BM25 với câu truy vấn mở rộng"),
  ED: ("Expanded query, Dense retriever", "Contriever với câu truy vấn mở rộng"),
  UB: ("Upper Bound", "Cận trên"),
  nDCG: ("normalized Discounted Cumulative Gain", "Độ lợi tích lũy chiết khấu chuẩn hóa"),
  BEIR: ("Benchmarking IR", "Bộ dữ liệu chuẩn đánh giá truy xuất"),
  FAISS: ("Facebook AI Similarity Search", "Thư viện tìm kiếm vector tương tự"),
))

#pagebreak()
#heading(level: 1, numbering: none)[MỞ ĐẦU]

Một hệ truy xuất thông tin hiện đại thường không dựa vào một bộ truy xuất duy nhất. Bộ truy xuất thưa (sparse) như BM25 mạnh ở việc khớp chính xác từ khóa, còn bộ truy xuất dày (dense) dựa trên mạng nơ-ron mạnh ở việc khớp ngữ nghĩa. Câu truy vấn cũng có thể được mở rộng bằng mô hình ngôn ngữ lớn (LLM) để bổ sung ngữ cảnh. Kết hợp các nguồn này cho ra nhiều danh sách kết quả, và câu hỏi đặt ra là: *nên trộn các danh sách theo công thức nào để có thứ hạng cuối cùng tốt nhất?*

Đồ án thuộc bài toán *"Chọn công thức tính giá trị xếp hạng tốt nhất"*. Đồ án chọn tái hiện bài báo QuDAR @kim2026qudar (ACL 2026), một phương pháp trộn bốn danh sách kết quả (BM25 hoặc Contriever, với câu truy vấn gốc hoặc câu truy vấn mở rộng) bằng trọng số được chọn riêng cho từng câu truy vấn. Theo gợi ý của giảng viên, đồ án chạy lại pipeline mã nguồn mở của chính các tác giả, bổ sung những phần còn thiếu để chạy được trên corpus hàng triệu tài liệu, và đối chiếu kết quả với bài báo.

*Mục tiêu của đồ án:*

- Hiện thực đầy đủ quy trình của một hệ truy xuất thông tin: phân tích tài liệu, lập chỉ mục, phân tích câu truy vấn, truy xuất chỉ mục và xếp hạng, với trọng tâm là khâu xếp hạng.
- Tái hiện kết quả chính của QuDAR trên đủ bốn bộ dữ liệu của bài báo: FiQA, SciDocs, Climate-FEVER và HotpotQA.
- Kiểm tra những điểm mã nguồn công bố khác với mô tả trong bài báo, và đo ảnh hưởng của từng điểm.
- Thử một số hướng mở rộng: trọng số tĩnh chọn trên tập huấn luyện, học trọng số theo từng câu truy vấn, khử thiên lệch vị trí của LLM, và đánh giá chất lượng câu trả lời RAG.
- Xây dựng một bản demo chạy được trên máy tính cá nhân.

*Cấu trúc báo cáo.* Chương 1 giới thiệu bài toán và bài báo QuDAR. Chương 2 trình bày cơ sở lý thuyết của từng khâu trong quy trình truy xuất và các công thức trộn điểm. Chương 3 mô tả dữ liệu, môi trường và cách hiện thực sáu khâu. Chương 4 trình bày kết quả tái hiện và những điểm mã nguồn công bố khác bài báo. Chương 5 trình bày các hướng mở rộng và đánh giá RAG. Chương 6 giới thiệu bản demo, kết luận và hướng phát triển.

Toàn bộ mã nguồn, notebook, log và kết quả của đồ án được công bố tại GitHub `KarryChuong/CS419-IR-QuDAR-replication`.

#pagebreak()
#heading(level: 1, numbering: none)[THÀNH VIÊN VÀ PHÂN CÔNG]
#set text(font: "New Computer Modern")

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, left, left),
    table.header([*Họ và tên*], [*MSSV*], [*Phụ trách*], [*Công việc*]),
    [Bùi Huy Chương], [24520230], [Xếp hạng: Lõi QuDAR],
    [Toàn bộ: tìm hiểu bài báo; chuẩn bị dữ liệu; lập chỉ mục và sinh câu truy vấn mở rộng; chạy truy xuất và các phương pháp trộn; chấm điểm bằng LLM; phân tích, kiểm định thống kê; các thí nghiệm mở rộng; đánh giá RAG; xây dựng demo; viết báo cáo.],
    [Nguyễn Thái Bảo],[24520173],[Đánh giá, RAG, demo],[Step 5: vẽ bảng so sánh với bài báo, các hình, kiểm định thống kê. Step 7: đánh giá RAG với Qwen. Step 8: demo Gradio và selftest. Làm slide, video demo, dàn trang báo cáo Typst, viết chương 6],
    [Phạm Duy Tân],[24521584],[Dữ liệu, chỉ mục, truy xuất],[Step 0: tải BEIR, tạo dataset Kaggle. Step 1: thay rank_bm25 bằng bm25s, encode Contriever thành shard, sinh câu truy vấn mở rộng bằng vLLM. Step 2: chạy 4 danh sách OS/OD/ES/ED, xử lý lỗi RAM/GPU, kiểm tra shard. Chẩn đoán độ lệch BM25 và dense so với bài báo. Viết chương 3 và phần BM25/Contriever của chương 2]
  ),
  caption: [Thành viên và phân công công việc],
)

// 4. Nội dung các chương (Mainmatter)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#include "chapters/01.typ"

#pagebreak()
#include "chapters/02.typ"

#pagebreak()
#include "chapters/03.typ"

#pagebreak()
#include "chapters/04.typ"

#pagebreak()
#include "chapters/05.typ"

#pagebreak()
#include "chapters/06.typ"

// 5. Kết bài & Tài liệu tham khảo (Backmatter)
#pagebreak()
#bibliography(
  "ref.bib",
  title: "TÀI LIỆU THAM KHẢO",
  style: "ieee",
  full: true,
)
