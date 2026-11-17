= KẾT QUẢ TÁI HIỆN VÀ KIỂM CHỨNG

Chương này so sánh kết quả của đồ án với bài báo, phân tích những điểm mã nguồn công bố khác mô tả trong bài báo, và tái hiện các phân tích về trọng số tối ưu.

== Kết quả tái hiện

@tab:main so sánh Recall\@10 của đồ án với bài báo.

#figure(
  text(size: 9pt, table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header([*Phương pháp*], [*FiQA*], [*SciDocs*], [*Climate-FEVER*], [*HotpotQA*]),
    [OS], [0.3178 / 0.3135], [0.1653 / 0.1655], [0.1733 / 0.1750], [0.5995 / 0.6039],
    [OD], [0.1492 / 0.1649], [0.0950 / 0.1112], [0.0999 / 0.1235], [0.3849 / 0.4011],
    [ES], [0.2047 / 0.2085], [0.1459 / 0.1497], [0.2576 / 0.2598], [0.4047 / 0.4151],
    [ED], [0.1261 / 0.1835], [0.1361 / 0.1487], [0.1234 / 0.1960], [0.4526 / 0.4838],
    [DAT], [0.2894 / 0.2842], [0.1624 / 0.1614], [0.1679 / 0.1699], [0.6009 / 0.5781],
    [Static UB], [0.3631 / 0.3715], [0.1925 / 0.1917], [0.2929 / 0.2972], [0.7041 / 0.7110],
    [QuDAR-simple (RRF)], [0.3145 / 0.3219], [0.1877 / 0.1893], [0.2422 / 0.2411], [0.6700 / 0.6741],
    [QuDAR-simple (Equal)], [0.3333 / 0.3402], [0.1901 / 0.1906], [0.2504 / 0.2464], [0.6828 / 0.6924],
    [QuDAR-confidence], [0.3354 / 0.3438], [0.1907 / 0.1920], [0.2507 / 0.2473], [0.6853 / 0.6962],
    [QuDAR-llm], [0.3281 / 0.3639], [0.1899 / 0.1920], [0.2567 / 0.2583], [0.6704 / 0.6883],
    [Query-wise UB], [0.4944 / 0.5011], [0.2749 / 0.2745], [0.3734 / 0.3859], [0.7708 / 0.7891],
  )),
  caption: [Recall\@10 trên tập test, dạng *đồ án / bài báo*],
) <tab:main>

Ba nhận xét:

- *Các phương pháp trộn và các cận trên khớp bài báo.* RRF, Equal, QuDAR-confidence, Static UB và Query-wise UB lệch bài báo trong khoảng ±3% (lớn nhất 3,2%) trên cả bốn bộ.
- *Độ lệch tập trung ở bộ truy xuất dense.* OD thấp hơn bài báo 4–19%, ED thấp hơn tới 31% (FiQA) và 37% (Climate-FEVER). Dù vậy, kết quả sau khi trộn vẫn khớp: trộn có trọng số tự giảm ảnh hưởng của danh sách yếu. Ví dụ trên Climate-FEVER, QuDAR-confidence đạt 0.2507 so với 0.2473 của bài báo trong khi ED chỉ đạt 0.1234 so với 0.1960.
- *QuDAR-llm lệch nhiều nhất trên FiQA* (0.3281 so với 0.3639). Nguyên nhân được phân tích ở @sec:llm-order.

Cận trên theo câu truy vấn cao hơn trọng số tĩnh tốt nhất 9–43% (Recall\@10), xác nhận động lực chính của bài báo: chọn trọng số riêng cho từng câu truy vấn có dư địa cải thiện lớn.

== Những điểm mã nguồn khác bài báo

=== BM25 <sec:bm25>

Mã nguồn gốc dùng `rank_bm25` với tách từ bằng `split()`, không bỏ stopword và không bỏ dấu câu, nên `"cheque?"` và `"cheque"` thành hai từ khác nhau. @tab:bm25 cho thấy cấu hình này thấp hơn bài báo khoảng 30%, còn bm25s mặc định khớp gần như tuyệt đối (trùng P\@1 trên FiQA và nDCG\@10 trên SciDocs). Repo của tác giả cũng khai báo thư viện PyStemmer, vốn là stemmer mà bm25s khuyên dùng; nhiều khả năng thí nghiệm của bài báo đã chạy bằng bm25s.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Cấu hình*], [*FiQA*], [*SciDocs*]),
    [`rank_bm25` (mã nguồn gốc)], [0.2156 / 0.1678 / 0.1481], [0.1412 / 0.1390 / 0.1750],
    [bm25s Lucene + stopword (đồ án dùng)], [*0.3178 / 0.2514 / 0.2299*], [*0.1653 / 0.1578 / 0.1830*],
    [bm25s Lucene, không stopword], [0.3173 / 0.2517 / 0.2315], [0.1637 / 0.1561 / 0.1780],
    [Bài báo], [0.3135 / 0.2500 / 0.2299], [0.1655 / 0.1578 / 0.1810],
  ),
  caption: [BM25 với câu truy vấn gốc (OS), dạng Recall\@10 / nDCG\@10 / P\@1],
) <tab:bm25>

=== Tiền tố của Contriever

Bộ mã hóa của tác giả thêm `query: ` và `passage: `, vốn là quy ước của mô hình E5 chứ không phải của Contriever. Trên FiQA, bỏ tiền tố cho OD đạt 0.1525 / 0.1241 / 0.1235 so với 0.1492 / 0.1188 / 0.1096 khi có tiền tố và 0.1649 / 0.1299 / 0.1173 của bài báo. Không cấu hình nào khớp hẳn bài báo, nên đồ án giữ cách của tác giả (hai corpus lớn đã được encode theo cách này).

=== Cách encode câu truy vấn mở rộng cho dense

ED trên FiQA thấp hơn bài báo 31% trong khi ES (cùng nội dung, qua BM25) chỉ lệch 2%. Đồ án thử bốn cách encode (@tab:ed).

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Cách encode*], [*FiQA*], [*SciDocs*]),
    [A: `passage: ` + mở rộng (mã nguồn gốc)], [0.1261 / 0.0939 / 0.0741], [0.1361 / 0.1283 / 0.1460],
    [B: `query: ` + mở rộng], [0.1753 / 0.1198 / 0.0772], [0.1496 / 0.1428 / 0.1710],
    [C: `passage: ` + câu hỏi + mở rộng], [0.1512 / 0.1180 / 0.1049], [0.1403 / 0.1333 / 0.1540],
    [D: `query: ` + câu hỏi + mở rộng], [0.2036 / 0.1532 / 0.1281], [0.1573 / 0.1469 / 0.1640],
    [Bài báo], [0.1835 / 0.1363 / 0.1049], [0.1487 / 0.1390 / 0.1570],
  ),
  caption: [Bốn cách encode câu truy vấn mở rộng cho ED, dạng Recall\@10 / nDCG\@10 / P\@1],
) <tab:ed>

Vì OD của đồ án vốn chỉ bằng khoảng 0,90 lần bài báo ở FiQA và 0,85 lần ở SciDocs, cách encode đúng phải cho ED với cùng tỷ lệ. Không cách nào nhất quán trên cả hai bộ, nên đồ án giữ cách A. Ở HotpotQA, ED chỉ lệch 6%, tương đương OD, nên khoảng hụt lớn ở FiQA và Climate-FEVER nhiều khả năng đến từ nội dung câu truy vấn mở rộng (dense nhạy với văn phong hơn BM25) chứ không phải từ cách encode.

=== QuDAR-llm: thứ tự đoạn văn và prompt <sec:llm-order>

Phụ lục D của bài báo mô tả bốn đoạn văn top-1 được *xáo trộn ngẫu nhiên* rồi chấm bằng Prompt 1 (thang điểm đầy đủ, trả về mảng JSON). Mã nguồn lại luôn đưa theo *thứ tự cố định* ES, OS, ED, OD với một prompt rút gọn. Đồ án chạy theo bài báo (xáo trộn với seed là mã câu truy vấn) cho kết quả chính, rồi làm ablation 2×2 (prompt × thứ tự) trên cả bốn bộ (@tab:ablation; HotpotQA dùng 1.500 câu truy vấn đầu).

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header([*Bộ dữ liệu*], [*Prompt bài báo, xáo trộn*], [*Prompt bài báo, cố định*], [*Prompt mã nguồn, xáo trộn*], [*Prompt mã nguồn, cố định*]),
    [FiQA], [0.3281], [0.3423], [0.3386], [*0.3522*],
    [SciDocs], [0.1899], [0.1906], [*0.1920*], [0.1889],
    [Climate-FEVER], [0.2567], [0.2552], [*0.2609*], [0.2591],
    [HotpotQA], [0.6737], [0.6823], [*0.6927*], [0.6883],
  ),
  caption: [Ablation QuDAR-llm (Recall\@10)],
) <tab:ablation>

- Thứ tự cố định có lợi có ý nghĩa thống kê ở FiQA (cả hai prompt) và HotpotQA (prompt bài báo), không có ý nghĩa ở Climate-FEVER, và kém hơn ở SciDocs (prompt mã nguồn). Hiệu ứng phụ thuộc bộ dữ liệu.
- Prompt của mã nguồn tốt hơn prompt bài báo có ý nghĩa ở 5/8 ô theo nDCG\@10. Cách diễn đạt prompt ảnh hưởng nhất quán hơn thứ tự đoạn văn.
- Trên FiQA, cấu hình theo mã nguồn (0.3522) gần bài báo (0.3639) nhất, gợi ý bài báo có thể đã chạy theo mã nguồn công bố.

=== Thiên lệch vị trí của LLM chấm điểm

Khi xáo trộn, mỗi danh sách xuất hiện đều ở mọi vị trí, nên có thể đo điểm trung bình gpt-4o-mini cho theo vị trí đoạn văn trong prompt (@tab:position).

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header([*Run*], [*Vị trí 1*], [*Vị trí 2*], [*Vị trí 3*], [*Vị trí 4*]),
    [FiQA test], [3.52], [3.09], [2.74], [2.10],
    [SciDocs test], [3.98], [3.68], [3.20], [2.49],
    [Climate-FEVER test], [2.87], [2.65], [2.51], [2.07],
    [HotpotQA test], [3.12], [2.56], [2.48], [2.27],
  ),
  caption: [Điểm trung bình gpt-4o-mini cho theo vị trí đoạn văn (thang 0–5)],
) <tab:position>

Đoạn văn đứng đầu được chấm cao hơn đoạn đứng cuối 0,8–1,5 điểm, ở cả 6 run (4 tập test và 2 tập train). Thiên lệch vị trí khi dùng LLM làm giám khảo đã được ghi nhận trong các nghiên cứu trước @zheng2023judging; ở đây nó ảnh hưởng trực tiếp tới trọng số trộn. Khi xáo trộn, thiên lệch này thành nhiễu trong trọng số. Khi cố định, hai đoạn BM25 luôn đứng đầu và OD luôn đứng cuối; ở FiQA, BM25 lại là tín hiệu mạnh nhất nên thứ tự cố định vô tình có lợi. Đây là lời giải thích hợp lý nhất cho kết quả QuDAR-llm cao trên FiQA của bài báo.

#figure(
  image("../assets/figures/fig6_position_bias.png", width: 90%),
  caption: [Thiên lệch vị trí của gpt-4o-mini trên các run],
) <fig:position>

=== DAT

Repo của tác giả không có DAT. Đồ án dựng lại prompt theo mô tả trong bài DAT @hsu2025dat. Kết quả khớp bài báo trong khoảng ±4% (@tab:main).

== Phân tích trọng số tối ưu

@fig:static và @fig:ub tái hiện Figure 2–4b của bài báo: chỉ số khi quét trọng số tĩnh theo từng trục, và khoảng cách giữa trọng số tĩnh với cận trên theo câu truy vấn.

#figure(
  image("../assets/figures/fig2_3_static_sweeps.png", width: 100%),
  caption: [Recall\@10 khi quét trọng số tĩnh (tái hiện Figure 2–3)],
) <fig:static>

#figure(
  image("../assets/figures/fig4b_upper_bounds.png", width: 90%),
  caption: [Static UB so với Query-wise UB (tái hiện Figure 4b)],
) <fig:ub>

@fig:opthist và @fig:heatmap cho thấy trọng số tối ưu phân tán rộng giữa các câu truy vấn. Vì Recall\@10 và nDCG\@10 chỉ nhận ít giá trị rời rạc, nhiều trọng số cho cùng một điểm; nếu ép chọn một giá trị (ví dụ gần 0.5) sẽ tạo đỉnh giả trên biểu đồ. Đồ án chia đều phần đếm của mỗi câu truy vấn cho mọi trọng số tối ưu, và bỏ các câu mà mọi trọng số cho cùng điểm.

#figure(
  image("../assets/figures/fig4a_optimal_hist.png", width: 90%),
  caption: [Phân bố trọng số tối ưu theo câu truy vấn (tái hiện Figure 4a)],
) <fig:opthist>

#figure(
  image("../assets/figures/fig5_optimal_heatmap.png", width: 90%),
  caption: [Heatmap trọng số tối ưu (tái hiện Figure 5)],
) <fig:heatmap>
