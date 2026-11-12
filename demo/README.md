# Demo QuDAR trên FiQA (Step 8)

Trang web chạy trên laptop (CPU là đủ) minh hoạ đủ 6 khâu của hệ truy xuất cho một câu hỏi: phân tích tài liệu, lập chỉ mục, phân tích câu truy vấn, truy xuất 4 danh sách (OS, OD, ES, ED), xếp hạng bằng mọi phương pháp gán trọng số (QuDAR của paper và các mở rộng của Step 6), và đánh giá khi câu hỏi có nhãn. Có thêm tab sinh câu trả lời (RAG).

| Tab | Nội dung |
|---|---|
| Kết quả cuối | top-10 sau khi trộn theo phương pháp đang chọn; thanh màu cho biết mỗi danh sách đóng góp bao nhiêu điểm; tài liệu có nhãn liên quan được đánh dấu |
| So sánh phương pháp | trọng số (w_OS, w_OD, w_ES, w_ED và α, β1, β2) mà từng phương pháp gán cho câu này, kèm Recall@10, nDCG@10, P@1 |
| 4 danh sách | top-10 của từng danh sách, điểm thô và điểm sau chuẩn hoá min-max |
| Quy trình & thời gian | 6 khâu áp dụng cho câu này, thời gian từng công đoạn, kiểm tra khớp với Step 2 và Step 6 |
| Sinh câu trả lời (RAG) | câu trả lời khi đưa top-5 của OS và của phương pháp đang chọn cho LLM |

## Chế độ chạy

- **Câu hỏi test của FiQA (648 câu, có nhãn):** chạy hoàn toàn offline. Query mở rộng lấy từ Step 1 (LLaMA 3.1-8B), điểm LLM từ Step 4 (gpt-4o-mini), câu trả lời từ Step 7 (Qwen2.5-7B, 500 câu). Bật "Dùng 4 danh sách đã lưu từ Step 2" để ra đúng số của bảng kết quả.
- **Câu tự nhập:** cần một LLM để sinh query mở rộng. Có hai lựa chọn: OpenAI gpt-4o-mini (khoảng 0,001 USD mỗi câu, tính cả chấm điểm và RAG), hoặc Ollama chạy trên laptop.

## Cài đặt (Windows, PowerShell)

```powershell
cd <thư mục repo>\demo
py -3.11 -m venv .venv            # Python 3.10–3.12
.\.venv\Scripts\Activate.ps1      # nếu bị chặn: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Giải nén `demo_data.zip` (tạo bởi `scripts/pack_demo_data.py` trên Kaggle) vào `demo\demo_data\`. Muốn dùng OpenAI thì chép `.env.example` thành `.env` rồi điền key.

```powershell
python app.py              # mở http://127.0.0.1:7860
python selftest.py         # kiểm tra laptop tái lập đúng Step 2 và Step 6 (5–10 phút)
```

Lần chạy đầu tải Contriever (khoảng 440 MB) từ Hugging Face và dựng chỉ mục BM25 (dưới 1 phút). Các lần sau chỉ mất vài giây để mở.

## Ollama (tuỳ chọn, không cần OpenAI)

Cài Ollama từ ollama.com, rồi chạy `ollama pull llama3.1:8b` (mở rộng truy vấn, cùng họ model với paper) và `ollama pull qwen2.5:7b` (RAG, cùng model với Step 7). Bản 4-bit cần khoảng 5 GB RAM. Có GPU thì mỗi lần sinh mất vài giây; chỉ có CPU thì mất 30–90 giây.

## Khác biệt so với thí nghiệm

- Query mở rộng sinh trực tiếp (gpt-4o-mini, hoặc LLaMA 4-bit qua Ollama) khác cấu hình của Step 1 (LLaMA 3.1-8B fp16, vLLM). Chỉ câu test dùng đúng query mở rộng của thí nghiệm.
- Retriever-Type Fusion, Query-Format Fusion và Query-wise UB là oracle chọn trên chính tập test, chỉ để tham chiếu.
- Điểm LLM đã lưu được chấm trên top-1 của danh sách Step 2. Khi truy xuất trực tiếp, top-1 có thể khác đôi chút; tab "Quy trình" cho biết mức trùng.
