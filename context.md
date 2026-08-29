# AIC HCMC 2026 - System Context & Architecture

Tài liệu này cung cấp cái nhìn tổng quan toàn diện về hệ thống, cấu trúc thư mục, cơ sở dữ liệu và các luồng xử lý chính trong dự án **4US-1F-AIC-Online**, giúp một người mới có thể đọc và hiểu ngay toàn bộ cách hệ thống hoạt động.

---

## 1. Mục tiêu hệ thống
Hệ thống được thiết kế dưới dạng một **Online Pipeline** cho cuộc thi AI Challenge (AIC) HCMC 2026, nhằm giải quyết 3 bài toán truy xuất video chính:
1. **Textual KIS (Known Item Search):** Tìm chính xác đoạn video/khung hình dựa trên mô tả văn bản.
2. **VQA (Visual Question Answering):** Trả lời câu hỏi dựa trên ngữ cảnh hình ảnh/video tìm được.
3. **TRAKE (Temporal Retrieval and Alignment of Key Events):** Truy xuất và căn chỉnh chuỗi sự kiện theo thời gian thực trong video.

---

## 2. Kiến trúc & Cấu trúc thư mục (Directory Structure)

Hệ thống được chia thành các thành phần chính:

- **`BackEnd/app/`**: Chứa toàn bộ mã nguồn xử lý logic chính của FastAPI.
  - **`api/`**: Cấu hình FastAPI, định nghĩa các routers/endpoints (`routes.py`), schema Pydantic (`models.py`) và pipeline chính (`pipeline.py`).
  - **`Database/`**: Định nghĩa cấu trúc cơ sở dữ liệu (PostgreSQL) thông qua SQLAlchemy (`sql_models.py`).
  - **`intent_extractor/` & `query_planner/`**: Module phân tích ý định truy vấn của người dùng, sử dụng Agent/LLM để phân rã câu truy vấn phức tạp thành các bước xử lý (Proposal 2).
  - **`retrieval/` & `retrieval_tools/`**: Các công cụ kết nối với FAISS và Elasticsearch để truy xuất dữ liệu (ảnh, text, OCR, ASR).
  - **`aggregator.py`**: Module gộp các kết quả thô (khung hình, shot, clip) từ nhiều nguồn về một chuẩn chung gọi là `CandidateRegion`.
  - **`Fusion/`**: Chịu trách nhiệm rerank, hợp nhất (Hybrid Fusion) điểm số từ nhiều phương pháp truy xuất khác nhau (semantic text, visual, v.v.).
  - **`KIS/`, `vqa/`, `trake/`**: Các handlers riêng biệt xử lý output cuối cùng cho từng loại bài toán.
  - **`verification/`**: (Selective Verifier) Sử dụng Vision-Language Models (VLM) để đánh giá lại các top candidates nhằm tăng độ chính xác.

- **`scripts/`**: Chứa các đoạn script chạy độc lập (offline/batch) như `solve_batch_queries.py`, `build_submissions.py`, `test_local_multimodal.py` dùng để test local, đánh giá benchmark hoặc format file nộp bài (submission).

- **`docker-compose.runtime.yml`**: Tệp định nghĩa cơ sở hạ tầng (Postgres, Elasticsearch, và API Server) dùng chung trong quá trình runtime.

---

## 3. Hạ tầng dữ liệu (Database & Storage)

Hệ thống sử dụng kiến trúc lưu trữ đa thành phần nhằm tối ưu hoá cho từng bài toán tìm kiếm:

### A. PostgreSQL (Metadata & Relational Data)
Được định nghĩa trong `BackEnd/app/Database/sql_models.py`. Lưu trữ cấu trúc phân tầng của video:
- **`Video`**: Bảng gốc lưu metadata của video (title, description, duration, url).
- **`Shot`**: Đoạn video liên tục không bị cắt cảnh (start_ms, end_ms).
- **`ClipWindow`**: Đoạn cắt thời gian nhỏ hơn (ví dụ sliding windows) dùng để trích xuất đặc trưng FAISS.
- **`Frame`**: Chứa các keyframe trích xuất từ video (hoặc frame chính thức do BTC cung cấp). Có liên kết tới Shot và Video.
- **`OCR`**: Chứa text trích xuất từ hình ảnh tại các khung hình kèm bounding box (`x_min`, `y_min`, `x_max`, `y_max`).
- **`ObjectDetection` & `ObjectTrack`**: Quản lý bounding box của các vật thể do mô hình YOLO/Detection nhận diện, cũng như chuỗi di chuyển của vật thể (Track) qua các khung hình.
- **`TranscriptSegment`**: Lưu các câu thoại (ASR) được nhận diện theo thời gian của video.
- **`Caption`**: Lưu mô tả chi tiết của từng frame/clip/shot sinh ra bởi VLM (Vision-Language Model).
- **Các bảng FAISS Mapping** (`FrameEmbeddingRecord`, `ClipEmbeddingRecord`, `ShotEmbeddingRecord`): Quản lý sự ánh xạ (mapping) giữa ID của vector trong FAISS và thực thể tương ứng trong CSDL.

### B. Elasticsearch (Textual Search)
Lưu trữ nội dung văn bản (OCR, ASR, Caption) giúp truy vấn nhanh chóng dựa trên từ khoá. (Ví dụ: `match_phrase`, `match`, fuzzy search). Thường được ứng dụng nhiều nhất cho các truy vấn KIS hoặc TRAKE có chứa từ vựng chi tiết như *"gỏi cuốn chay"*, *"hổ con"*.

### C. FAISS (Visual/Vector Search)
Chứa các embeddings (vector đặc trưng) của:
- Ảnh (Frame-level)
- Shot (Shot-level)
- Clip (Clip-level)
Được mã hóa qua các mô hình như CLIP (`clip-ViT-B-32`). Dùng cho truy xuất ngữ nghĩa hình ảnh (Visual Retrieval).

---

## 4. Luồng xử lý Pipeline (Flowchart Tóm tắt)

Luồng hoạt động của hệ thống được quy định chi tiết tại `ProposalOnlinePipeline.md`:

1. **User Query:** Nhận truy vấn bằng text hoặc ảnh từ người dùng.
2. **Intent Extractor / Query Planner:** Phân tích xem câu hỏi thuộc dạng KIS, VQA hay TRAKE. Nếu phức tạp, Agent sẽ lập kế hoạch tách câu (ví dụ: tìm "hổ con" + "Thảo Cầm Viên").
3. **Retrieval Tools:** 
   - Nếu truy vấn mang tính hình ảnh/ngữ nghĩa: Gọi FAISS (Visual Retrieval).
   - Nếu mang tính văn bản/từ khóa (OCR/ASR): Gọi Elasticsearch (Text Retrieval).
4. **Candidate Aggregator:** Gom tất cả các hits thô trả về và gộp thành các `CandidateRegion` (đại diện cho một đoạn thời gian tiềm năng trong video).
5. **Hybrid Fusion & Ranking:** Rerank lại các Region dựa trên tổng điểm, khoảng cách thời gian.
6. **Task Handlers (KIS/VQA/TRAKE):** 
   - **KIS**: Trích xuất ra khung hình đại diện tốt nhất cho đoạn video.
   - **VQA**: Nạp các khung hình tốt nhất vào VLM để tạo ra câu trả lời (Text).
   - **TRAKE**: Sắp xếp và căn chỉnh các sự kiện nối tiếp nhau.
7. **Verification & UI:** Đưa qua Verification Model để kiểm chứng chéo và gửi kết quả về UI.

---

## 5. Các lưu ý về kĩ thuật (Technical Notes)
- Ứng dụng giao tiếp qua cổng `8000` (FastAPI), DB PostgreSQL trên `5432`, Elasticsearch trên `9200`.
- Scripts tạo submission (VD: `solve_batch_queries.py`, `build_submissions.py`) sẽ gọi trực tiếp vào DB, ES, hoặc FAISS offline mà không cần thông qua API để tối ưu tốc độ cho quá trình đánh giá (Benchmark).
- Tất cả timestamps (`start_ms`, `end_ms`, `timestamp_ms`) đều tính bằng `milliseconds`.
- Kết quả trả về cho VQA phải được chuẩn hoá (chữ thường, không khoảng trắng/ký tự đặc biệt) theo yêu cầu định dạng gắt gao của BTC.
