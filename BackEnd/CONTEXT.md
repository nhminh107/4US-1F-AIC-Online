# CONTEXT — AIC 2026 Online Pipeline

## Glossary

### StructuredQuery
Output chuẩn hóa của Intent Extractor. Là Pydantic model duy nhất truyền ý nghĩa query qua toàn bộ pipeline. Mọi module phía sau chỉ đọc StructuredQuery, không re-parse raw query.

### SearchHit
Contract chuẩn mà mọi Retrieval Tool phải trả về. Bắt buộc có: `source`, `entity_type`, `entity_id`, `video_id`, `start_ms`, `end_ms`, `rank`, `raw_score`. Không có field này → tool không được tích hợp.

### CandidateRegion
Đơn vị gom evidence sau Aggregator. Nhiều SearchHit từ cùng vùng thời gian trong cùng video_id được merge thành một CandidateRegion. Không merge giữa các video hay giữa các event_id khác nhau.

### EvidenceBundle
Tập hợp đa-modality (frame, OCR, ASR, caption, object, track) gắn với một vùng thời gian cụ thể. Chỉ được tạo bởi Shared Data & Evidence Layer, không tự tạo trong Online modules.

### event_id
Định danh sự kiện trong một TRAKE query (E1, E2, ...). Phải được giữ nguyên xuyên suốt từ Intent Extractor → ToolCall → SearchHit → CandidateRegion → TRAKE Aligner. Mất event_id ở bất kỳ bước nào là lỗi contract.

### Tool Timeout Policy
Timeout là policy của Fast Path, không phải implementation detail của từng tool. Đặt bằng `asyncio.wait_for()` tại caller, giá trị cấu hình trong `TOOL_TIMEOUTS` dict tập trung. Mỗi tool có thể có timeout khác nhau (FAISS nhanh hơn Elasticsearch).

### Pipeline Orchestration (v1)
Fast Path và Query Planner Agent chạy song song bằng `asyncio.gather()`. UI chờ kết quả final — không streaming. Kết quả của cả hai được merge tại Candidate Aggregator.

### ToolCall
Output của Query Planner Agent. Field `tool` là `Literal` type liệt kê toàn bộ tool hợp lệ — không dùng `str` tự do. Instructor retry tự động khi LLM hallucinate tên tool không có trong Literal.

### Query Planner Agent (single-shot planner)
Nhận StructuredQuery, output toàn bộ ToolCall[] trong một LLM call duy nhất. Không thực hiện multi-turn agent loop trong v1. Replan chỉ xảy ra khi có Feedback từ UI (Module 8), không phải internal loop. TRAKE multi-event planning bị defer sang phase sau.

### Intent Extractor (two-call pattern)
Bước 1: LLM call nhẹ classify task (KIS / VQA / TRAKE). Bước 2: LLM call chuyên biệt theo task để extract StructuredQuery đầy đủ. Không dùng một prompt chung cho cả ba task.

### canonical entity reference
Bộ ID chuẩn hóa gồm video_id, shot_id, clip_id, frame_id, start_ms, end_ms. Mọi FAISS ID hay Elasticsearch doc ID phải được resolve về canonical reference trước khi trả lên Online layer.
