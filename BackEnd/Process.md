# Online Pipeline Process

## Phạm vi

Đã dựng skeleton v1 cho Online Pipeline video retrieval. Intent Extractor hỗ trợ
`KIS`, `VQA` và `TRAKE`; TRAKE Temporal Aligner phía sau vẫn là phần cần hoàn thiện.

## Kết quả đã thực hiện

### Shared contracts và config

- `SearchHit` là contract chung cho mọi retrieval tool.
- `TOOL_TIMEOUTS`, `TOP_K_DEFAULTS` và `LLM_CONFIG` tập trung toàn bộ policy runtime.

### Intent Extractor

- `StructuredQuery` là contract chung theo `task`.
- `SearchHit` được import từ shared contracts.
- Prompt single-shot inject schema bằng `model_json_schema()` và yêu cầu trả trực tiếp
  `StructuredQuery` cho KIS, VQA hoặc TRAKE.
- Instructor bọc OpenAI client, chạy một LLM call cho mỗi `RawQuery`.
- `RawQuery.feedback` được đưa vào prompt, giữ nguyên trong `StructuredQuery.feedback`,
  và được ưu tiên khi mâu thuẫn với truy vấn gốc.
- Khi hết retry, log warning và fallback về StructuredQuery KIS từ raw text, vẫn giữ feedback.
- Extractor nhận cả chuỗi và object có thuộc tính `.text`.

### Query Planner

- `ToolName` là allow-list Literal cho 8 tool.
- `ToolCall` là một class duy nhất, validate parameters qua `TOOL_PARAMS`.
- Mỗi tool có Params schema riêng.
- Planner chạy single-shot với `response_model=list[ToolCall]`.
- Prompt mô tả tool, quy tắc chọn tool theo KIS/VQA/TRAKE và cách chọn `top_k`.
- Planner fallback về `[]` khi `InstructorRetryException`.

### Retrieval tools

- Visual: `clip_search`, `frame_search`, `shot_search`.
- Text: `ocr_search`, `asr_search`, `caption_search`.
- Object/tracking: `object_search`, `track_search`.
- Tất cả map kết quả về `SearchHit` và giữ `event_id`.
- Visual tool có placeholder cho embedding, FAISS và Shared Data Layer resolve.
- Text tools dùng một module-level `AsyncElasticsearch` client; object/tracking
  tools query trực tiếp PostgreSQL qua `PostgreManager`.

### Fast Path và orchestration

- Fast Path chỉ gọi modality có query tương ứng.
- Mỗi call được bọc `asyncio.wait_for()` và chạy song song bằng `asyncio.gather(..., return_exceptions=True)`.
- Query Planner và Fast Path chạy song song.
- Planner tool calls được execute song song và kết quả được merge.
- `pipeline.py` chỉ điều phối; chưa làm Candidate Aggregation hoặc Ranking.

## Flow runtime

```text
RawQuery -> Intent Extractor -> StructuredQuery
                              +--------------+
                              |              |
                              v              v
                         Fast Path      Query Planner
                              |              |
                              |          Tool Executor
                              +-------> SearchHit[]
                                         |
                                         v
                                  fast_hits + agent_hits
```

## Trạng thái backend

Các điểm sau vẫn là placeholder và cần nối hệ thống thật:

- `retrieval_tools.visual.embed_text()`.
- `retrieval_tools.visual.search_faiss()`.
- `retrieval_tools.visual.resolve_entity()`.
- Text embedding trong `retrieval_tools.text`.
- Elasticsearch indexes và mapping thực tế.
- Shared Data Layer canonical ID resolver.

Môi trường phát triển hiện tại chưa có `instructor`, `openai` và `elasticsearch[async]`; các module dùng import lazy hoặc guarded để compile/smoke-test được.

## Kiểm thử đã thực hiện

- `compileall` cho toàn bộ package.
- Smoke test schemas, prompt injection, extractor two-call/fallback.
- Smoke test planner, executor, timeout, error isolation và `event_id`.
- Smoke test visual/text/object retrieval bằng fake backend.
- Smoke test pipeline orchestration và merge result.

## Gaps cần xử lý tiếp

- Thêm dependency manifest và cấu hình môi trường.
- Implement embedding model, FAISS search và canonical resolver.
- Cấu hình Elasticsearch endpoint, credentials, mappings và indexes.
- Bổ sung executor/timeout policy cho `object_search` và `track_search` nếu đưa hai tool này vào agent execution.
- Quyết định config chính thức cho `frame_search` trong Fast Path.
- Đồng bộ các contract cũ còn duplicate trong `shared/contracts.py`.
- Viết test suite chính thức thay cho smoke tests thủ công.
