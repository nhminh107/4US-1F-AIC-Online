# Run Guide

## 1. Vào đúng thư mục

```bash
cd "/Users/thanhbao/Desktop/AI_Challenge/Online Process/4US-1F-AIC-Online/BackEnd"
```

## 2. Tạo môi trường Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pydantic openai instructor "elasticsearch[async]"
```

Repository hiện chưa có `pyproject.toml` hoặc `requirements.txt`, nên dependency cần được cài thủ công cho tới khi bổ sung manifest.

## 3. Kiểm tra syntax

```bash
python -m compileall -q online_pipeline
```

## 4. Smoke test tối thiểu

```bash
python - <<'PY'
import asyncio

from online_pipeline.intent_extractor.schemas import VQAQuery
from online_pipeline.pipeline import RawQuery
from online_pipeline.query_planner.schemas import ToolCall

query = VQAQuery(task="VQA", question="Who is visible?")
call = ToolCall(tool="clip_search", parameters={"query": "person"})
raw = RawQuery(text="Who is visible?", session_id="smoke")

assert query.task == "VQA"
assert call.parameters["top_k"] == 200
assert raw.session_id == "smoke"

async def main():
    module = __import__("online_pipeline.pipeline", fromlist=["run_pipeline"])
    assert asyncio.iscoroutinefunction(module.run_pipeline)

asyncio.run(main())
print("smoke test passed")
PY
```

Smoke test này chỉ kiểm tra import và contracts; nó không gọi LLM hoặc backend thật.

## 5. Cấu hình LLM thật

```bash
export OPENAI_API_KEY="<your-api-key>"
```

`LLM_CONFIG` mặc định:

```text
model_name = gpt-4o
max_retries = 3
temperature = 0.0
```

Client được tạo bởi `instructor.from_openai(OpenAI())` trong Intent Extractor và Query Planner.

## 6. Cấu hình retrieval backend

Trước khi gọi pipeline thật, cần implement hoặc inject:

- `embed_text(query)`.
- `search_faiss(index_name, vector, top_k)`.
- `resolve_entity(faiss_id, index_name)`.
- Elasticsearch indexes và mappings cho `ocr_index`, `asr_index`, `caption_index`, `object_detection_index`, `tracking_index`.

Text/object modules tạo `AsyncElasticsearch` một lần ở module level. Visual module không tự truy cập SQL; canonical reference phải đi qua `resolve_entity()`.

## 7. Gọi pipeline

```python
import asyncio

from online_pipeline.pipeline import RawQuery, run_pipeline

async def main():
    results = await run_pipeline(
        RawQuery(
            text="Tìm cảnh có người mặc áo đỏ",
            session_id="demo-session",
        )
    )
    for hit in results:
        print(hit.model_dump())

asyncio.run(main())
```

## 8. Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Pipeline log số lượng hit từ Fast Path, số `ToolCall` từ planner, số agent hits và tổng số hit sau merge. Retrieval/timeout failures được log và bỏ qua theo ADR.

## 9. Lưu ý trước khi chạy live

- Không chạy live pipeline nếu các placeholder backend chưa được implement.
- `frame_search` chưa được bật trong Fast Path vì chưa có config flag.
- Planner hiện allow-list `object_search`/`track_search`, nhưng executor và timeout registry cần được mở rộng trước khi thực thi hai tool này qua agent.
- Pipeline hiện chưa có Candidate Aggregation, deduplication hoặc Ranking.
