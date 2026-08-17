# Tổng hợp Grilling Session — Intent Extractor, Query Planner Agent, Fast Path, Retrieval Tools

## Scope bạn phụ trách
- Module 1: Intent Extractor
- Module 2A: Fast Retrieval Path
- Module 2B: Query Planner Agent
- Module 3: Retrieval Tools (Retrieval Executor)

**Ưu tiên v1**: KIS và VQA. TRAKE defer sang phase sau.

---

## 12 Quyết định đã chốt

---

### 1. LLM Framework — Instructor
Dùng `instructor` wrap trên raw API client để enforce Pydantic schema từ LLM output.
- Validation error tự động feed ngược vào retry prompt — không tự viết retry loop.
- Dễ swap LLM provider (OpenAI / Gemini / Anthropic).
- Không lock vào LangChain.

```python
import instructor
client = instructor.from_openai(OpenAI())

structured_query = client.chat.completions.create(
    model="gpt-4o",
    response_model=StructuredQuery,
    messages=[{"role": "user", "content": prompt}],
    max_retries=3,
)
```

**ADR**: `0001-instructor-for-structured-output.md`

---

### 2. Intent Extractor — Two-call pattern
Không dùng một prompt tổng hợp cho cả ba task.

- **Call 1** (nhẹ): classify task → `"KIS" | "VQA" | "TRAKE"`
- **Call 2** (chuyên biệt): dùng prompt riêng theo task để extract `StructuredQuery` đầy đủ

Lý do: prompt TRAKE (event decomposition + temporal constraint) rất khác prompt VQA. Nhét chung làm model hallucinate field không liên quan.

---

### 3. StructuredQuery — Discriminated union theo task

```python
class KISQuery(BaseModel):
    task: Literal["KIS"]
    visual_queries: list[str] = []
    ocr_constraints: list[str] = []
    asr_constraints: list[str] = []
    negative_constraints: list[str] = []

class VQAQuery(BaseModel):
    task: Literal["VQA"]
    question: str               # required
    visual_queries: list[str] = []
    ocr_constraints: list[str] = []

class TRAKEQuery(BaseModel):
    task: Literal["TRAKE"]
    events: list[Event]         # required
    temporal_constraints: list[TemporalConstraint]
    visual_queries: list[str] = []

StructuredQuery = Annotated[
    KISQuery | VQAQuery | TRAKEQuery,
    Field(discriminator="task")
]
```

Field required của từng task là truly required — Instructor retry ngay nếu thiếu.
Modules nhận `StructuredQuery` dùng `match` statement hoặc `isinstance` check.

**ADR**: `0002-structured-query-discriminated-union.md`

---

### 4. Query Planner Agent — Single-shot planner

Agent nhận `StructuredQuery`, output toàn bộ `ToolCall[]` trong **một LLM call duy nhất**. Không có multi-turn agent loop trong v1.

- Replan chỉ xảy ra khi có Feedback từ UI (Module 8) — không phải internal loop.
- Giúp kiểm soát budget/timeout dễ dàng.
- Ranh giới rõ: Agent *lập kế hoạch*, Retrieval Tools *thực thi*.

---

### 5. Tool Allow-list — `Literal` type trong Pydantic

```python
ToolName = Literal[
    "clip_search", "frame_search", "shot_search",
    "ocr_search", "asr_search", "caption_search",
    "object_search", "track_search",
]
```

LLM hallucinate tên tool không có trong Literal → Pydantic reject → Instructor retry ngay với error message chính xác. Không cần tự viết validation riêng.

---

### 6. ToolCall — Một class duy nhất với `model_validator`

Không dùng discriminated union (quá nhiều boilerplate). Thay vào đó:

```python
# Params schema nhỏ cho từng tool
class ClipSearchParams(BaseModel):
    query: str
    top_k: int = 200

class OCRSearchParams(BaseModel):
    query: str
    top_k: int = 100
    mode: Literal["exact", "fuzzy", "similarity"] = "similarity"

# Registry — thêm tool mới chỉ cần thêm 1 dòng
TOOL_PARAMS: dict[str, type[BaseModel]] = {
    "clip_search": ClipSearchParams,
    "ocr_search":  OCRSearchParams,
    # ...
}

# Một ToolCall duy nhất
class ToolCall(BaseModel):
    tool: ToolName
    parameters: dict
    event_id: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "ToolCall":
        schema = TOOL_PARAMS[self.tool]
        self.parameters = schema(**self.parameters).model_dump()
        return self
```

Khi execute:
```python
async def execute_tool(call: ToolCall) -> list[SearchHit]:
    match call.tool:
        case "clip_search":
            return await clip_search(**call.parameters)
        case "ocr_search":
            return await ocr_search(**call.parameters)
        # ...
```

**ADR**: `0004-toolcall-model-validator.md`

---

### 7. Fast Path — asyncio thuần

Hai quy ước bắt buộc:
- **Elasticsearch**: dùng `AsyncElasticsearch` (`elasticsearch[async]`) — không dùng sync client.
- **FAISS**: wrap bằng `loop.run_in_executor(None, ...)` — không block event loop.

```python
results = await asyncio.gather(
    asyncio.wait_for(clip_search(query, 200), timeout=TOOL_TIMEOUTS["clip_search"]),
    asyncio.wait_for(ocr_search(text, 100),  timeout=TOOL_TIMEOUTS["ocr_search"]),
    return_exceptions=True  # một tool lỗi không crash toàn Fast Path
)

hits: list[SearchHit] = []
for r in results:
    if isinstance(r, Exception):
        logger.warning("Tool failed", error=r)
    else:
        hits.extend(r)
```

**ADR**: `0003-async-fast-path.md`

---

### 8. Timeout — Policy của Fast Path, không phải của tool

```python
TOOL_TIMEOUTS: dict[str, float] = {
    "clip_search":   2.0,
    "frame_search":  2.0,
    "shot_search":   2.0,
    "ocr_search":    1.5,
    "asr_search":    1.5,
    "caption_search": 1.5,
}
```

Timeout đặt tại caller bằng `asyncio.wait_for()`. Tool không tự enforce timeout nội bộ — để caller quyết định budget.

---

### 9. Orchestration — `asyncio.gather()`, UI chờ final

Fast Path và Query Planner Agent chạy song song:

```python
async def run_pipeline(raw_query: RawQuery) -> PipelineResult:
    # Step 1: Intent Extraction
    structured_query = await extract_intent(raw_query)

    # Step 2: Fast Path + Query Planner song song
    fast_hits, tool_calls = await asyncio.gather(
        run_fast_path(structured_query),
        run_query_planner(structured_query),
    )

    # Step 3: Execute Agent tool calls
    agent_hits = await execute_tool_calls(tool_calls)

    # Step 4: Merge → Aggregator
    all_hits = fast_hits + agent_hits
    return all_hits
```

UI chờ kết quả final — không cần streaming infrastructure cho v1.

---

### 10. Fallback khi Instructor hết retry

**Intent Extractor fail** → fallback `KISQuery` tối giản từ raw text:
```python
except InstructorRetryException:
    logger.warning("Intent extraction failed, fallback to raw text", query=raw_query.text)
    return KISQuery(task="KIS", visual_queries=[raw_query.text])
```

**Query Planner Agent fail** → trả `[]`, Fast Path kết quả vẫn đi tiếp:
```python
except InstructorRetryException:
    logger.warning("Query planner failed, using Fast Path only")
    return []
```

Quy tắc bắt buộc: **không được silent fail** — mọi fallback phải log warning với đủ context.

**ADR**: `0005-fallback-strategy.md`

---

### 11. Retrieval Tools — Scope rõ ràng

Retrieval Tool **không** tự resolve FAISS ID → canonical reference. Việc đó thuộc Shared Data & Evidence Layer — tool chỉ gọi API của tầng đó.

Mỗi Retrieval Tool phải:
- Khai báo là `async def`
- Trả đúng contract `list[SearchHit]`
- Không tự JOIN bảng, không tự viết SQL

---

## Cấu trúc file gợi ý

```
online_pipeline/
├── intent_extractor/
│   ├── schemas.py          # KISQuery, VQAQuery, TRAKEQuery, StructuredQuery
│   ├── prompts.py          # prompt KIS / VQA / TRAKE
│   └── extractor.py        # two-call logic + fallback
├── query_planner/
│   ├── schemas.py          # ToolName, ToolCall, TOOL_PARAMS registry
│   ├── planner.py          # single-shot LLM call + fallback
│   └── executor.py         # execute_tool_calls() + match statement
├── fast_path/
│   ├── runner.py           # asyncio.gather() + TOOL_TIMEOUTS
│   └── merger.py           # gom SearchHit từ fast + agent
├── retrieval_tools/
│   ├── visual.py           # clip_search, frame_search, shot_search
│   ├── text.py             # ocr_search, asr_search, caption_search
│   └── object.py           # object_search, track_search
└── pipeline.py             # run_pipeline() — entry point
```

---

## ADR Index

| # | Quyết định |
|---|-----------|
| 0001 | Instructor cho structured output |
| 0002 | StructuredQuery discriminated union |
| 0003 | Fast Path asyncio + AsyncElasticsearch + FAISS executor |
| 0004 | ToolCall model_validator thay vì discriminated union |
| 0005 | Fallback strategy khi Instructor hết retry |
