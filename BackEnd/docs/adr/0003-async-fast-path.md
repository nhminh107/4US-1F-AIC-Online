# ADR-0003: Fast Path dùng asyncio với AsyncElasticsearch và FAISS wrapped executor

## Status
Accepted

## Context
Fast Path cần chạy nhiều retrieval tool song song. Codebase dùng Elasticsearch và FAISS. FAISS không có async client.

## Decision
Dùng `asyncio.gather()` với `return_exceptions=True`. Hai quy ước bắt buộc:

1. **Elasticsearch**: dùng `AsyncElasticsearch` từ `elasticsearch[async]` — không dùng sync client trong async context.
2. **FAISS**: wrap bằng `loop.run_in_executor(None, ...)` — overhead chấp nhận được vì FAISS in-process và nhanh.

```python
results = await asyncio.gather(
    clip_search(query, top_k=200),
    ocr_search(ocr_text, top_k=100),
    return_exceptions=True  # một tool lỗi/timeout không crash toàn Fast Path
)
```

Timeout từng tool riêng lẻ bằng `asyncio.wait_for(tool(), timeout=X)`.

## Consequences
- Một tool lỗi hoặc timeout không block các tool còn lại.
- Cần migrate Elasticsearch client sang `AsyncElasticsearch` — một lần duy nhất.
- FAISS call vẫn là sync nhưng không block event loop nhờ executor.
- Mọi Retrieval Tool phải khai báo là `async def` — đây là contract bắt buộc.
