# ADR-0005: Fallback strategy khi Instructor hết retry

## Status
Accepted

## Context
Intent Extractor và Query Planner Agent có thể fail sau max_retries. Cần quyết định pipeline làm gì tiếp thay vì crash.

## Decision

**Intent Extractor fail**: fallback về `KISQuery` tối giản từ raw text.
```python
except InstructorRetryException:
    logger.warning("Intent extraction failed, using raw text fallback", query=raw_query.text)
    return KISQuery(task="KIS", visual_queries=[raw_query.text])
```

**Query Planner Agent fail**: trả `[]`, Fast Path kết quả vẫn đi tiếp bình thường.
```python
except InstructorRetryException:
    logger.warning("Query planner failed, using Fast Path only")
    return []
```

## Consequences
- Pipeline không bao giờ crash hoàn toàn vì LLM fail — luôn có ít nhất Fast Path result.
- Mọi fallback đều phải log warning với đủ context để debug.
- Không được silent fail — log là bắt buộc, không phải optional.
- Trade-off: kết quả có thể kém hơn nhưng system vẫn trả response cho người dùng.
