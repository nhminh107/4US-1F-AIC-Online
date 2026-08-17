# ADR-0001: Dùng Instructor để enforce structured output từ LLM

## Status
Accepted

## Context
Intent Extractor và Query Planner Agent đều phải nhận output từ LLM và ép về Pydantic schema cụ thể. Cần retry tự động khi LLM trả sai schema. Không muốn lock vào một LLM provider cụ thể.

## Decision
Dùng thư viện **Instructor** (pip: `instructor`) wrap trên top của raw API client.

## Consequences
- Validation error tự động được feed ngược vào LLM prompt khi retry — không cần tự viết retry loop.
- Schema chính là Pydantic class — không có layer riêng để duy trì.
- Dễ swap LLM provider (OpenAI / Gemini / Anthropic) chỉ cần đổi client.
- Không lock vào LangChain abstraction.
- Trade-off: thêm một dependency; team phải quen Instructor pattern.
