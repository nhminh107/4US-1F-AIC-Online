# ADR-0004: ToolCall dùng một class duy nhất với model_validator thay vì discriminated union

## Status
Accepted

## Context
Cần validate parameters của từng tool (tránh LLM hallucinate sai key) nhưng không muốn boilerplate của discriminated union với nhiều ToolCall subclass.

## Decision
Một class `ToolCall` duy nhất. Parameters validate qua `@model_validator(mode="after")` dựa vào `TOOL_PARAMS` registry:

```python
TOOL_PARAMS: dict[str, type[BaseModel]] = {
    "clip_search": ClipSearchParams,
    "ocr_search":  OCRSearchParams,
    ...
}

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

Khi execute: `await some_tool(**call.parameters)` — clean vì parameters đã normalized.

## Consequences
- Thêm tool mới = thêm Params class nhỏ + một dòng trong TOOL_PARAMS — không động ToolCall schema.
- Instructor vẫn retry targeted vì ValidationError xảy ra trong model_validator trước khi trả kết quả.
- Trade-off: `parameters` vẫn là `dict` trên type level — IDE không autocomplete. Chấp nhận được vì execution code dùng `**unpack`.
