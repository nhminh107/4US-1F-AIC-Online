# ADR-0002: StructuredQuery dùng Pydantic discriminated union theo task

## Status
Accepted

## Context
Ba task KIS / VQA / TRAKE có required field khác nhau hoàn toàn (TRAKE cần `events`, VQA cần `question`). Cần enforce tại parse time, không để lỗi lan xuống module phía sau.

## Decision
`StructuredQuery` là discriminated union:
```python
StructuredQuery = Annotated[
    KISQuery | VQAQuery | TRAKEQuery,
    Field(discriminator="task")
]
```
Mỗi subclass chỉ chứa field liên quan. Field required của từng task là truly required, không phải `Optional`.

## Consequences
- Instructor sẽ retry ngay khi TRAKE query thiếu `events` — không để lỗi chạy xuống TRAKE Aligner.
- Modules nhận `StructuredQuery` phải dùng `isinstance` check hoặc `match` statement — đây là sự thật của bài toán, không nên che giấu.
- Trade-off: thêm boilerplate (3 class thay vì 1), nhưng type safety xứng đáng.
