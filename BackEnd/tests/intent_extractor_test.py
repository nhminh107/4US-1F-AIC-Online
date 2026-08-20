from __future__ import annotations

from BackEnd.app.contracts.models import RawQuery, StructuredQuery
from BackEnd.app.intent_extractor.extractor import extract_intent_sync
from BackEnd.app.intent_extractor.schemas import TaskClassification


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, *, response_model, **kwargs):
        payload = self._responses.pop(0)
        if response_model is TaskClassification:
            return TaskClassification.model_validate(payload)
        if response_model is StructuredQuery:
            return StructuredQuery.model_validate(payload)
        raise AssertionError(f"Unexpected response model: {response_model!r}")


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)


def test_extract_intent_kis_uses_app_structured_query_contract():
    raw_query = RawQuery(text="Tìm cảnh người đàn ông áo đỏ đứng cạnh bảng HCMC")
    client = _FakeClient(
        [
            {"task": "KIS"},
            {
                "query_id": "wrong_id",
                "task": "KIS",
                "visual_queries": ["người đàn ông áo đỏ đứng cạnh bảng HCMC"],
                "ocr_constraints": ["HCMC"],
            },
        ]
    )

    result = extract_intent_sync(raw_query, client=client)

    assert isinstance(result, StructuredQuery)
    assert result.task == "KIS"
    assert result.query_id.startswith("query_")
    assert result.visual_queries == ["người đàn ông áo đỏ đứng cạnh bảng HCMC"]
    assert result.ocr_constraints == ["HCMC"]


def test_extract_intent_vqa_preserves_question():
    raw_query = RawQuery(
        query_id="q-vqa-001",
        text="Người phụ nữ trong video đang làm gì?",
    )
    client = _FakeClient(
        [
            {"task": "VQA"},
            {
                "query_id": "wrong_id",
                "task": "VQA",
                "question": "Người phụ nữ trong video đang làm gì?",
                "visual_queries": ["người phụ nữ"],
            },
        ]
    )

    result = extract_intent_sync(raw_query, client=client)

    assert isinstance(result, StructuredQuery)
    assert result.task == "VQA"
    assert result.query_id == "q-vqa-001"
    assert result.question == "Người phụ nữ trong video đang làm gì?"
    assert result.visual_queries == ["người phụ nữ"]
