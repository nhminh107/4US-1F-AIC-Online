from __future__ import annotations

from BackEnd.app.contracts.models import RawQuery, StructuredQuery
from BackEnd.app.intent_extractor import extractor as extractor_module
from BackEnd.app.intent_extractor.extractor import extract_intent_sync


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, *, response_model, **kwargs):
        self.calls.append({"response_model": response_model, **kwargs})
        payload = self._responses.pop(0)
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
    assert result.feedback == []
    assert len(client.chat.completions.calls) == 1
    assert client.chat.completions.calls[0]["response_model"] is StructuredQuery


def test_extract_intent_vqa_preserves_question():
    raw_query = RawQuery(
        query_id="q-vqa-001",
        text="Người phụ nữ trong video đang làm gì?",
    )
    client = _FakeClient(
        [
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
    assert len(client.chat.completions.calls) == 1


def test_extract_intent_preserves_feedback_and_sends_it_to_the_llm():
    feedback = "Áo màu xanh, không phải màu đỏ"
    raw_query = RawQuery(
        query_id="q-feedback-001",
        text="Tìm người mặc áo đỏ",
        feedback=feedback,
    )
    client = _FakeClient(
        [
            {
                "query_id": "wrong_id",
                "task": "KIS",
                "visual_queries": ["người mặc áo xanh"],
                "negative_constraints": ["áo đỏ"],
                "feedback": [feedback],
            }
        ]
    )

    result = extract_intent_sync(raw_query, client=client)

    assert result.query_id == "q-feedback-001"
    assert result.visual_queries == ["người mặc áo xanh"]
    assert result.negative_constraints == ["áo đỏ"]
    assert result.feedback == [feedback]
    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert feedback in prompt


def test_extract_intent_falls_back_to_kis_and_preserves_feedback_on_retry(
    monkeypatch,
):
    class RetryError(Exception):
        pass

    class FailingCompletions:
        def create(self, **kwargs):
            raise RetryError("LLM unavailable")

    class FailingClient:
        class chat:
            completions = FailingCompletions()

    monkeypatch.setattr(extractor_module, "InstructorRetryException", RetryError)
    result = extract_intent_sync(
        RawQuery(
            text="Tìm người mặc áo đỏ",
            feedback="Ưu tiên cảnh ở ngoài trời",
        ),
        client=FailingClient(),
    )

    assert result.task == "KIS"
    assert result.visual_queries == ["Tìm người mặc áo đỏ"]
    assert result.feedback == ["Ưu tiên cảnh ở ngoài trời"]


def test_extract_intent_supports_trake_with_events_and_temporal_constraints():
    client = _FakeClient(
        [
            {
                "query_id": "wrong_id",
                "task": "TRAKE",
                "events": [
                    {"event_id": "E1", "description": "người đàn ông bước lên sân khấu"},
                    {"event_id": "E2", "description": "người đàn ông nhận huy chương"},
                    {"event_id": "E3", "description": "người đàn ông khóc"},
                ],
                "temporal_constraints": [
                    {"before": "E1", "after": "E2"},
                    {"before": "E2", "after": "E3"},
                ],
            }
        ]
    )

    result = extract_intent_sync(
        "Tìm video người đàn ông bước lên sân khấu, nhận huy chương rồi khóc",
        client=client,
    )

    assert result.task == "TRAKE"
    assert [event.event_id for event in result.events] == ["E1", "E2", "E3"]
    assert [(item.before, item.after) for item in result.temporal_constraints] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]
    assert len(client.chat.completions.calls) == 1


def test_extract_intent_normalizes_and_limits_object_classes():
    client = _FakeClient(
        [
            {
                "query_id": "wrong_id",
                "task": "KIS",
                "visual_queries": ["xe buýt đi trong thành phố"],
                "object_constraints": [
                    "xe buýt",
                    "đường phố",
                    "tòa nhà",
                    "Person",
                ],
            }
        ]
    )

    result = extract_intent_sync(
        "Tìm cho tôi khung cảnh xe buýt đi trong thành phố",
        client=client,
    )

    assert result.object_constraints == ["bus", "person"]
    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "object_constraints may contain only" in prompt
    assert "bus" in prompt
    assert "building" not in result.object_constraints


def test_forced_trake_fallback_parses_event_markers_without_colons(monkeypatch):
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("intent model unavailable")

    class FailingClient:
        class chat:
            completions = FailingCompletions()

    result = extract_intent_sync(
        RawQuery(
            text=(
                "Mở đầu là đầu lân trắng cạnh lá cờ. "
                "E1 Hai con rồng vàng xoay vòng. "
                "E2 Con lân hoàn tất cú xoay trên trụ. "
                "E3 Dùi chạm vào kẻng đồng."
            ),
            task_hint="TRAKE",
        ),
        client=FailingClient(),
    )

    assert result.task == "TRAKE"
    assert [event.event_id for event in result.events] == ["E1", "E2", "E3"]
    assert [
        (constraint.before, constraint.after)
        for constraint in result.temporal_constraints
    ] == [("E1", "E2"), ("E2", "E3")]
    assert result.visual_queries == ["Mở đầu là đầu lân trắng cạnh lá cờ."]


def test_forced_kis_fallback_does_not_become_vqa_because_of_question_mark():
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("intent model unavailable")

    class FailingClient:
        class chat:
            completions = FailingCompletions()

    result = extract_intent_sync(
        RawQuery(text="KIS: Người này đang cầm gì?", task_hint="KIS"),
        client=FailingClient(),
    )

    assert result.task == "KIS"
    assert result.question == ""
