from __future__ import annotations

import asyncio

from BackEnd.app.contracts.models import SearchHit, StructuredQuery, ToolCall
from BackEnd.app.query_planner.executor import execute_tool_calls
from BackEnd.app.query_planner.planner import _run_query_planner_sync
from BackEnd.app.query_planner.schemas import PlannedToolCall


class _FakeCompletions:
    def create(self, *, response_model, **kwargs):
        assert response_model == list[PlannedToolCall]
        return [
            PlannedToolCall.model_validate(
                {
                    "tool": "clip_search",
                    "parameters": {"query": "red shirt"},
                }
            ),
            PlannedToolCall.model_validate(
                {
                    "tool": "ocr_search",
                    "parameters": {"query": "HCMC", "mode": "exact"},
                    "event_id": "event-1",
                }
            ),
        ]


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_run_query_planner_returns_app_toolcall_contract():
    query = StructuredQuery(
        query_id="q1",
        task="KIS",
        visual_queries=["red shirt"],
        ocr_constraints=["HCMC"],
    )

    tool_calls = _run_query_planner_sync(query, client=_FakeClient())

    assert [call.tool_call_id for call in tool_calls] == ["tc_001", "tc_002"]
    assert [call.tool_name for call in tool_calls] == ["clip_search", "ocr_search"]
    assert tool_calls[0].parameters["top_k"] == 200
    assert tool_calls[1].parameters["mode"] == "exact"
    assert tool_calls[1].event_id == "event-1"


def test_execute_tool_calls_dispatches_and_preserves_call_context(monkeypatch):
    async def fake_clip_search(
        query: str,
        top_k: int,
        event_id: str | None,
        tool_call_id: str | None,
    ):
        assert query == "red shirt"
        assert top_k == 5
        assert event_id == "event-1"
        assert tool_call_id == "tc_001"
        return [
            SearchHit(
                source="clip_embedding",
                entity_type="clip",
                entity_id="clip-1",
                video_id="video-1",
                start_ms=0,
                end_ms=100,
                rank=1,
                raw_score=0.9,
            )
        ]

    import BackEnd.app.retrieval_tools.visual as visual_tools

    async def fake_warmup():
        return None

    monkeypatch.setattr(visual_tools, "clip_search", fake_clip_search)
    monkeypatch.setattr(visual_tools, "warmup_visual_retrieval_tools", fake_warmup)

    hits = asyncio.run(
        execute_tool_calls(
            [
                ToolCall(
                    tool_call_id="tc_001",
                    tool_name="clip_search",
                    event_id="event-1",
                    parameters={"query": "red shirt", "top_k": 5},
                )
            ]
        )
    )

    assert len(hits) == 1
    assert hits[0].event_id == "event-1"
    assert hits[0].tool_call_id == "tc_001"
