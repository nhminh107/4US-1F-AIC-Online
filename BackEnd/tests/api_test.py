from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

import BackEnd.app.api.pipeline as pipeline_module
from BackEnd.app.api.models import (
    QueryRequest,
    QueryResponse,
    TopKParameters,
    VerificationSummary,
)
from BackEnd.app.api.pipeline import OnlinePipeline
from BackEnd.app.api.routes import get_frame_image, get_online_pipeline
from BackEnd.app.contracts.models import (
    Event,
    SearchHit,
    StructuredQuery,
    ToolCall,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.main import app


def _frame(
    frame_id: str,
    *,
    timestamp_ms: int,
    frame_idx: int,
    source: str,
) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="video-1",
        shot_id="shot-1" if source == "extracted" else None,
        timestamp_ms=timestamp_ms,
        fps=25.0,
        frame_idx=frame_idx,
        source=source,
        frame_path=Path(f"data/{frame_id}.jpg"),
    )


class FakePostgreManager:
    def __init__(self) -> None:
        self.extracted_1 = _frame(
            "frame-extracted-1",
            timestamp_ms=1_000,
            frame_idx=25,
            source="extracted",
        )
        self.extracted_2 = _frame(
            "frame-extracted-2",
            timestamp_ms=3_000,
            frame_idx=75,
            source="extracted",
        )
        self.official_1 = _frame(
            "frame-official-1",
            timestamp_ms=1_040,
            frame_idx=26,
            source="official",
        )
        self.official_2 = _frame(
            "frame-official-2",
            timestamp_ms=3_040,
            frame_idx=76,
            source="official",
        )
        self.frames = {
            frame.frame_id: frame
            for frame in (
                self.extracted_1,
                self.extracted_2,
                self.official_1,
                self.official_2,
            )
        }

    def get_evidence_by_video_id_and_time(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        **_kwargs,
    ) -> list[list]:
        frames = [
            frame
            for frame in (self.extracted_1, self.extracted_2)
            if frame.video_id == video_id
            and start_ms <= frame.timestamp_ms <= end_ms
        ]
        return [[], [], frames, [], [], [], [], []]

    def get_frame_record_by_frame_id(self, frame_id: str) -> FrameMetadata:
        if frame_id not in self.frames:
            raise ValueError(frame_id)
        return self.frames[frame_id]

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        return [frame for frame in self.frames.values() if frame.video_id == video_id]

    def get_nearest_official_frame(
        self,
        video_id: str,
        timestamp_ms: int,
        *,
        require_image: bool = True,
    ) -> FrameMetadata | None:
        del require_image
        candidates = [
            self.official_1,
            self.official_2,
        ]
        candidates = [frame for frame in candidates if frame.video_id == video_id]
        return min(
            candidates,
            key=lambda frame: (abs(frame.timestamp_ms - timestamp_ms), frame.frame_idx),
        )


def _hit(frame: FrameMetadata, *, event_id: str | None = None) -> SearchHit:
    return SearchHit(
        tool_call_id="tc-1",
        event_id=event_id,
        source="frame_embedding",
        entity_type="frame",
        entity_id=frame.frame_id,
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        start_ms=frame.timestamp_ms,
        end_ms=frame.timestamp_ms,
        rank=1,
        raw_score=0.9,
    )


def test_vqa_returns_kis_style_result_with_nearest_official_frame(monkeypatch):
    db = FakePostgreManager()
    pipeline = OnlinePipeline(db, selective_verifier_enabled=False)
    structured_query = StructuredQuery(
        query_id="query-vqa",
        task="VQA",
        question="What is happening?",
    )

    async def fake_extract_intent(_raw_query):
        return structured_query

    async def fake_fast_path(_query, *, top_k):
        assert top_k["clip_search"] == 5
        return [_hit(db.extracted_1)]

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(pipeline_module, "run_fast_path", fake_fast_path)

    response = asyncio.run(
        pipeline.execute(
            QueryRequest(
                prompt="question",
                top_k=TopKParameters(clip_search=5, result_top_k=3),
            )
        )
    )

    assert response.task == "VQA"
    assert response.execution_path == "fast_path"
    assert len(response.results) == 1
    result = response.results[0]
    assert result.representative_frame_id == "frame-extracted-1"
    assert result.display_frame_id == "frame-official-1"
    assert result.frame_idx == 26
    assert result.img_url == "data/frame-official-1.jpg"
    assert response.top_k.clip_search == 5
    assert response.top_k.result_top_k == 3
    assert response.verification.reason == "selective_verifier_disabled"


def test_kis_runs_shared_pipeline_and_returns_display_metadata(monkeypatch):
    db = FakePostgreManager()
    pipeline = OnlinePipeline(db, selective_verifier_enabled=False)
    structured_query = StructuredQuery(
        query_id="query-kis",
        task="KIS",
        visual_queries=["a person walking"],
    )

    async def fake_extract_intent(_raw_query):
        return structured_query

    async def fake_fast_path(_query, *, top_k):
        return [_hit(db.extracted_1)]

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(pipeline_module, "run_fast_path", fake_fast_path)

    response = asyncio.run(pipeline.execute(QueryRequest(prompt="find a frame")))

    assert response.task == "KIS"
    assert len(response.results) == 1
    assert response.results[0].representative_frame_id == "frame-extracted-1"
    assert response.results[0].display_frame_id == "frame-official-1"
    assert response.results[0].frame_idx == 26


def test_trake_returns_one_official_frame_per_aligned_event(monkeypatch):
    db = FakePostgreManager()
    pipeline = OnlinePipeline(db, selective_verifier_enabled=False)
    structured_query = StructuredQuery(
        query_id="query-trake",
        task="TRAKE",
        events=[
            Event(event_id="E1", description="first event"),
            Event(event_id="E2", description="second event"),
        ],
    )
    tool_calls = [
        ToolCall(
            tool_call_id="tc-1",
            tool_name="clip_search",
            event_id="E1",
            parameters={"query": "first event", "top_k": 10},
        ),
        ToolCall(
            tool_call_id="tc-2",
            tool_name="clip_search",
            event_id="E2",
            parameters={"query": "second event", "top_k": 10},
        ),
    ]

    async def fake_extract_intent(_raw_query):
        return structured_query

    async def fake_query_planner(_query):
        return tool_calls

    async def fake_tool_executor(_tool_calls):
        assert all(call.parameters["top_k"] == 7 for call in _tool_calls)
        return [
            _hit(db.extracted_1, event_id="E1"),
            _hit(db.extracted_2, event_id="E2"),
        ]

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(pipeline_module, "run_query_planner", fake_query_planner)
    monkeypatch.setattr(pipeline_module, "execute_tool_calls", fake_tool_executor)

    response = asyncio.run(
        pipeline.execute(
            QueryRequest(
                prompt="events",
                top_k=TopKParameters(clip_search=7, result_top_k=2),
            )
        )
    )

    assert response.task == "TRAKE"
    assert response.execution_path == "query_planner"
    assert response.trake_status == "success"
    assert len(response.results) == 1
    assert [event.display_frame_id for event in response.results[0].events] == [
        "frame-official-1",
        "frame-official-2",
    ]
    assert [event.frame_idx for event in response.results[0].events] == [26, 76]


def test_selective_verifier_bool_is_reflected_when_enabled():
    pipeline = OnlinePipeline(
        FakePostgreManager(),
        selective_verifier_enabled=True,
    )
    query = StructuredQuery(query_id="query-kis", task="KIS")

    summary, warnings = asyncio.run(
        pipeline._verify_results(query, [], [])
    )

    assert summary.enabled is True
    assert summary.applied is False
    assert summary.reason == "no_displayable_results"
    assert warnings == []


class FakeFrameResolver:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def media_path(self, _frame_id: str) -> Path:
        return self.image_path


class FakeRoutePipeline:
    def __init__(self, image_path: Path) -> None:
        self.frame_resolver = FakeFrameResolver(image_path)

    async def execute(self, _request: QueryRequest) -> QueryResponse:
        query = StructuredQuery(query_id="query-api", task="KIS")
        return QueryResponse(
            query_id=query.query_id,
            task=query.task,
            structured_query=query,
            top_k=TopKParameters(),
            execution_path="fast_path",
            search_hit_count=0,
            candidate_count=0,
            results=[],
            verification=VerificationSummary(
                enabled=False,
                applied=False,
                reason="selective_verifier_disabled",
            ),
        )


def test_fastapi_query_and_media_routes(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image-bytes")
    fake_pipeline = FakeRoutePipeline(image_path)

    async def override_pipeline():
        return fake_pipeline

    app.dependency_overrides[get_online_pipeline] = override_pipeline

    try:
        async def call_api():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                query_response = await client.post(
                    "/api/v1/query",
                    json={"prompt": "find it"},
                )
            media_response = await get_frame_image(
                "frame-official-1",
                fake_pipeline,
            )
            return query_response, media_response

        query_response, media_response = asyncio.run(call_api())
    finally:
        app.dependency_overrides.clear()

    assert query_response.status_code == 200
    assert query_response.json()["query_id"] == "query-api"
    assert Path(media_response.path) == image_path
