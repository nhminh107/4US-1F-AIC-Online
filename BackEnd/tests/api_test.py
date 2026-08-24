from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

import BackEnd.app.api.pipeline as pipeline_module
from BackEnd.app.api.models import (
    QueryRequest,
    QueryResponse,
    TopKParameters,
    VerificationItem,
    VerificationSummary,
)
from BackEnd.app.api.pipeline import OnlinePipeline
from BackEnd.app.api.routes import get_frame_image, get_online_pipeline
from BackEnd.app.contracts.models import (
    Event,
    SearchHit,
    StructuredQuery,
    TemporalEventResult,
    TemporalSequence,
    ToolCall,
    VerifiedResult,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.retrieval_v2.contracts import (
    CoverageCell,
    MomentBand,
    RetrievalPlan,
    SearchControllerResult,
    SearchRound,
    SearchSessionState,
    VideoHypothesis,
)
from BackEnd.main import app
from BackEnd.app.trake import TrakeAlignerResult


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
    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=False,
    )
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
    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=False,
    )
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


def test_opt_in_v2_pipeline_uses_controller_and_exposes_session(monkeypatch):
    db = FakePostgreManager()
    structured_query = StructuredQuery(
        query_id="query-v2",
        task="KIS",
        visual_queries=["a person walking"],
    )
    hit = _hit(db.extracted_1).model_copy(
        update={"atom_id": "A1", "retriever_family": "legacy_clip_b32"}
    )
    coverage = {"A1": CoverageCell(atom_id="A1", status="PASS", score=0.9, evidence_ids=[hit.entity_id])}
    band = MomentBand(
        band_id="mb-1",
        video_id="video-1",
        start_ms=1_000,
        end_ms=1_000,
        peak_ms=1_000,
        coverage=coverage,
        hits=[hit],
        score=0.9,
    )
    wrong_hits = [
        _hit(db.extracted_2).model_copy(update={
            "entity_type": "ocr",
            "source": "ocr_index",
            "entity_id": f"ocr-wrong-{index}",
            "atom_id": "A1",
            "retriever_family": "ocr_search",
            "text_content": "Wrong Place",
            "raw_score": 0.99,
        })
        for index in range(5)
    ]
    wrong_band = MomentBand(
        band_id="mb-vqa-wrong",
        video_id="video-1",
        start_ms=3_000,
        end_ms=3_000,
        peak_ms=3_000,
        coverage=coverage,
        hits=wrong_hits,
        score=0.5,
    )
    hypothesis = VideoHypothesis(
        video_id="video-1",
        video_confidence=0.9,
        moment_confidence=0.8,
        coverage=coverage,
        band_ids=[band.band_id, wrong_band.band_id],
    )
    session = SearchSessionState(
        query_id="query-v2",
        rounds=[SearchRound(round_index=0, phase="GLOBAL", hit_count=1)],
        raw_hit_count=1,
        deduplicated_hit_count=1,
        hypotheses=[hypothesis],
    )
    controller_result = SearchControllerResult(
        plan=RetrievalPlan(
            query_id="query-v2",
            task="KIS",
            execution_profile="KIS_MOMENT",
        ),
        bands=[band, wrong_band],
        reranked_bands=[band, wrong_band],
        hypotheses=[hypothesis],
        session=session,
    )

    class FakeController:
        async def search(self, query):
            assert query == structured_query
            return controller_result

    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=True,
        search_controller=FakeController(),
    )

    async def fake_extract_intent(_raw_query):
        return structured_query

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)

    response = asyncio.run(pipeline.execute(QueryRequest(prompt="find a frame")))

    assert response.execution_path == "retrieval_v2"
    assert response.retrieval_v2_session == session
    assert response.results[0].display_frame_id == "frame-official-1"


def test_v2_vqa_returns_only_grounded_answer_from_candidate_evidence(monkeypatch):
    db = FakePostgreManager()
    query = StructuredQuery(
        query_id="query-v2-vqa",
        task="VQA",
        question="Which place is shown?",
        ocr_constraints=["place name"],
    )
    hit = _hit(db.extracted_1).model_copy(update={
        "entity_type": "ocr",
        "source": "ocr_index",
        "entity_id": "ocr-grounded",
        "atom_id": "A1",
        "retriever_family": "ocr_search",
        "text_content": "Ha Noi",
    })
    coverage = {
        "A1": CoverageCell(
            atom_id="A1",
            retrieval_status="RETRIEVED",
            status="UNKNOWN",
            score=0.9,
            evidence_ids=[hit.entity_id],
        )
    }
    band = MomentBand(
        band_id="mb-vqa",
        video_id="video-1",
        start_ms=1_000,
        end_ms=1_000,
        peak_ms=1_000,
        coverage=coverage,
        hits=[hit],
        score=0.9,
    )
    hypothesis = VideoHypothesis(
        video_id="video-1",
        video_confidence=0.9,
        moment_confidence=0.8,
        coverage=coverage,
        band_ids=[band.band_id],
    )
    result = SearchControllerResult(
        plan=RetrievalPlan(
            query_id=query.query_id,
            task="VQA",
            execution_profile="VQA",
        ),
        bands=[band],
        reranked_bands=[band],
        hypotheses=[hypothesis],
        session=SearchSessionState(
            query_id=query.query_id,
            rounds=[SearchRound(round_index=0, phase="GLOBAL", hit_count=1)],
            raw_hit_count=6,
            deduplicated_hit_count=6,
            hypotheses=[hypothesis],
        ),
    )

    class Controller:
        async def search(self, _query):
            return result

    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=True,
        search_controller=Controller(),
    )

    async def fake_extract_intent(_raw_query):
        return query

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    response = asyncio.run(pipeline.execute(QueryRequest(prompt="question")))

    assert response.answer == "Ha Noi"
    assert response.answer_status == "answered"
    assert response.results[0].answer == "Ha Noi"


def test_v2_translation_boundary_falls_back_without_http_500(monkeypatch):
    db = FakePostgreManager()
    query = StructuredQuery(
        query_id="query-vietnamese-fallback",
        task="KIS",
        visual_queries=["ba người đi dưới mưa"],
    )

    class TranslationFailingController:
        async def search(self, _query):
            raise ValueError(
                "Visual query atoms must be translated to English before CLIP planning"
            )

    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=True,
        search_controller=TranslationFailingController(),
    )

    async def fake_extract_intent(_raw_query):
        return query

    async def fake_fast_path(_query, *, top_k):
        del top_k
        return [_hit(db.extracted_1)]

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(pipeline_module, "run_fast_path", fake_fast_path)

    response = asyncio.run(pipeline.execute(QueryRequest(prompt="tìm cảnh")))

    assert response.execution_path == "query_planner_fallback"
    assert response.results
    assert any("English visual plan" in warning for warning in response.warnings)


def test_trake_returns_one_official_frame_per_aligned_event(monkeypatch):
    db = FakePostgreManager()
    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=False,
    )
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


def test_trake_nearest_official_frame_outside_tolerance_is_rejected():
    pipeline = OnlinePipeline(
        FakePostgreManager(),
        selective_verifier_enabled=False,
        retrieval_v2_enabled=False,
    )
    sequence = TemporalSequence(
        video_id="video-1",
        sequence_score=0.9,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="far-event",
                start_ms=100_000,
                end_ms=101_000,
            )
        ],
    )

    assert pipeline._resolve_monotonic_trake_frames(sequence) is None


def test_trake_status_is_synchronized_after_task_gate_rejects_every_sequence():
    original = TrakeAlignerResult(
        status="success",
        sequences=[],
        replan_required=False,
    )

    synchronized = OnlinePipeline._synchronize_trake_result(
        original,
        [],
        ["E1", "E2"],
    )

    assert synchronized.status == "no_valid_sequence"
    assert synchronized.replan_required is True
    assert synchronized.missing_event_ids == ["E1", "E2"]


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


def test_rejected_verification_item_is_removed_from_api_output(monkeypatch):
    db = FakePostgreManager()
    pipeline = OnlinePipeline(
        db,
        selective_verifier_enabled=False,
        retrieval_v2_enabled=False,
    )
    query = StructuredQuery(
        query_id="query-verification-gate",
        task="KIS",
        visual_queries=["a person walking"],
    )

    async def fake_extract_intent(_raw_query):
        return query

    async def fake_fast_path(_query, *, top_k):
        del top_k
        return [_hit(db.extracted_1)]

    async def fake_verify(_query, results, _ranked):
        assert results
        return (
            VerificationSummary(
                enabled=True,
                applied=True,
                items=[
                    VerificationItem(
                        target_id=results[0].representative_frame_id,
                        result=VerifiedResult(
                            status="rejected",
                            confidence=0.99,
                            failed_constraints=["wrong_action"],
                        ),
                    )
                ],
            ),
            [],
        )

    monkeypatch.setattr(pipeline_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(pipeline_module, "run_fast_path", fake_fast_path)
    monkeypatch.setattr(pipeline, "_verify_results", fake_verify)

    response = asyncio.run(pipeline.execute(QueryRequest(prompt="find it")))

    assert response.results == []
    assert any("rejected 1" in warning for warning in response.warnings)


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
