import asyncio

from BackEnd.app.contracts.models import Event, SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.contracts import (
    CandidateBudget,
    CoverageCell,
    MomentBand,
    SearchCall,
    VideoHypothesis,
)
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[SearchCall] = []

    async def search(self, call: SearchCall) -> list[SearchHit]:
        self.calls.append(call)
        result: list[SearchHit] = []
        for index in range(call.top_k):
            video_id = call.video_ids[index % len(call.video_ids)] if call.video_ids else f"V{index % 60:03d}"
            timestamp = (index % 25) * 20_000
            result.append(
                SearchHit(
                    source=f"{call.retriever}_source",
                    entity_type={
                        "frame_search": "frame",
                        "clip_search": "clip",
                        "shot_search": "shot",
                    }.get(call.retriever, "frame"),
                    entity_id=f"{call.call_id}-{index}",
                    video_id=video_id,
                    start_ms=timestamp,
                    end_ms=timestamp + (2_000 if call.retriever != "frame_search" else 0),
                    rank=index + 1,
                    raw_score=max(0.0, 1.0 - index / max(call.top_k, 1)),
                )
            )
        return result


def test_controller_runs_large_global_pool_then_filtered_local_search():
    gateway = FakeGateway()
    controller = SearchController(
        gateway,
        budget=CandidateBudget(
            raw_retrieval_target=1_800,
            raw_retrieval_max=2_600,
            moment_band_limit=300,
            video_shortlist_limit=12,
            local_retrieval_k=240,
            rerank_limit=100,
        ),
    )
    query = StructuredQuery(
        query_id="q-controller",
        task="KIS",
        visual_queries=["lion dancers jumping on poles and biting a pumpkin"],
    )

    result = asyncio.run(controller.search(query))

    global_calls = [call for call in gateway.calls if not call.video_ids]
    local_calls = [call for call in gateway.calls if call.video_ids]
    deepest_calls = [
        call for call in global_calls if call.call_id.startswith("global_deepen_")
    ] or [call for call in global_calls if call.call_id.startswith("global_")]
    assert 1_800 <= sum(call.top_k for call in deepest_calls) <= 2_600
    assert local_calls
    assert all(len(call.video_ids) <= 12 for call in local_calls)
    assert result.session.raw_hit_count >= 1_800
    assert len(result.bands) <= 300
    assert len(result.reranked_bands) <= 100
    assert [round_.phase for round_ in result.session.rounds[:2]] == ["GLOBAL", "LOCAL"]


def test_controller_reports_video_and_moment_confidence_separately():
    gateway = FakeGateway()
    controller = SearchController(
        gateway,
        budget=CandidateBudget(
            raw_retrieval_target=100,
            raw_retrieval_max=150,
            moment_band_limit=50,
            video_shortlist_limit=5,
            local_retrieval_k=20,
            rerank_limit=20,
        ),
    )
    query = StructuredQuery(
        query_id="q-confidence",
        task="KIS",
        visual_queries=["a woman holding a teacup beside woven handicrafts"],
    )

    result = asyncio.run(controller.search(query))

    assert result.hypotheses
    top = result.hypotheses[0]
    assert 0.0 <= top.video_confidence <= 1.0
    assert 0.0 <= top.moment_confidence <= 1.0
    assert top.video_confidence != top.moment_confidence
    assert top.coverage


def test_controller_diagnoses_missing_atom_for_targeted_retry():
    class MissingAtomGateway(FakeGateway):
        async def search(self, call: SearchCall) -> list[SearchHit]:
            if call.atom_id == "A2":
                self.calls.append(call)
                return []
            return await super().search(call)

    gateway = MissingAtomGateway()
    controller = SearchController(
        gateway,
        budget=CandidateBudget(
            raw_retrieval_target=100,
            raw_retrieval_max=200,
            moment_band_limit=50,
            video_shortlist_limit=5,
            local_retrieval_k=20,
            rerank_limit=20,
        ),
    )
    query = StructuredQuery(
        query_id="q-retry",
        task="KIS",
        visual_queries=["a pride of lions on wooden platforms", "two keepers weighing an animal"],
    )

    result = asyncio.run(controller.search(query))

    assert any(diagnosis.reason == "MISSING_REQUIRED_ATOM" for diagnosis in result.session.diagnoses)
    retry = next(diagnosis for diagnosis in result.session.diagnoses if diagnosis.reason == "MISSING_REQUIRED_ATOM")
    assert retry.atom_id == "A2"
    assert retry.action == "RETRY_WEAK_ATOM"


def test_local_search_scope_expands_when_moment_confidence_is_weak():
    controller = SearchController(
        FakeGateway(),
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=10,
            unique_candidate_min=1,
            unique_candidate_max=10,
            moment_band_limit=10,
            video_shortlist_limit=3,
            local_retrieval_k=30,
            retry_retrieval_k=5,
            rerank_limit=5,
            max_retry_rounds=0,
        ),
    )
    plan = build_retrieval_plan(StructuredQuery(
        query_id="adaptive-local",
        task="KIS",
        visual_queries=["people walking in rain beside a pond"],
    ))
    bands = [
        MomentBand(
            band_id=f"B{index}",
            video_id=video_id,
            start_ms=100_000,
            end_ms=104_000,
            peak_ms=102_000,
        )
        for index, video_id in enumerate(("V-low", "V-mid", "V-high"), start=1)
    ]
    hypotheses = [
        VideoHypothesis(
            video_id="V-low",
            video_confidence=0.7,
            moment_confidence=0.3,
            band_ids=["B1"],
            lane_sources=["moment"],
        ),
        VideoHypothesis(
            video_id="V-mid",
            video_confidence=0.7,
            moment_confidence=0.65,
            band_ids=["B2"],
            lane_sources=["moment"],
        ),
        VideoHypothesis(
            video_id="V-high",
            video_confidence=0.7,
            moment_confidence=0.9,
            band_ids=["B3"],
            lane_sources=["moment"],
        ),
    ]

    calls = controller._local_calls(plan, hypotheses, bands)
    by_video = {video_id: [call for call in calls if call.video_ids == [video_id]] for video_id in ("V-low", "V-mid", "V-high")}

    assert all(call.start_ms is None and call.end_ms is None for call in by_video["V-low"])
    assert all(call.start_ms == 40_000 and call.end_ms == 164_000 for call in by_video["V-mid"])
    assert all(call.start_ms == 85_000 and call.end_ms == 119_000 for call in by_video["V-high"])


def test_retry_search_allocates_independent_calls_to_several_weak_videos():
    controller = SearchController(
        FakeGateway(),
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=10,
            unique_candidate_min=1,
            unique_candidate_max=10,
            moment_band_limit=10,
            video_shortlist_limit=5,
            local_retrieval_k=10,
            retry_retrieval_k=12,
            rerank_limit=5,
        ),
    )
    plan = build_retrieval_plan(StructuredQuery(
        query_id="multi-retry",
        task="KIS",
        visual_queries=["people walking in rain", "a house beside a pond"],
    ))
    from BackEnd.app.retrieval_v2.contracts import RetryDiagnosis
    diagnoses = [
        RetryDiagnosis(
            reason="LOW_MOMENT_CONFIDENCE",
            action="EXPAND_LOCAL_SEARCH",
            atom_id=plan.atoms[0].atom_id,
            video_id="V1",
        ),
        RetryDiagnosis(
            reason="MISSING_ACTION",
            action="RETRY_ACTION_PROMPT",
            atom_id=plan.atoms[-1].atom_id,
            video_id="V2",
        ),
    ]

    calls = controller._retry_calls(plan, diagnoses, retry_index=1)

    assert {tuple(call.video_ids) for call in calls} == {("V1",), ("V2",)}
    assert sum(call.top_k for call in calls) == 12


def test_kis_sequence_ranking_keeps_partial_evidence_for_rescue():
    controller = SearchController(FakeGateway(), budget=CandidateBudget(
        raw_retrieval_target=10,
        raw_retrieval_max=10,
        unique_candidate_min=1,
        unique_candidate_max=10,
        moment_band_limit=10,
        video_shortlist_limit=5,
        local_retrieval_k=10,
        retry_retrieval_k=5,
        rerank_limit=5,
    ))
    plan = build_retrieval_plan(StructuredQuery(
        query_id="partial-sequence",
        task="KIS",
        events=[
            Event(event_id="E1", description="people walk down a rainy slope"),
            Event(event_id="E2", description="people approach a house beside a pond"),
        ],
    ))
    atom = next(atom for atom in plan.atoms if atom.event_id == "E1")
    band = MomentBand(
        band_id="partial-E1",
        video_id="V-partial",
        event_id="E1",
        start_ms=1_000,
        end_ms=2_000,
        peak_ms=1_500,
        coverage={atom.atom_id: CoverageCell(
            atom_id=atom.atom_id,
            retrieval_status="RETRIEVED",
            score=0.8,
        )},
        score=0.8,
    )
    hypotheses = [VideoHypothesis(
        video_id="V-partial",
        video_confidence=0.6,
        moment_confidence=0.3,
        band_ids=[band.band_id],
        lane_sources=["moment"],
    )]

    ranked = controller._rank_for_task(plan, [band], hypotheses)

    assert ranked
    assert ranked[0].video_id == "V-partial"
