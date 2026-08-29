"""RED regression specifications for the retrieval V2 remediation work.

These tests intentionally describe the target behavior before the production
implementation exists. Do not weaken the assertions to match current behavior.
"""

from __future__ import annotations

import asyncio

from BackEnd.app.contracts.models import (
    Event,
    SearchHit,
    StructuredQuery,
    TemporalConstraint,
)
from BackEnd.app.retrieval_v2.contracts import (
    CandidateBudget,
    CandidateReview,
    RetryDiagnosis,
    CoverageCell,
    MomentBand,
    SearchCall,
)
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.moment_bands import build_moment_bands
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan
from BackEnd.app.retrieval_v2.sequence import collapse_kis_sequences


def _hit(
    entity_id: str,
    *,
    atom_id: str | None,
    video_id: str = "V1",
    start_ms: int = 1_000,
    end_ms: int = 1_000,
) -> SearchHit:
    return SearchHit(
        source="frame_embedding",
        entity_type="frame",
        entity_id=entity_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=end_ms,
        rank=1,
        raw_score=0.92,
        atom_id=atom_id,
        prompt_role="global",
        retriever_family="legacy_clip_b32",
    )


def _event_band(
    video_id: str,
    event_id: str,
    start_ms: int,
    end_ms: int,
) -> MomentBand:
    atom_id = event_id.replace("E", "A")
    return MomentBand(
        band_id=f"{video_id}-{event_id}-{start_ms}",
        video_id=video_id,
        event_id=event_id,
        start_ms=start_ms,
        end_ms=end_ms,
        peak_ms=(start_ms + end_ms) // 2,
        coverage={
            atom_id: CoverageCell(
                atom_id=atom_id,
                status="PASS",
                score=0.8,
                evidence_ids=[f"evidence-{video_id}-{event_id}-{start_ms}"],
            )
        },
        score=0.8,
    )


def test_long_event_is_semantically_decomposed_into_atomic_visual_atoms():
    description = (
        "A lion dancer stands upright and spins on top of a pole, then jumps "
        "across two adjacent poles, dives headfirst to bite a pumpkin decorated "
        "with a yellow flower, and finally continues jumping to the next poles."
    )
    query = StructuredQuery(
        query_id="q-atomic-decomposition",
        task="KIS",
        events=[Event(event_id="E1", description=description)],
    )

    plan = build_retrieval_plan(query)
    event_atoms = [
        atom
        for atom in plan.atoms
        if atom.event_id == "E1" and atom.modality == "visual"
    ]
    normalized = [atom.text.lower() for atom in event_atoms]

    assert len(event_atoms) >= 4
    assert all(atom.text != description for atom in event_atoms)
    assert any("upright" in text for text in normalized)
    assert any("spin" in text and "pole" in text for text in normalized)
    assert any("jump" in text and "two" in text and "pole" in text for text in normalized)
    assert any("bite" in text and "pumpkin" in text for text in normalized)
    assert any("yellow flower" in text for text in normalized)


def test_must_not_only_evidence_never_becomes_required_positive_coverage():
    class NegativeOnlyGateway:
        async def search(self, call: SearchCall) -> list[SearchHit]:
            if "written word spacecraft" not in call.query.lower():
                return []
            return [_hit("negative-only-frame", atom_id=None, video_id="V-negative")]

    query = StructuredQuery(
        query_id="q-negative-only",
        task="KIS",
        visual_queries=["a spacecraft physically visible above a city"],
        negative_constraints=["the written word spacecraft"],
    )
    controller = SearchController(
        NegativeOnlyGateway(),
        budget=CandidateBudget(
            raw_retrieval_target=20,
            raw_retrieval_max=20,
            moment_band_limit=10,
            video_shortlist_limit=5,
            local_retrieval_k=5,
            retry_retrieval_k=5,
            rerank_limit=10,
            max_retry_rounds=0,
        ),
    )

    result = asyncio.run(controller.search(query))
    negative_atom_ids = {
        atom.atom_id for atom in result.plan.atoms if atom.operator == "MUST_NOT"
    }

    assert negative_atom_ids
    assert all(
        atom_id not in band.coverage
        for band in result.bands
        for atom_id in negative_atom_ids
    )
    assert result.reranked_bands == []


def test_retrieval_support_does_not_automatically_verify_moment_band_atom():
    bands = build_moment_bands(
        [_hit("retrieved-frame", atom_id="A1")],
        required_atom_ids=["A1"],
    )

    assert len(bands) == 1
    cell = bands[0].coverage["A1"]
    assert cell.evidence_ids == ["retrieved-frame"]
    assert cell.score > 0.0
    assert cell.status == "UNKNOWN"


def test_controller_honors_two_retries_and_reviews_each_changed_shortlist():
    class ChangingGateway:
        def __init__(self) -> None:
            self.retry_number = 0

        async def search(self, call: SearchCall) -> list[SearchHit]:
            if call.call_id.startswith("retry_"):
                self.retry_number += 1
                number = self.retry_number
                return [
                    _hit(
                        f"retry-frame-{number}",
                        atom_id=None,
                        video_id=f"V-retry-{number}",
                        start_ms=number * 20_000,
                        end_ms=number * 20_000,
                    )
                ]
            return [
                _hit(
                    f"{call.call_id}-initial-frame",
                    atom_id=None,
                    video_id="V-initial",
                )
            ]

    class RejectingReviewer:
        def __init__(self) -> None:
            self.reviewed_band_ids: list[tuple[str, ...]] = []

        async def review(self, plan, bands):
            self.reviewed_band_ids.append(tuple(band.band_id for band in bands))
            band = bands[0]
            required = next(atom for atom in plan.atoms if atom.required)
            return [
                CandidateReview(
                    band_id=band.band_id,
                    verdict="mismatch",
                    confidence=0.99,
                    atom_status={required.atom_id: "FAIL"},
                    scope="VIDEO",
                    failure_reason="wrong_video",
                    video_id=band.video_id,
                )
            ]

    gateway = ChangingGateway()
    reviewer = RejectingReviewer()
    controller = SearchController(
        gateway,
        reviewer=reviewer,
        budget=CandidateBudget(
            raw_retrieval_target=20,
            raw_retrieval_max=20,
            moment_band_limit=10,
            video_shortlist_limit=3,
            local_retrieval_k=5,
            retry_retrieval_k=5,
            rerank_limit=5,
            review_limit=1,
            max_retry_rounds=2,
        ),
    )
    query = StructuredQuery(
        query_id="q-two-retries",
        task="KIS",
        visual_queries=["a lion dancer biting a pumpkin on top of poles"],
    )

    result = asyncio.run(controller.search(query))
    retry_rounds = [round_ for round_ in result.session.rounds if round_.phase == "RETRY"]

    assert len(retry_rounds) == 2
    assert [round_.round_index for round_ in retry_rounds] == [2, 3]
    assert len(reviewer.reviewed_band_ids) == 3
    assert len(set(reviewer.reviewed_band_ids)) == 3
    assert gateway.retry_number == 2


def test_low_moment_confidence_retry_scans_the_known_video_without_time_scope():
    controller = SearchController(
        object(),
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=10,
            unique_candidate_min=1,
            unique_candidate_max=5,
            local_retrieval_k=4,
            retry_retrieval_k=4,
        ),
    )
    plan = build_retrieval_plan(
        StructuredQuery(
            query_id="q-full-video-retry",
            task="KIS",
            visual_queries=["a red car"],
        )
    )

    calls = controller._retry_calls(
        plan,
        [
            RetryDiagnosis(
                reason="LOW_MOMENT_CONFIDENCE",
                action="EXPAND_LOCAL_SEARCH",
                video_id="V-known",
            )
        ],
        retry_index=1,
    )

    assert calls
    assert all(call.video_ids == ["V-known"] for call in calls)
    assert all(call.start_ms is None and call.end_ms is None for call in calls)


def test_kis_sequence_rejects_event_pair_beyond_max_gap_ms():
    bands = [
        _event_band("V-valid", "E1", 1_000, 2_000),
        _event_band("V-valid", "E2", 6_000, 7_000),
        _event_band("V-too-far", "E1", 1_000, 2_000),
        _event_band("V-too-far", "E2", 20_000, 21_000),
    ]
    constraints = [
        TemporalConstraint(
            before="E1",
            after="E2",
            max_gap_ms=5_000,
        )
    ]

    sequences = collapse_kis_sequences(
        bands,
        ["E1", "E2"],
        temporal_constraints=constraints,
        limit=10,
    )

    assert [sequence.video_id for sequence in sequences] == ["V-valid"]
