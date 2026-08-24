"""RED specifications for batched, gain-aware retrieval budgeting.

The tests intentionally define the next public behavior before production code
implements it. They must remain unskipped and failing until that slice lands.
"""

from __future__ import annotations

import asyncio

import pytest

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.contracts import CandidateBudget, SearchCall
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan


def _hit(
    entity_id: str,
    *,
    video_id: str,
    start_ms: int,
) -> SearchHit:
    return SearchHit(
        source="frame_embedding",
        entity_type="frame",
        entity_id=entity_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=start_ms,
        rank=1,
        raw_score=0.9,
    )


def _budget(**overrides) -> CandidateBudget:
    values = {
        "raw_retrieval_target": 20,
        "raw_retrieval_max": 40,
        "moment_band_limit": 20,
        "video_shortlist_limit": 3,
        "local_retrieval_k": 12,
        "retry_retrieval_k": 5,
        "rerank_limit": 10,
        "max_retry_rounds": 0,
    }
    values.update(overrides)
    return CandidateBudget(**values)


def test_controller_prefers_gateway_search_many_over_individual_search_calls():
    class BatchGateway:
        def __init__(self) -> None:
            self.batch_calls: list[tuple[SearchCall, ...]] = []
            self.individual_calls: list[SearchCall] = []

        async def search_many(
            self,
            calls: list[SearchCall],
        ) -> list[list[SearchHit]]:
            """Return one result list per call, preserving input order."""
            self.batch_calls.append(tuple(calls))
            return [[] for _ in calls]

        async def search(self, call: SearchCall) -> list[SearchHit]:
            self.individual_calls.append(call)
            raise AssertionError("search() must not be used when search_many() is available")

    gateway = BatchGateway()
    controller = SearchController(gateway, budget=_budget())
    query = StructuredQuery(
        query_id="q-batch-preference",
        task="KIS",
        visual_queries=["a lion dancer jumping between tall poles"],
    )

    asyncio.run(controller.search(query))

    assert len(gateway.batch_calls) == 1
    assert gateway.batch_calls[0]
    assert gateway.individual_calls == []


def test_candidate_budget_exposes_validated_unique_canonical_candidate_range():
    budget = CandidateBudget(
        raw_retrieval_target=2_400,
        raw_retrieval_max=3_000,
        unique_candidate_min=1_600,
        unique_candidate_max=2_000,
    )

    assert budget.unique_candidate_min == 1_600
    assert budget.unique_candidate_max == 2_000

    with pytest.raises(ValueError, match="unique_candidate_min"):
        CandidateBudget(
            raw_retrieval_target=2_400,
            raw_retrieval_max=3_000,
            unique_candidate_min=2_001,
            unique_candidate_max=2_000,
        )


def test_controller_deepens_on_unique_moment_and_video_gain_not_requested_top_k():
    class UniqueRangeBudget(CandidateBudget):
        # Test-only bridge for the intended CandidateBudget fields.
        unique_candidate_min: int = 3
        unique_candidate_max: int = 6

    class GainGateway:
        def __init__(self) -> None:
            self.batches: list[tuple[SearchCall, ...]] = []
            self.individual_calls: list[SearchCall] = []
            self.global_batch_number = 0

        async def search_many(
            self,
            calls: list[SearchCall],
        ) -> list[list[SearchHit]]:
            self.batches.append(tuple(calls))
            is_global = bool(calls) and calls[0].call_id.startswith("global_")
            if not is_global:
                return [[] for _ in calls]

            self.global_batch_number += 1
            if self.global_batch_number == 1:
                # The requested top_k budget is exhausted, but every call maps
                # to the same canonical moment in the same video.
                return [
                    [_hit("same-canonical-frame", video_id="V1", start_ms=1_000)]
                    for _ in calls
                ]

            # A deeper wave adds genuinely new videos and moments.
            return [
                [
                    _hit(
                        f"new-canonical-frame-{index}",
                        video_id=f"V{index + 2}",
                        start_ms=(index + 2) * 10_000,
                    )
                ]
                for index, _call in enumerate(calls)
            ]

        async def search(self, call: SearchCall) -> list[SearchHit]:
            self.individual_calls.append(call)
            return [_hit("same-canonical-frame", video_id="V1", start_ms=1_000)]

    budget = UniqueRangeBudget(
        raw_retrieval_target=20,
        raw_retrieval_max=40,
        moment_band_limit=20,
        video_shortlist_limit=6,
        local_retrieval_k=12,
        retry_retrieval_k=5,
        rerank_limit=10,
        max_retry_rounds=0,
        unique_candidate_min=3,
        unique_candidate_max=6,
    )
    gateway = GainGateway()
    controller = SearchController(gateway, budget=budget)
    query = StructuredQuery(
        query_id="q-adaptive-unique-gain",
        task="KIS",
        visual_queries=["a cook adding broth and ingredients to a bowl of noodles"],
    )

    result = asyncio.run(controller.search(query))
    global_batches = [
        calls
        for calls in gateway.batches
        if calls and calls[0].call_id.startswith("global_")
    ]

    assert len(global_batches) >= 2
    assert sum(call.top_k for call in global_batches[0]) >= budget.raw_retrieval_target
    assert sum(call.top_k for call in global_batches[1]) > sum(
        call.top_k for call in global_batches[0]
    )
    canonical_moments = {
        (band.video_id, band.start_ms, band.end_ms) for band in result.bands
    }
    assert budget.unique_candidate_min <= len(canonical_moments) <= budget.unique_candidate_max
    assert gateway.individual_calls == []


def test_local_retrieval_allocates_a_separate_quota_to_each_shortlisted_video():
    class ThreeVideoGateway:
        def __init__(self) -> None:
            self.calls: list[SearchCall] = []

        async def search(self, call: SearchCall) -> list[SearchHit]:
            self.calls.append(call)
            if call.call_id.startswith("global_"):
                return [
                    _hit(f"{call.call_id}-{video_id}", video_id=video_id, start_ms=1_000)
                    for video_id in ("V1", "V2", "V3")
                ]
            return []

    gateway = ThreeVideoGateway()
    budget = _budget(video_shortlist_limit=3, local_retrieval_k=12)
    controller = SearchController(gateway, budget=budget)
    query = StructuredQuery(
        query_id="q-local-video-quotas",
        task="KIS",
        visual_queries=["three people walking down a rainy slope with umbrellas"],
    )

    asyncio.run(controller.search(query))
    local_calls = [call for call in gateway.calls if call.call_id.startswith("local_")]

    assert local_calls
    assert all(len(call.video_ids) == 1 for call in local_calls)
    quota_by_video = {
        call.video_ids[0]: sum(
            candidate.top_k
            for candidate in local_calls
            if candidate.video_ids == call.video_ids
        )
        for call in local_calls
    }
    assert set(quota_by_video) == {"V1", "V2", "V3"}
    assert sum(quota_by_video.values()) == budget.local_retrieval_k
    assert max(quota_by_video.values()) - min(quota_by_video.values()) <= 1


def test_canonical_cap_preserves_rare_atom_evidence_from_generic_flood():
    controller = SearchController(
        object(),
        budget=_budget(unique_candidate_min=2, unique_candidate_max=4),
    )
    hits = [
        _hit(f"generic-{index}", video_id=f"V{index}", start_ms=index * 10_000).model_copy(
            update={"atom_id": "A-common", "rank": index + 1, "raw_score": 0.99}
        )
        for index in range(10)
    ]
    hits.append(
        _hit("rare", video_id="V-rare", start_ms=999_000).model_copy(
            update={"atom_id": "A-rare", "rank": 100, "raw_score": 0.1}
        )
    )

    limited = controller._limit_canonical_candidates(hits)

    assert any(hit.atom_id == "A-rare" for hit in limited)
    assert len(controller._canonical_candidate_keys(limited)) == 4


def test_online_selectivity_reweights_atom_after_global_wave():
    plan = build_retrieval_plan(
        StructuredQuery(
            query_id="q-selectivity",
            task="KIS",
            visual_queries=["a red ceremonial mask"],
        )
    )
    atom_id = plan.atoms[0].atom_id
    broad_hits = [
        _hit(f"broad-{index}", video_id=f"V{index}", start_ms=index * 2_000).model_copy(
            update={"atom_id": atom_id, "prompt_role": "global"}
        )
        for index in range(10)
    ]
    focused_hits = [
        _hit(f"focused-{index}", video_id="V1", start_ms=index * 2_000).model_copy(
            update={"atom_id": atom_id, "prompt_role": "global"}
        )
        for index in range(10)
    ]

    broad = SearchController._calibrate_plan_selectivity(plan, broad_hits)
    focused = SearchController._calibrate_plan_selectivity(plan, focused_hits)

    assert focused.atoms[0].discriminative_weight > broad.atoms[0].discriminative_weight
