from __future__ import annotations

import asyncio

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import SearchCall
from BackEnd.app.retrieval_v2.gateway import ToolSearchGateway


def _hit(entity_id: str, entity_type: str = "frame") -> SearchHit:
    return SearchHit(
        source="test",
        entity_type=entity_type,
        entity_id=entity_id,
        video_id="V1",
        start_ms=1_000,
        end_ms=1_000,
        rank=1,
        raw_score=0.9,
    )


def test_batch_failure_is_isolated_by_call_and_family(monkeypatch):
    async def broken_visual_batch(_requests):
        raise RuntimeError("batch failed")

    async def working_frame(**_kwargs):
        return [_hit("frame-ok")]

    async def broken_asr(**_kwargs):
        raise RuntimeError("asr failed")

    async def working_object(**_kwargs):
        return [_hit("object-ok", "object_detection")]

    monkeypatch.setattr(
        "BackEnd.app.retrieval_v2.gateway.visual_search_many",
        broken_visual_batch,
    )
    monkeypatch.setattr("BackEnd.app.retrieval_v2.gateway.frame_search", working_frame)
    monkeypatch.setattr("BackEnd.app.retrieval_v2.gateway.asr_search", broken_asr)
    monkeypatch.setattr("BackEnd.app.retrieval_v2.gateway.object_search", working_object)

    calls = [
        SearchCall(
            call_id="visual",
            atom_id="A1",
            retriever="frame_search",
            query="red car",
            top_k=3,
        ),
        SearchCall(
            call_id="asr",
            atom_id="A2",
            retriever="asr_search",
            query="xin chao",
            top_k=3,
        ),
        SearchCall(
            call_id="object",
            atom_id="A3",
            retriever="object_search",
            query="person",
            top_k=3,
        ),
    ]

    results = asyncio.run(ToolSearchGateway().search_many(calls))

    assert [[hit.entity_id for hit in group] for group in results] == [
        ["frame-ok"],
        [],
        ["object-ok"],
    ]


def test_gateway_applies_local_time_scope_to_visual_hits(monkeypatch):
    async def visual_batch(_requests):
        return [[
            _hit("inside"),
            _hit("outside").model_copy(update={"start_ms": 20_000, "end_ms": 20_000}),
        ]]

    monkeypatch.setattr(
        "BackEnd.app.retrieval_v2.gateway.visual_search_many",
        visual_batch,
    )
    call = SearchCall(
        call_id="local",
        atom_id="A1",
        retriever="frame_search",
        query="red car",
        top_k=3,
        video_ids=["V1"],
        start_ms=500,
        end_ms=2_000,
    )

    results = asyncio.run(ToolSearchGateway().search_many([call]))

    assert [hit.entity_id for hit in results[0]] == ["inside"]


def test_gateway_chunks_large_visual_batches_without_losing_order(monkeypatch):
    batch_sizes: list[int] = []

    async def visual_batch(requests):
        batch_sizes.append(len(requests))
        return [[_hit(request.tool_call_id)] for request in requests]

    monkeypatch.setattr(
        "BackEnd.app.retrieval_v2.gateway.visual_search_many",
        visual_batch,
    )
    calls = [
        SearchCall(
            call_id=f"visual-{index:02d}",
            atom_id="A1",
            retriever="frame_search",
            query="rainy hillside",
            top_k=3,
        )
        for index in range(25)
    ]

    results = asyncio.run(ToolSearchGateway().search_many(calls))

    assert batch_sizes == [12, 12, 1]
    assert [group[0].entity_id for group in results] == [call.call_id for call in calls]
