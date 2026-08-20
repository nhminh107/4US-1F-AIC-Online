from __future__ import annotations

import asyncio

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.fast_path import runner


def _hit(source: str, entity_type: str, entity_id: str) -> SearchHit:
    return SearchHit(
        source=source,
        entity_type=entity_type,
        entity_id=entity_id,
        video_id="video-1",
        start_ms=0,
        end_ms=100,
        rank=1,
        raw_score=0.9,
    )


def test_run_fast_path_uses_visual_ocr_and_asr_constraints(monkeypatch):
    async def fake_clip_search(query: str, top_k: int):
        assert query == "man red shirt"
        assert top_k == 200
        return [_hit("clip_embedding", "clip", "clip-1")]

    async def fake_ocr_search(query: str, top_k: int):
        assert query == "HCMC"
        assert top_k == 100
        return [_hit("ocr_index", "ocr", "ocr-1")]

    async def fake_asr_search(query: str, top_k: int):
        assert query == "hello"
        assert top_k == 100
        return [_hit("asr_index", "asr", "asr-1")]

    monkeypatch.setattr(runner, "clip_search", fake_clip_search)
    monkeypatch.setattr(runner, "ocr_search", fake_ocr_search)
    monkeypatch.setattr(runner, "asr_search", fake_asr_search)

    query = StructuredQuery(
        query_id="q1",
        task="KIS",
        visual_queries=["man red shirt"],
        ocr_constraints=["HCMC"],
        asr_constraints=["hello"],
    )

    hits = asyncio.run(runner.run_fast_path(query))

    assert [hit.entity_type for hit in hits] == ["clip", "ocr", "asr"]


def test_run_fast_path_skips_failed_tool(monkeypatch):
    async def broken_clip_search(query: str, top_k: int):
        raise RuntimeError("boom")

    async def fake_ocr_search(query: str, top_k: int):
        return [_hit("ocr_index", "ocr", "ocr-1")]

    monkeypatch.setattr(runner, "clip_search", broken_clip_search)
    monkeypatch.setattr(runner, "ocr_search", fake_ocr_search)

    query = StructuredQuery(
        query_id="q1",
        task="KIS",
        visual_queries=["scene"],
        ocr_constraints=["HCMC"],
    )

    hits = asyncio.run(runner.run_fast_path(query))

    assert [hit.entity_id for hit in hits] == ["ocr-1"]


def test_run_fast_path_applies_request_top_k(monkeypatch):
    async def fake_clip_search(query: str, top_k: int):
        assert query == "city bus"
        assert top_k == 7
        return [_hit("clip_embedding", "clip", "clip-1")]

    monkeypatch.setattr(runner, "clip_search", fake_clip_search)
    query = StructuredQuery(
        query_id="q-top-k",
        task="KIS",
        visual_queries=["city bus"],
    )

    hits = asyncio.run(
        runner.run_fast_path(query, top_k={"clip_search": 7})
    )

    assert len(hits) == 1
