"""Tests for Phase H: adapters, cache, metrics, logging, and video level index integration."""

import asyncio
import numpy as np

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.adapters.reranker import (
    ClipOfficialFrameReranker,
    DeterministicFallbackReranker,
)
from BackEnd.app.retrieval_v2.adapters.vlm_reviewer import DeterministicFallbackReviewer
from BackEnd.app.retrieval_v2.cache import QueryEmbeddingCache
from BackEnd.app.retrieval_v2.contracts import CandidateBudget, MomentBand, QueryAtom, RetrievalPlan, SearchCall
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.logging import build_search_audit_log, emit_audit_log
from BackEnd.app.retrieval_v2.metrics import RetrievalMetricsCollector
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex
from BackEnd.app.retrieval_v2.frame_selector import FrameCandidate


def test_query_embedding_cache_lru_and_hit_ratio():
    cache = QueryEmbeddingCache(maxsize=2)
    assert cache.hit_ratio == 0.0

    cache.put("m1", "query a", [0.1, 0.2])
    cache.put("m1", "query b", [0.3, 0.4])

    assert cache.get("m1", "query a") == [0.1, 0.2]  # Hit
    assert cache.get("m1", "query c") is None         # Miss
    assert cache.hit_ratio == 0.5

    # Insert 3rd item -> evicts oldest unaccessed ("query b")
    cache.put("m1", "query d", [0.5, 0.6])
    assert cache.get("m1", "query b") is None
    assert cache.get("m1", "query a") == [0.1, 0.2]


def test_metrics_collector_session_and_summary():
    collector = RetrievalMetricsCollector()
    s1 = collector.start_session("q1", "KIS")
    s1.total_latency_ms = 10.5
    s1.raw_hit_count = 150
    s1.moment_band_count = 20

    s2 = collector.start_session("q2", "VQA")
    s2.total_latency_ms = 5.5
    s2.raw_hit_count = 50
    s2.moment_band_count = 10

    summary = collector.summary()
    assert summary["total_queries"] == 2
    assert summary["avg_latency_ms"] == 8.0
    assert summary["total_raw_hits"] == 200
    assert summary["total_bands"] == 30


def test_reranker_fallback_sorts_bands_deterministically():
    reranker = DeterministicFallbackReranker()
    b1 = MomentBand(band_id="b1", video_id="V1", start_ms=1000, end_ms=2000, peak_ms=1500, score=0.4)
    b2 = MomentBand(band_id="b2", video_id="V2", start_ms=1000, end_ms=2000, peak_ms=1500, score=0.9)
    atoms = [QueryAtom(atom_id="A1", text="test", modality="visual", discriminative_weight=1.0)]

    reranked = asyncio.run(reranker.rerank([b1, b2], atoms, limit=1))
    assert len(reranked) == 1
    assert reranked[0].band_id == "b2"


def test_official_frame_reranker_can_promote_atom_image_alignment(tmp_path):
    first = tmp_path / "V1.jpg"
    second = tmp_path / "V2.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    class Provider:
        def get_official_frames(self, video_id, start_ms, end_ms):
            path = first if video_id == "V1" else second
            return [FrameCandidate(
                video_id=video_id,
                frame_idx=1,
                timestamp_ms=1_500,
                img_url=str(path),
            )]

    class Embedder:
        def encode_texts(self, _texts):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def encode_image(self, image_path):
            return np.asarray(
                [1.0, 0.0] if image_path.endswith("V1.jpg") else [0.0, 1.0],
                dtype=np.float32,
            )

    bands = [
        MomentBand(band_id="b1", video_id="V1", start_ms=1_000, end_ms=2_000, peak_ms=1_500, score=0.6),
        MomentBand(band_id="b2", video_id="V2", start_ms=1_000, end_ms=2_000, peak_ms=1_500, score=0.8),
    ]
    atom = QueryAtom(atom_id="A1", text="target", modality="visual", discriminative_weight=1.0)

    reranked = asyncio.run(
        ClipOfficialFrameReranker(Provider(), Embedder()).rerank(bands, [atom], 2)
    )

    assert [band.band_id for band in reranked] == ["b1", "b2"]
    assert reranked[0].score_breakdown["fine_grained_visual"] == 1.0


def test_vlm_reviewer_fallback_emits_structured_reviews():
    reviewer = DeterministicFallbackReviewer()
    b1 = MomentBand(band_id="b1", video_id="V1", start_ms=1000, end_ms=2000, peak_ms=1500, score=0.8)
    plan = RetrievalPlan(query_id="q1", task="KIS", execution_profile="KIS_MOMENT", atoms=[
        QueryAtom(atom_id="A1", text="car", modality="visual", discriminative_weight=1.0)
    ])

    reviews = asyncio.run(reviewer.review(plan, [b1]))
    assert len(reviews) == 1
    assert reviews[0].band_id == "b1"
    assert reviews[0].verdict == "uncertain"
    assert reviews[0].confidence == 0.0


def test_controller_with_video_index_and_audit_logging():
    class SimpleGateway:
        async def search(self, call: SearchCall):
            return [
                SearchHit(
                    source="clip_embedding",
                    entity_type="clip",
                    entity_id=f"{call.call_id}-1",
                    video_id="V001",
                    start_ms=1000,
                    end_ms=3000,
                    rank=1,
                    raw_score=0.9,
                    atom_id=call.atom_id,
                    retriever_family="legacy_clip_b32",
                )
            ]

    # Build dummy video index
    vectors = {"V001": np.ones((2, 4), dtype=np.float32), "V002": np.zeros((2, 4), dtype=np.float32)}
    vindex = VideoLevelIndex.build(vectors)

    controller = SearchController(
        SimpleGateway(),
        video_index=vindex,
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=20,
            moment_band_limit=5,
            video_shortlist_limit=2,
            local_retrieval_k=5,
            rerank_limit=5,
        ),
    )
    query = StructuredQuery(query_id="q-audit", task="KIS", visual_queries=["car"])
    result = asyncio.run(controller.search(query))

    assert result.bands
    audit = build_search_audit_log(result)
    assert audit["query_id"] == "q-audit"
    assert audit["task"] == "KIS"
    assert audit["raw_hits"] >= 1
    # emit_audit_log should not raise
    emit_audit_log(result)
