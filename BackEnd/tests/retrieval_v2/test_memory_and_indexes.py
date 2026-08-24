import numpy as np
import pytest
import json

from BackEnd.app.retrieval_v2.embedding_registry import (
    EmbeddingBackendRegistry,
    EmbeddingBackendSpec,
)
from BackEnd.app.retrieval_v2.memory import GoldRecord, JsonlGoldMemory
from BackEnd.app.retrieval_v2.corpus_stats import CorpusDocument, CorpusStats
from BackEnd.app.retrieval_v2.readiness import inspect_retrieval_artifacts
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex


def test_gold_memory_round_trips_confirmed_and_rejected_regions(tmp_path):
    store = JsonlGoldMemory(tmp_path / "gold.jsonl")
    record = GoldRecord(
        query_id="q1",
        query_text="a lion dance on poles",
        task="KIS",
        accepted_video_ids=["V1"],
        accepted_intervals_ms={"V1": [[10_000, 20_000]]},
        rejected_regions=["V2:0-5000"],
        effective_prompts=["lion dancer jumping between poles"],
    )

    store.append(record)

    assert store.list_records() == [record]


def test_video_level_index_searches_aggregated_video_vectors(tmp_path):
    index = VideoLevelIndex.build(
        {
            "V1": np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
            "V2": np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        }
    )
    path = tmp_path / "video-index.npz"
    index.save(path)

    loaded = VideoLevelIndex.load(path)
    results = loaded.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [video_id for video_id, _score in results] == ["V1", "V2"]


def test_video_level_index_preserves_a_rare_representative():
    index = VideoLevelIndex.build(
        {
            "V-rare": np.array(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            "V-generic": np.array([[0.8, 0.6]], dtype=np.float32),
        },
        representatives_per_video=2,
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert results[0][0] == "V-rare"
    assert len({video_id for video_id, _score in results}) == 2


def test_versioned_video_index_rejects_missing_or_mismatched_manifest(tmp_path):
    path = tmp_path / "video.npz"
    index = VideoLevelIndex.build({"V1": np.array([[1.0, 0.0]], dtype=np.float32)})
    index.save(path)

    with pytest.raises(ValueError, match="Missing video index manifest"):
        VideoLevelIndex.load_versioned(path)

    path.with_suffix(".manifest.json").write_text(
        json.dumps({
            "schema_version": "video-level-index-v1",
            "source_entity": "shot",
            "representative_count": 99,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="representative counts differ"):
        VideoLevelIndex.load_versioned(path)


def test_retrieval_artifact_readiness_validates_both_versioned_files(tmp_path):
    stats_path = tmp_path / "corpus_stats.json"
    CorpusStats.from_documents([
        CorpusDocument(document_id="D1", video_id="V1", text="lion on platform")
    ]).save(stats_path)
    index_path = tmp_path / "video_index.npz"
    VideoLevelIndex.build({"V1": np.array([[1.0, 0.0]], dtype=np.float32)}).save(index_path)
    index_path.with_suffix(".manifest.json").write_text(
        json.dumps({
            "schema_version": "video-level-index-v1",
            "source_entity": "shot",
            "representative_count": 1,
        }),
        encoding="utf-8",
    )

    inspect_retrieval_artifacts.cache_clear()
    ready = inspect_retrieval_artifacts(str(stats_path), str(index_path))

    assert ready.corpus_stats_ready is True
    assert ready.video_index_ready is True
    assert ready.degraded_reasons == ()


def test_shadow_embedding_backend_requires_explicit_promotion():
    registry = EmbeddingBackendRegistry()
    registry.register(
        EmbeddingBackendSpec(
            backend_id="legacy_clip_b32",
            model_name="clip-ViT-B-32",
            entity_type="frame",
            status="active",
        )
    )
    registry.register(
        EmbeddingBackendSpec(
            backend_id="siglip2_shadow",
            model_name="siglip2",
            entity_type="frame",
            status="shadow",
        )
    )

    with pytest.raises(ValueError):
        registry.promote("siglip2_shadow", approved=False)

    registry.promote("siglip2_shadow", approved=True)
    assert registry.active("frame").backend_id == "siglip2_shadow"
