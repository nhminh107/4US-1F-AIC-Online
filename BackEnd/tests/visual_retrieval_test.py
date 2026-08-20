"""Test script cho Module 3.1 - Visual Retrieval Tools.

Theo đúng convention test hiện có của repo (xem ``aggregator_test.py``):
script độc lập, không phụ thuộc pytest, chạy trực tiếp bằng:

    python -m BackEnd.tests.visual_retrieval_test

Chỉ test phần logic thuần (không cần GPU/model CLIP thật, không cần
PostgreSQL/FAISS thật) bằng cách thay ``registry``/``embedder``/``db_mng``
bằng các fake nhẹ triển khai đúng interface mà ``VisualRetrievalTools``
cần (duck typing) - tương ứng các test case "Unit (no GPU)" ở mục 9
``Markdown_Doc/module_visual_retrieval_tools.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BackEnd.app.contracts.pipeline import (
    ClipEmbeddingHit,
    FrameEmbeddingHit,
    FrameMetadata,
    ShotEmbeddingHit,
)
from BackEnd.app.retrieval.visual_retrieval import (
    DEFAULT_FAISS_INDEX_MODEL_NAME,
    DEFAULT_TEXT_EMBEDDING_MODEL_NAME,
    FaissIndexRegistry,
    VisualRetrievalTools,
    VisualSearchConfig,
    build_default_visual_retrieval_tools,
)


# ---------------------------------------------------------------------------
# Fakes - triển khai đúng interface mà VisualRetrievalTools cần, không đụng
# tới FAISS/PostgreSQL/model CLIP thật.
# ---------------------------------------------------------------------------


class FakeRegistry:
    """Giả lập FaissIndexRegistry: trả về danh sách (faiss_id, score) đã
    set trước, không đọc file .faiss nào cả."""

    def __init__(self, frame_hits=None, clip_hits=None, shot_hits=None):
        self._frame_hits = frame_hits or []
        self._clip_hits = clip_hits or []
        self._shot_hits = shot_hits or []
        self.last_top_k: int | None = None

    def search_frame_index(self, vector: np.ndarray, top_k: int):
        self.last_top_k = top_k
        return self._frame_hits[:top_k]

    def search_clip_index(self, vector: np.ndarray, top_k: int):
        self.last_top_k = top_k
        return self._clip_hits[:top_k]

    def search_shot_index(self, vector: np.ndarray, top_k: int):
        self.last_top_k = top_k
        return self._shot_hits[:top_k]


class FakeEmbedder:
    """Giả lập ClipEmbedder: luôn trả về 1 vector hằng số, không load model."""

    def encode_text(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)


class FakeDbManager:
    """Giả lập các hàm resolve của PostgreManager mà VisualRetrievalTools
    gọi tới, dữ liệu lấy từ dict truyền vào lúc khởi tạo."""

    def __init__(
        self,
        frame_hits: dict[int, FrameEmbeddingHit] | None = None,
        clip_hits: dict[int, ClipEmbeddingHit] | None = None,
        shot_hits: dict[int, ShotEmbeddingHit] | None = None,
        frame_records: dict[str, FrameMetadata] | None = None,
    ):
        self._frame_hits = frame_hits or {}
        self._clip_hits = clip_hits or {}
        self._shot_hits = shot_hits or {}
        self._frame_records = frame_records or {}

    def get_frame_hits_by_faiss_ids(self, faiss_ids, *, index_version, model_name):
        return [self._frame_hits[fid] for fid in faiss_ids if fid in self._frame_hits]

    def get_clip_hits_by_faiss_ids(self, faiss_ids, *, index_version, model_name):
        return [self._clip_hits[fid] for fid in faiss_ids if fid in self._clip_hits]

    def get_shot_hits_by_faiss_ids(
        self, faiss_ids, *, index_version, model_name, model_version, pooling_method
    ):
        return [self._shot_hits[fid] for fid in faiss_ids if fid in self._shot_hits]

    def get_frame_record_by_frame_id(self, frame_id: str) -> FrameMetadata:
        if frame_id not in self._frame_records:
            raise ValueError(f"Frame '{frame_id}' does not exist.")
        return self._frame_records[frame_id]


def _make_tools(registry: FakeRegistry, db_mng: FakeDbManager, **config_overrides) -> VisualRetrievalTools:
    config = VisualSearchConfig(**config_overrides) if config_overrides else None
    return VisualRetrievalTools(
        registry=registry,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        db_mng=db_mng,  # type: ignore[arg-type]
        config=config,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_frame_search_query_and_image_ref_both_set_raises():
    tools = _make_tools(FakeRegistry(), FakeDbManager())
    try:
        tools.frame_search(query="a", image_ref="F1")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when both query and image_ref are set.")


def test_frame_search_no_args_raises():
    tools = _make_tools(FakeRegistry(), FakeDbManager())
    try:
        tools.frame_search()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when neither query nor image_ref is set.")


def test_top_k_is_clamped_to_max_top_k():
    registry = FakeRegistry(frame_hits=[(i, 1.0 - i * 0.01) for i in range(1, 300)])
    db_mng = FakeDbManager(
        frame_hits={
            i: FrameEmbeddingHit(
                faiss_id=i, frame_id=f"F{i}", video_id="V1", shot_id="S1",
                start_ms=i * 1000, end_ms=i * 1000,
            )
            for i in range(1, 300)
        }
    )
    tools = _make_tools(registry, db_mng, default_top_k=50, max_top_k=100)
    tools.frame_search(query="anything", top_k=999)
    assert registry.last_top_k == 100, f"Expected clamp to 100, got {registry.last_top_k}"


def test_unresolved_faiss_id_is_skipped_not_raised():
    registry = FakeRegistry(frame_hits=[(1, 0.9), (2, 0.8)])
    db_mng = FakeDbManager(
        frame_hits={
            1: FrameEmbeddingHit(
                faiss_id=1, frame_id="F1", video_id="V1", shot_id="S1",
                start_ms=1000, end_ms=1000,
            )
            # faiss_id=2 co tinh khong co trong DB -> mo phong DB<->FAISS lech du lieu
        }
    )
    tools = _make_tools(registry, db_mng)
    hits = tools.frame_search(query="anything")
    assert len(hits) == 1, f"Expected 1 resolved hit, got {len(hits)}"
    assert hits[0].entity_id == "F1"


def test_event_id_and_rank_propagate_correctly():
    registry = FakeRegistry(clip_hits=[(10, 0.95), (11, 0.80), (12, 0.60)])
    db_mng = FakeDbManager(
        clip_hits={
            10: ClipEmbeddingHit(faiss_id=10, clip_id="C10", video_id="V1", shot_id="S1", start_ms=0, end_ms=1000),
            11: ClipEmbeddingHit(faiss_id=11, clip_id="C11", video_id="V1", shot_id="S1", start_ms=1000, end_ms=2000),
            12: ClipEmbeddingHit(faiss_id=12, clip_id="C12", video_id="V1", shot_id="S2", start_ms=2000, end_ms=3000),
        }
    )
    tools = _make_tools(registry, db_mng)
    hits = tools.clip_search(query="a scene", event_id="E2", tool_call_id="T01")

    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.raw_score for h in hits] == [0.95, 0.80, 0.60]
    assert all(h.event_id == "E2" for h in hits)
    assert all(h.tool_call_id == "T01" for h in hits)
    assert all(h.source == "clip_embedding" and h.entity_type == "clip" for h in hits)


def test_shot_search_uses_shot_source_and_entity_type():
    registry = FakeRegistry(shot_hits=[(5, 0.7)])
    db_mng = FakeDbManager(
        shot_hits={5: ShotEmbeddingHit(faiss_id=5, shot_id="S5", video_id="V9", start_ms=0, end_ms=4000)}
    )
    tools = _make_tools(registry, db_mng)
    hits = tools.shot_search(query="a scene")

    assert len(hits) == 1
    assert hits[0].source == "shot_embedding"
    assert hits[0].entity_type == "shot"
    assert hits[0].entity_id == "S5"


def test_image_similarity_search_is_alias_of_frame_search_image_ref():
    registry = FakeRegistry(frame_hits=[(1, 0.9)])
    db_mng = FakeDbManager(
        frame_hits={
            1: FrameEmbeddingHit(
                faiss_id=1, frame_id="F1", video_id="V1", shot_id="S1",
                start_ms=1000, end_ms=1000,
            )
        },
        frame_records={"F1": FrameMetadata(
            frame_id="F1", video_id="V1", shot_id="S1", timestamp_ms=1000,
            fps=25.0, frame_idx=0, frame_path=Path("dummy.jpg"),
        )},
    )
    tools = _make_tools(registry, db_mng)
    hits = tools.image_similarity_search(image_ref="F1", event_id="E1")
    assert len(hits) == 1
    assert hits[0].entity_type == "frame"
    assert hits[0].source == "frame_embedding"
    assert hits[0].event_id == "E1"


def test_image_ref_missing_in_db_raises_value_error():
    tools = _make_tools(FakeRegistry(), FakeDbManager())
    try:
        tools.frame_search(image_ref="does_not_exist")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when image_ref does not resolve in DB.")


def test_default_wiring_separates_text_embedding_and_faiss_index_models():
    registry = FakeRegistry()
    db_mng = FakeDbManager()
    fake_embedder = FakeEmbedder()
    environment = {
        "TEXT_EMBEDDING_MODEL_NAME": DEFAULT_TEXT_EMBEDDING_MODEL_NAME,
        "FAISS_INDEX_MODEL_NAME": DEFAULT_FAISS_INDEX_MODEL_NAME,
        "CLIP_MODEL_DEVICE": "cpu",
    }
    with (
        patch.dict(os.environ, environment, clear=False),
        patch.object(FaissIndexRegistry, "get_instance", return_value=registry),
        patch(
            "BackEnd.app.retrieval.visual_retrieval.ClipEmbedder",
            return_value=fake_embedder,
        ) as embedder_class,
    ):
        tools = build_default_visual_retrieval_tools(db_mng=db_mng)  # type: ignore[arg-type]

    embedder_class.assert_called_once_with(
        model_name=DEFAULT_TEXT_EMBEDDING_MODEL_NAME,
        image_model_name=DEFAULT_FAISS_INDEX_MODEL_NAME,
        device="cpu",
    )
    assert tools.config.model_name == DEFAULT_FAISS_INDEX_MODEL_NAME


# ---------------------------------------------------------------------------
# _raw_search: test truc tiep FaissIndexRegistry._raw_search bang 1 "index"
# gia lap (duck-typed) co method .search(), khong can cai dat thu vien faiss
# de chay unit test nay.
# ---------------------------------------------------------------------------


class FakeFaissIndex:
    """Giả lập faiss.Index.search(): trả (scores, ids) dạng ndarray 2 chiều,
    có lẫn -1 để mô phỏng trường hợp index có ít hơn top_k vector."""

    def search(self, vector: np.ndarray, top_k: int):
        ids = np.array([[3, 7, -1]], dtype=np.int64)
        scores = np.array([[0.9, 0.5, 0.0]], dtype=np.float32)
        return scores, ids


def _assert_close(result, expected):
    assert [fid for fid, _ in result] == [fid for fid, _ in expected]
    for (_, got_score), (_, want_score) in zip(result, expected):
        assert abs(got_score - want_score) < 1e-6, f"Unexpected result: {result}"


def test_raw_search_filters_out_negative_one_ids():
    result = FaissIndexRegistry._raw_search(FakeFaissIndex(), np.ones(8, dtype=np.float32), top_k=3)
    assert len(result) == 2, f"Expected faiss_id=-1 to be filtered out, got: {result}"
    _assert_close(result, [(3, 0.9), (7, 0.5)])


def test_raw_search_casts_float64_vector_to_float32():
    vector = np.ones(8, dtype=np.float64)
    result = FaissIndexRegistry._raw_search(FakeFaissIndex(), vector, top_k=3)
    _assert_close(result, [(3, 0.9), (7, 0.5)])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    test_frame_search_query_and_image_ref_both_set_raises,
    test_frame_search_no_args_raises,
    test_top_k_is_clamped_to_max_top_k,
    test_unresolved_faiss_id_is_skipped_not_raised,
    test_event_id_and_rank_propagate_correctly,
    test_shot_search_uses_shot_source_and_entity_type,
    test_image_similarity_search_is_alias_of_frame_search_image_ref,
    test_image_ref_missing_in_db_raises_value_error,
    test_default_wiring_separates_text_embedding_and_faiss_index_models,
    test_raw_search_filters_out_negative_one_ids,
    test_raw_search_casts_float64_vector_to_float32,
]


def run_visual_retrieval_test():
    print("=" * 80)
    print(" VISUAL RETRIEVAL TOOLS - UNIT TEST (no GPU, no DB, no FAISS that) ")
    print("=" * 80)
    for test_fn in _ALL_TESTS:
        test_fn()
        print(f"[OK] {test_fn.__name__}")
    print("-" * 80)
    print(f">>> ALL {len(_ALL_TESTS)} TESTS PASSED <<<")


if __name__ == "__main__":
    run_visual_retrieval_test()
