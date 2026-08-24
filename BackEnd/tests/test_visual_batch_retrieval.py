from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from BackEnd.app.retrieval import visual_retrieval
from BackEnd.app.retrieval.visual_retrieval import ClipEmbedder, VisualRetrievalTools
from BackEnd.app.retrieval_tools import visual as visual_tools


class RecordingTextModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool, bool]] = []

    def encode(self, texts, *, convert_to_numpy, normalize_embeddings):
        self.calls.append((list(texts), convert_to_numpy, normalize_embeddings))
        return np.asarray(
            [[float(index + 1), float((index + 1) * 10)] for index in range(len(texts))],
            dtype=np.float64,
        )


def test_clip_embedder_encode_texts_uses_one_model_call_and_preserves_order():
    model = RecordingTextModel()
    embedder = object.__new__(ClipEmbedder)
    embedder._text_model = model

    vectors = embedder.encode_texts(["first prompt", "second prompt", "third prompt"])

    assert model.calls == [
        (
            ["first prompt", "second prompt", "third prompt"],
            True,
            True,
        )
    ]
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 2)
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]


class BatchEmbedder:
    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encoded_batches.append(list(texts))
        return np.asarray(
            [[float(index + 1), 0.0] for index in range(len(texts))],
            dtype=np.float32,
        )

    def encode_text(self, _text: str) -> np.ndarray:
        raise AssertionError("native batch must not fall back to per-request encode_text()")


class VectorAwareRegistry:
    @staticmethod
    def _result(vector: np.ndarray, offset: int):
        return [(offset + int(vector[0]), 0.9)]

    def search_frame_index(self, vector: np.ndarray, _top_k: int):
        return self._result(vector, 100)

    def search_clip_index(self, vector: np.ndarray, _top_k: int):
        return self._result(vector, 200)

    def search_shot_index(self, vector: np.ndarray, _top_k: int):
        return self._result(vector, 300)


class BatchDbManager:
    def get_frame_hits_by_faiss_ids(self, faiss_ids, **_kwargs):
        return [
            SimpleNamespace(
                faiss_id=faiss_id,
                frame_id=f"F{faiss_id}",
                video_id="V-frame",
                shot_id="S-frame",
                start_ms=1_000,
                end_ms=1_000,
            )
            for faiss_id in faiss_ids
        ]

    def get_clip_hits_by_faiss_ids(self, faiss_ids, **_kwargs):
        return [
            SimpleNamespace(
                faiss_id=faiss_id,
                clip_id=f"C{faiss_id}",
                video_id="V-clip",
                shot_id="S-clip",
                start_ms=2_000,
                end_ms=4_000,
            )
            for faiss_id in faiss_ids
        ]

    def get_shot_hits_by_faiss_ids(self, faiss_ids, **_kwargs):
        return [
            SimpleNamespace(
                faiss_id=faiss_id,
                shot_id=f"S{faiss_id}",
                video_id="V-shot",
                start_ms=5_000,
                end_ms=9_000,
            )
            for faiss_id in faiss_ids
        ]


def test_native_mixed_visual_batch_preserves_request_order_and_metadata():
    request = visual_retrieval.VisualSearchRequest
    requests = [
        request(
            retriever="shot_search",
            query="wide zoo enclosure",
            top_k=7,
            event_id="E3",
            tool_call_id="T-shot",
        ),
        request(
            retriever="frame_search",
            query="yellow flower on a pumpkin",
            top_k=5,
            event_id="E1",
            tool_call_id="T-frame",
        ),
        request(
            retriever="clip_search",
            query="lion dancer jumping between poles",
            top_k=6,
            event_id="E2",
            tool_call_id="T-clip",
        ),
    ]
    embedder = BatchEmbedder()
    tools = VisualRetrievalTools(
        registry=VectorAwareRegistry(),
        embedder=embedder,
        db_mng=BatchDbManager(),
    )

    results = tools.search_many(requests)

    assert embedder.encoded_batches == [[request.query for request in requests]]
    assert len(results) == len(requests)
    assert [[hit.entity_type for hit in group] for group in results] == [
        ["shot"],
        ["frame"],
        ["clip"],
    ]
    assert [[hit.entity_id for hit in group] for group in results] == [
        ["S301"],
        ["F102"],
        ["C203"],
    ]
    assert [group[0].event_id for group in results] == ["E3", "E1", "E2"]
    assert [group[0].tool_call_id for group in results] == [
        "T-shot",
        "T-frame",
        "T-clip",
    ]


def test_async_visual_batch_wrapper_delegates_once_and_preserves_nested_results():
    request = visual_retrieval.VisualSearchRequest(
        retriever="frame_search",
        query="two keepers weighing an animal",
        top_k=4,
        event_id="E1",
        tool_call_id="T1",
    )

    class FakeTools:
        def __init__(self) -> None:
            self.calls = []

        def search_many(self, requests):
            self.calls.append(requests)
            return [["resolved-frame-hit"]]

    fake_tools = FakeTools()
    visual_tools.configure_visual_retrieval_tools(fake_tools)
    try:
        results = asyncio.run(visual_tools.search_many([request]))
    finally:
        visual_tools.configure_visual_retrieval_tools(None)

    assert fake_tools.calls == [[request]]
    assert results == [["resolved-frame-hit"]]

