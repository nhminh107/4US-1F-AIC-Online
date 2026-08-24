from __future__ import annotations

from types import SimpleNamespace

import pytest

from BackEnd.app.retrieval_tools import object as object_tools
from BackEnd.app.retrieval_tools import text as text_tools


class _FakeElasticsearch:
    def __init__(self):
        self.requests = []

    async def search(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "entity-1",
                        "_score": 0.75,
        "_source": {
                            "entity_id": "frame-1",
                            "frame_id": "frame-1",
                            "video_id": "video-1",
                            "timestamp_ms": 10,
                        },
                    }
                ]
            }
        }


@pytest.mark.anyio
async def test_text_retrieval_returns_search_hit_contract():
    client = _FakeElasticsearch()
    text_tools.configure_text_search(client=client)

    hits = await text_tools.ocr_search(
        "HCMC",
        top_k=3,
        mode="exact",
        event_id="event-1",
        tool_call_id="tc_001",
    )

    assert client.requests[0]["index"] == "aic_hcm2026_text_ocr_active"
    assert hits[0].entity_type == "ocr"
    assert hits[0].entity_id == "frame-1"
    assert hits[0].frame_id == "frame-1"
    assert hits[0].start_ms == hits[0].end_ms == 10
    assert hits[0].event_id == "event-1"
    assert hits[0].tool_call_id == "tc_001"

    text_tools.configure_text_search(client=None)


@pytest.mark.anyio
async def test_asr_retrieval_uses_transcript_alias_and_full_text_by_default():
    client = _FakeElasticsearch()
    text_tools.configure_text_search(client=client)

    hits = await text_tools.asr_search("Việt Nam", top_k=3)

    assert client.requests[0]["index"] == "aic_hcm2026_text_transcript_active"
    assert (
        "bool" in client.requests[0]["query"]
        or "match" in client.requests[0]["query"]
    )
    assert hits[0].entity_type == "asr"

    text_tools.configure_text_search(client=None)


@pytest.mark.anyio
async def test_object_retrieval_uses_postgresql_canonical_references():
    class _FakePostgreManager:
        def search_object_detections(self, object_class: str):
            assert object_class == "person"
            return [
                (
                    SimpleNamespace(detection_id=1, confidence=0.7),
                    SimpleNamespace(
                        frame_id="frame-1",
                        video_id="video-1",
                        shot_id="shot-1",
                        timestamp_ms=100,
                    ),
                ),
                (
                    SimpleNamespace(detection_id=2, confidence=0.9),
                    SimpleNamespace(
                        frame_id="frame-1",
                        video_id="video-1",
                        shot_id="shot-1",
                        timestamp_ms=100,
                    ),
                ),
                (
                    SimpleNamespace(detection_id=4, confidence=0.95),
                    SimpleNamespace(
                        frame_id="frame-2",
                        video_id="video-1",
                        shot_id="shot-1",
                        timestamp_ms=200,
                    ),
                ),
            ]

        def search_object_tracks(self, object_class: str):
            assert object_class == "person"
            return [
                (
                    SimpleNamespace(track_id=3, start_ms=50, end_ms=250, avg_confidence=0.8),
                    SimpleNamespace(shot_id="shot-1", video_id="video-1"),
                )
            ]

    object_tools.configure_object_search_manager(_FakePostgreManager())

    object_hits = await object_tools.object_search("person", min_count=2)
    track_hits = await object_tools.track_search("person")

    assert object_hits[0].entity_type == "object_detection"
    assert object_hits[0].frame_id == "frame-1"
    assert object_hits[0].source == "postgresql_object_detection"
    assert track_hits[0].entity_type == "object_track"
    assert track_hits[0].shot_id == "shot-1"
    assert track_hits[0].source == "postgresql_object_track"

    object_tools.configure_object_search_manager(None)


@pytest.mark.anyio
async def test_object_count_ignores_low_confidence_and_duplicate_boxes():
    def detection(detection_id, confidence, box):
        return SimpleNamespace(
            detection_id=detection_id,
            confidence=confidence,
            x_min=box[0],
            x_max=box[1],
            y_min=box[2],
            y_max=box[3],
        )

    frame = SimpleNamespace(
        frame_id="frame-1",
        video_id="video-1",
        shot_id="shot-1",
        timestamp_ms=100,
    )

    class Manager:
        def search_object_detections(self, *_args, **_kwargs):
            return [
                (detection(1, 0.95, (0.1, 0.4, 0.1, 0.5)), frame),
                (detection(2, 0.90, (0.11, 0.41, 0.11, 0.51)), frame),
                (detection(3, 0.40, (0.6, 0.8, 0.2, 0.6)), frame),
            ]

    object_tools.configure_object_search_manager(Manager())

    hits = await object_tools.object_search("person", min_count=2)

    assert hits == []
    object_tools.configure_object_search_manager(None)
