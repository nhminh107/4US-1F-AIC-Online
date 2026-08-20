from types import SimpleNamespace

from BackEnd.app.retrieval.text_object_retrieval import (
    ObjectTrackingRetrievalTools,
    TextRetrievalTools,
)


class FakeElasticsearch:
    def search(self, *, index, **kwargs):
        assert index == "ocr_index"
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "OCR1",
                        "_score": 4.2,
                        "_source": {
                            "entity_id": "OCR1",
                            "video_id": "L01_V001",
                            "frame_id": "F001",
                            "timestamp_ms": 1200,
                        },
                    }
                ]
            }
        }


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _statement):
        return FakeResult(self.rows)


def test_ocr_search_returns_shared_search_hit():
    hits = TextRetrievalTools(FakeElasticsearch()).ocr_search("HCMC")
    assert len(hits) == 1
    assert hits[0].entity_type == "ocr"
    assert hits[0].frame_id == "F001"
    assert hits[0].start_ms == hits[0].end_ms == 1200


def test_track_search_preserves_event_and_canonical_range():
    track = SimpleNamespace(
        track_id=7,
        shot_id="S001",
        start_ms=1000,
        end_ms=3000,
        avg_confidence=0.8,
    )
    shot = SimpleNamespace(video_id="L01_V001")
    object_type = SimpleNamespace(class_name="Person")
    tools = ObjectTrackingRetrievalTools(
        lambda: FakeSession([(track, shot, object_type)])
    )
    hits = tools.track_search("Person", event_id="E1")
    assert len(hits) == 1
    assert hits[0].entity_type == "object_track"
    assert hits[0].event_id == "E1"
    assert hits[0].start_ms == 1000
    assert hits[0].end_ms == 3000
