from __future__ import annotations

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
                            "video_id": "video-1",
                            "start_ms": 10,
                            "end_ms": 20,
                        },
                    }
                ]
            }
        }


@pytest.mark.asyncio
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

    assert client.requests[0]["index"] == "ocr_index"
    assert hits[0].entity_type == "ocr"
    assert hits[0].event_id == "event-1"
    assert hits[0].tool_call_id == "tc_001"

    text_tools.configure_text_search(client=None)


@pytest.mark.asyncio
async def test_object_retrieval_uses_app_entity_types():
    client = _FakeElasticsearch()
    object_tools.configure_object_search_client(client)

    object_hits = await object_tools.object_search("person")
    track_hits = await object_tools.track_search("person")

    assert object_hits[0].entity_type == "object_detection"
    assert track_hits[0].entity_type == "object_track"

    object_tools.configure_object_search_client(None)
