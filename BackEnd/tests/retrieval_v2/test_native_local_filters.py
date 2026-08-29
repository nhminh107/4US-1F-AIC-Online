from __future__ import annotations

import asyncio

from BackEnd.app.retrieval_tools.object import (
    configure_object_search_manager,
    object_search,
)
from BackEnd.app.retrieval_tools.text import asr_search, configure_text_search


def test_asr_search_sends_video_and_time_filters_to_elasticsearch():
    class Client:
        def __init__(self) -> None:
            self.request = None

        async def search(self, **kwargs):
            self.request = kwargs
            return {"hits": {"hits": []}}

    client = Client()
    configure_text_search(client=client)

    asyncio.run(
        asr_search(
            "đập thủy lợi",
            mode="fuzzy",
            video_ids=["V1", "V2"],
            start_ms=1_000,
            end_ms=9_000,
        )
    )

    filters = client.request["query"]["bool"]["filter"]
    assert {"terms": {"video_id": ["V1", "V2"]}} in filters
    assert {"range": {"start_ms": {"lte": 9_000}}} in filters
    assert {"range": {"end_ms": {"gte": 1_000}}} in filters


def test_object_search_passes_video_and_time_scope_to_database_query():
    class Manager:
        def __init__(self) -> None:
            self.kwargs = None

        def search_object_detections(self, object_class, **kwargs):
            self.kwargs = kwargs
            return []

    manager = Manager()
    configure_object_search_manager(manager)

    asyncio.run(
        object_search(
            "person",
            video_ids=["V3"],
            start_ms=2_000,
            end_ms=7_000,
        )
    )

    assert manager.kwargs["video_ids"] == ["V3"]
    assert manager.kwargs["start_ms"] == 2_000
    assert manager.kwargs["end_ms"] == 7_000
