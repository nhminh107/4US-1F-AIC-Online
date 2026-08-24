from __future__ import annotations

import asyncio

import numpy as np

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.contracts import CandidateBudget, SearchCall
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex


def test_video_lane_rescues_video_missing_from_global_moment_hits_for_local_search():
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[SearchCall] = []

        async def search(self, call: SearchCall):
            self.calls.append(call)
            if call.call_id.startswith("global_"):
                return [
                    SearchHit(
                        source="frame_embedding",
                        entity_type="frame",
                        entity_id=f"{call.call_id}-v1",
                        video_id="V1",
                        start_ms=1_000,
                        end_ms=1_000,
                        rank=1,
                        raw_score=0.8,
                    )
                ]
            return []

    index = VideoLevelIndex.build(
        {
            "V1": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "V2": np.asarray([[0.0, 1.0]], dtype=np.float32),
        }
    )
    gateway = Gateway()
    controller = SearchController(
        gateway,
        video_index=index,
        video_query_encoder=lambda _text: np.asarray([0.0, 1.0], dtype=np.float32),
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=10,
            unique_candidate_min=2,
            unique_candidate_max=5,
            moment_band_limit=10,
            video_shortlist_limit=2,
            local_retrieval_k=4,
            retry_retrieval_k=2,
            rerank_limit=5,
            max_retry_rounds=0,
        ),
    )

    asyncio.run(
        controller.search(
            StructuredQuery(
                query_id="video-lane-rescue",
                task="KIS",
                visual_queries=["a rare lion dance pumpkin bite"],
            )
        )
    )

    local_video_ids = {
        call.video_ids[0]
        for call in gateway.calls
        if call.call_id.startswith("local_") and call.video_ids
    }
    assert local_video_ids == {"V1", "V2"}
