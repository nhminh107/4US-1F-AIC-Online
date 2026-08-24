from __future__ import annotations

import asyncio
import logging

from BackEnd.CONFIG import TOOL_TIMEOUTS
from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_tools.object import object_search, track_search
from BackEnd.app.retrieval_tools.text import asr_search, ocr_search
from BackEnd.app.retrieval.visual_retrieval import VisualSearchRequest
from BackEnd.app.retrieval_tools.visual import (
    clip_search,
    frame_search,
    search_many as visual_search_many,
    shot_search,
)
from BackEnd.app.retrieval_v2.contracts import SearchCall

logger = logging.getLogger(__name__)

# Pipeline.md §6C: Minimum exact hits before falling through to fuzzy
_MIN_OCR_EXACT_HITS = 5
_VISUAL_BATCH_SIZE = 12


class ToolSearchGateway:
    """Execute V2 calls through the repository's typed retrieval tools."""

    async def search_many(self, calls: list[SearchCall]) -> list[list[SearchHit]]:
        """Use one native embedding batch for visual calls; batch I/O otherwise."""

        results: list[list[SearchHit] | None] = [None] * len(calls)
        visual_indexes = [
            index
            for index, call in enumerate(calls)
            if call.retriever in {"frame_search", "clip_search", "shot_search"}
        ]
        if visual_indexes:
            for offset in range(0, len(visual_indexes), _VISUAL_BATCH_SIZE):
                chunk_indexes = visual_indexes[offset:offset + _VISUAL_BATCH_SIZE]
                requests = [
                    VisualSearchRequest(
                        retriever=calls[index].retriever,
                        query=calls[index].query,
                        top_k=calls[index].top_k,
                        event_id=calls[index].event_id,
                        tool_call_id=calls[index].call_id,
                        video_ids=calls[index].video_ids or None,
                        start_ms=calls[index].start_ms,
                        end_ms=calls[index].end_ms,
                    )
                    for index in chunk_indexes
                ]
                timeout = max(TOOL_TIMEOUTS[calls[index].retriever] for index in chunk_indexes)
                try:
                    chunk_results = await asyncio.wait_for(
                        visual_search_many(requests),
                        timeout=timeout,
                    )
                    if len(chunk_results) != len(chunk_indexes):
                        raise RuntimeError("visual batch returned an incomplete result set")
                except Exception:
                    logger.exception("Native visual batch chunk failed; isolating calls")
                    chunk_results = await asyncio.gather(
                        *(self.search(calls[index]) for index in chunk_indexes),
                        return_exceptions=True,
                    )
                for index, value in zip(chunk_indexes, chunk_results, strict=True):
                    if isinstance(value, Exception):
                        logger.warning(
                            "Visual retrieval call failed: call=%s error=%r",
                            calls[index].call_id,
                            value,
                        )
                        results[index] = []
                    else:
                        results[index] = self._filter_time_scope(calls[index], value)

        other_indexes = [index for index, value in enumerate(results) if value is None]
        if other_indexes:
            other_results = await asyncio.gather(
                *(self.search(calls[index]) for index in other_indexes),
                return_exceptions=True,
            )
            for index, value in zip(other_indexes, other_results, strict=True):
                if isinstance(value, Exception):
                    logger.warning(
                        "Retrieval call failed: call=%s error=%r",
                        calls[index].call_id,
                        value,
                    )
                    results[index] = []
                else:
                    results[index] = value
        return [hits or [] for hits in results]

    async def search(self, call: SearchCall) -> list[SearchHit]:
        common = {
            "top_k": call.top_k,
            "event_id": call.event_id,
            "tool_call_id": call.call_id,
        }
        temporal = {"start_ms": call.start_ms, "end_ms": call.end_ms}
        if call.retriever == "frame_search":
            coroutine = frame_search(
                query=call.query,
                video_ids=call.video_ids or None,
                **temporal,
                **common,
            )
        elif call.retriever == "clip_search":
            coroutine = clip_search(
                query=call.query,
                video_ids=call.video_ids or None,
                **temporal,
                **common,
            )
        elif call.retriever == "shot_search":
            coroutine = shot_search(
                query=call.query,
                video_ids=call.video_ids or None,
                **temporal,
                **common,
            )
        elif call.retriever == "ocr_search":
            # W7: OCR cascade — exact first, fuzzy fallback
            coroutine = self._ocr_cascade(call, common)
        elif call.retriever == "asr_search":
            coroutine = asr_search(
                query=call.query,
                mode="fuzzy",
                video_ids=call.video_ids or None,
                **temporal,
                **common,
            )
        elif call.retriever == "object_search":
            coroutine = object_search(
                object_class=call.query,
                video_ids=call.video_ids or None,
                min_count=call.min_count,
                **temporal,
                **common,
            )
        elif call.retriever == "track_search":
            coroutine = track_search(
                object_class=call.query,
                video_ids=call.video_ids or None,
                **temporal,
                **common,
            )
        else:
            raise ValueError(f"Unsupported V2 retriever: {call.retriever}")

        if call.retriever == "ocr_search":
            # _ocr_cascade already returns hits; skip double-await
            hits = await coroutine
        else:
            hits = await asyncio.wait_for(coroutine, timeout=TOOL_TIMEOUTS[call.retriever])

        # W9: Post-filter for retrievers that don't support native video_id filtering
        if call.video_ids and call.retriever in {
            "asr_search",
            "object_search",
            "track_search",
        }:
            allowed = set(call.video_ids)
            hits = [hit for hit in hits if hit.video_id in allowed]
        return self._filter_time_scope(call, hits)

    async def _ocr_cascade(
        self,
        call: SearchCall,
        common: dict,
    ) -> list[SearchHit]:
        """OCR cascade: exact → fuzzy (W7, pipeline.md §6C).

        Run exact first. If enough hits, return early.
        Otherwise, run fuzzy and merge results (exact takes priority).
        """
        timeout = TOOL_TIMEOUTS["ocr_search"]
        try:
            exact_hits = await asyncio.wait_for(
                ocr_search(
                    query=call.query,
                    mode="exact",
                    video_ids=call.video_ids or None,
                    start_ms=call.start_ms,
                    end_ms=call.end_ms,
                    **common,
                ),
                timeout=timeout,
            )
        except Exception:
            logger.warning("OCR exact search failed for call=%s, falling through to fuzzy", call.call_id)
            exact_hits = []

        # W9: Filter by video_ids if provided
        if call.video_ids:
            allowed = set(call.video_ids)
            exact_hits = [h for h in exact_hits if h.video_id in allowed]

        if len(exact_hits) >= _MIN_OCR_EXACT_HITS:
            return exact_hits

        try:
            fuzzy_hits = await asyncio.wait_for(
                ocr_search(
                    query=call.query,
                    mode="fuzzy",
                    video_ids=call.video_ids or None,
                    start_ms=call.start_ms,
                    end_ms=call.end_ms,
                    **common,
                ),
                timeout=timeout,
            )
        except Exception:
            logger.warning("OCR fuzzy search failed for call=%s", call.call_id)
            fuzzy_hits = []

        if call.video_ids:
            allowed = set(call.video_ids)
            fuzzy_hits = [h for h in fuzzy_hits if h.video_id in allowed]

        # Merge: exact hits take priority via entity_id dedup
        seen = {h.entity_id for h in exact_hits}
        merged = list(exact_hits)
        for h in fuzzy_hits:
            if h.entity_id not in seen:
                merged.append(h)
                seen.add(h.entity_id)
        return merged

    @staticmethod
    def _filter_time_scope(call: SearchCall, hits: list[SearchHit]) -> list[SearchHit]:
        if call.start_ms is None and call.end_ms is None:
            return hits
        return [
            hit
            for hit in hits
            if (call.end_ms is None or hit.start_ms <= call.end_ms)
            and (call.start_ms is None or hit.end_ms >= call.start_ms)
        ]


__all__ = ["ToolSearchGateway"]
