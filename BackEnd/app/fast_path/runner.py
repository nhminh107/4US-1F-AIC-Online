from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Mapping

from BackEnd.CONFIG import TOOL_TIMEOUTS, TOP_K_DEFAULTS
from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_tools.text import asr_search, ocr_search
from BackEnd.app.retrieval_tools.visual import clip_search, frame_search, shot_search


logger = logging.getLogger(__name__)


def _planned_calls(
    structured_query: StructuredQuery,
    top_k: Mapping[str, int] | None = None,
) -> list[tuple[str, Awaitable[list[SearchHit]]]]:
    resolved_top_k = {**TOP_K_DEFAULTS, **dict(top_k or {})}
    calls: list[tuple[str, Awaitable[list[SearchHit]]]] = []
    visual_queries = [query for query in structured_query.visual_queries if query]
    ocr_constraints = [query for query in structured_query.ocr_constraints if query]
    asr_constraints = [query for query in structured_query.asr_constraints if query]

    if visual_queries:
        for query in visual_queries:
            calls.append(
                ("clip_search", clip_search(query, top_k=resolved_top_k["clip_search"]))
            )
            calls.append(
                ("frame_search", frame_search(query, top_k=resolved_top_k["frame_search"]))
            )
            calls.append(
                ("shot_search", shot_search(query, top_k=resolved_top_k["shot_search"]))
            )
    elif not ocr_constraints and not asr_constraints:
        primary_query = structured_query.question or " ".join(
            event.description for event in structured_query.events
        )
        if primary_query:
            calls.append(
                (
                    "clip_search",
                    clip_search(
                        primary_query,
                        top_k=resolved_top_k["clip_search"],
                    ),
                )
            )
            calls.append(
                (
                    "frame_search",
                    frame_search(
                        primary_query,
                        top_k=resolved_top_k["frame_search"],
                    ),
                )
            )
            calls.append(
                (
                    "shot_search",
                    shot_search(
                        primary_query,
                        top_k=resolved_top_k["shot_search"],
                    ),
                )
            )

    calls.extend(
        (
            "ocr_search",
            ocr_search(query, top_k=resolved_top_k["ocr_search"]),
        )
        for query in ocr_constraints
    )
    calls.extend(
        (
            "asr_search",
            asr_search(query, top_k=resolved_top_k["asr_search"]),
        )
        for query in asr_constraints
    )
    return calls


async def _run_with_timeout(
    tool_name: str,
    tool_coroutine: Awaitable[list[SearchHit]],
) -> list[SearchHit]:
    return await asyncio.wait_for(
        tool_coroutine,
        timeout=TOOL_TIMEOUTS[tool_name],
    )


async def run_fast_path(
    structured_query: StructuredQuery,
    *,
    top_k: Mapping[str, int] | None = None,
) -> list[SearchHit]:
    planned_calls = _planned_calls(structured_query, top_k)
    if not planned_calls:
        return []

    results = await asyncio.gather(
        *(
            _run_with_timeout(tool_name, tool_coroutine)
            for tool_name, tool_coroutine in planned_calls
        ),
        return_exceptions=True,
    )

    hits: list[SearchHit] = []
    for (tool_name, _), result in zip(planned_calls, results):
        if isinstance(result, Exception):
            logger.warning(
                "Fast Path tool failed: tool=%s error=%r",
                tool_name,
                result,
            )
            continue

        for raw_hit in result:
            try:
                hits.append(SearchHit.model_validate(raw_hit))
            except Exception as error:
                logger.warning(
                    "Fast Path tool returned invalid SearchHit: tool=%s error=%r",
                    tool_name,
                    error,
                )

    return hits


__all__ = ["run_fast_path"]
