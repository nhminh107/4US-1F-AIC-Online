from __future__ import annotations

import asyncio
import logging
from typing import Any

from online_pipeline.intent_extractor.schemas import StructuredQuery
from online_pipeline.retrieval_tools.text import asr_search, ocr_search
from online_pipeline.retrieval_tools.visual import clip_search
from online_pipeline.shared.config import TOOL_TIMEOUTS
from online_pipeline.shared.contracts import SearchHit


logger = logging.getLogger(__name__)


def _planned_calls(
    structured_query: StructuredQuery,
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    visual_queries = [
        query for query in getattr(structured_query, "visual_queries", []) if query
    ]
    ocr_constraints = [
        query for query in getattr(structured_query, "ocr_constraints", []) if query
    ]
    asr_constraints = [
        query for query in getattr(structured_query, "asr_constraints", []) if query
    ]

    if visual_queries:
        calls.extend(("clip_search", clip_search(query)) for query in visual_queries)
    elif not ocr_constraints and not asr_constraints:
        primary_query = getattr(structured_query, "question", "")
        calls.append(("clip_search", clip_search(primary_query)))

    calls.extend(("ocr_search", ocr_search(query)) for query in ocr_constraints)
    calls.extend(("asr_search", asr_search(query)) for query in asr_constraints)
    return calls


async def _run_with_timeout(tool_name: str, tool_coroutine: Any) -> list[SearchHit]:
    return await asyncio.wait_for(
        tool_coroutine,
        timeout=TOOL_TIMEOUTS[tool_name],
    )


async def run_fast_path(structured_query: StructuredQuery) -> list[SearchHit]:
    planned_calls = _planned_calls(structured_query)
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
