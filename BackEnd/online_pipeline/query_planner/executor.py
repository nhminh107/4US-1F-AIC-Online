from __future__ import annotations

import asyncio
import logging

from online_pipeline.query_planner.schemas import ToolCall
from online_pipeline.shared.config import TOOL_TIMEOUTS
from online_pipeline.shared.contracts import SearchHit


logger = logging.getLogger(__name__)


async def _dispatch_tool(call: ToolCall) -> list[SearchHit]:
    match call.tool:
        case "clip_search":
            from online_pipeline.retrieval_tools.visual import clip_search

            return await clip_search(**call.parameters)
        case "frame_search":
            from online_pipeline.retrieval_tools.visual import frame_search

            return await frame_search(**call.parameters)
        case "shot_search":
            from online_pipeline.retrieval_tools.visual import shot_search

            return await shot_search(**call.parameters)
        case "ocr_search":
            from online_pipeline.retrieval_tools.text import ocr_search

            return await ocr_search(**call.parameters)
        case "asr_search":
            from online_pipeline.retrieval_tools.text import asr_search

            return await asr_search(**call.parameters)
        case "caption_search":
            from online_pipeline.retrieval_tools.text import caption_search

            return await caption_search(**call.parameters)
        case _:
            raise ValueError(f"Unsupported retrieval tool: {call.tool}")


async def _execute_with_timeout(call: ToolCall) -> list[SearchHit]:
    return await asyncio.wait_for(
        _dispatch_tool(call),
        timeout=TOOL_TIMEOUTS[call.tool],
    )


def _with_event_id(hit: SearchHit, event_id: str | None) -> SearchHit:
    return hit.model_copy(update={"event_id": event_id})


async def execute_tool_calls(tool_calls: list[ToolCall]) -> list[SearchHit]:
    results = await asyncio.gather(
        *(_execute_with_timeout(call) for call in tool_calls),
        return_exceptions=True,
    )

    hits: list[SearchHit] = []
    for call, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            logger.warning(
                "Retrieval tool failed: tool=%s error=%r",
                call.tool,
                result,
            )
            continue

        if not isinstance(result, list):
            logger.warning(
                "Retrieval tool returned an invalid result: tool=%s result=%r",
                call.tool,
                result,
            )
            continue

        for raw_hit in result:
            try:
                hit = SearchHit.model_validate(raw_hit)
            except Exception as error:
                logger.warning(
                    "Retrieval tool returned invalid SearchHit: tool=%s error=%r",
                    call.tool,
                    error,
                )
                continue
            hits.append(_with_event_id(hit, call.event_id))

    return hits


__all__ = ["execute_tool_calls"]
