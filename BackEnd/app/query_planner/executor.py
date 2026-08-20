from __future__ import annotations

import asyncio
import logging

from BackEnd.CONFIG import TOOL_TIMEOUTS
from BackEnd.app.contracts.models import SearchHit, ToolCall


logger = logging.getLogger(__name__)


async def _dispatch_tool(call: ToolCall) -> list[SearchHit]:
    parameters = {
        **call.parameters,
        "event_id": call.event_id,
        "tool_call_id": call.tool_call_id,
    }

    match call.tool_name:
        case "clip_search":
            from BackEnd.app.retrieval_tools.visual import clip_search

            return await clip_search(**parameters)
        case "frame_search":
            from BackEnd.app.retrieval_tools.visual import frame_search

            return await frame_search(**parameters)
        case "shot_search":
            from BackEnd.app.retrieval_tools.visual import shot_search

            return await shot_search(**parameters)
        case "ocr_search":
            from BackEnd.app.retrieval_tools.text import ocr_search

            return await ocr_search(**parameters)
        case "asr_search":
            from BackEnd.app.retrieval_tools.text import asr_search

            return await asr_search(**parameters)
        case "caption_search":
            from BackEnd.app.retrieval_tools.text import caption_search

            return await caption_search(**parameters)
        case "object_search":
            from BackEnd.app.retrieval_tools.object import object_search

            return await object_search(**parameters)
        case "track_search":
            from BackEnd.app.retrieval_tools.object import track_search

            return await track_search(**parameters)
        case _:
            raise ValueError(f"Unsupported retrieval tool: {call.tool_name}")


async def _execute_with_timeout(call: ToolCall) -> list[SearchHit]:
    return await asyncio.wait_for(
        _dispatch_tool(call),
        timeout=TOOL_TIMEOUTS[call.tool_name],
    )


def _with_event_and_tool_call(hit: SearchHit, call: ToolCall) -> SearchHit:
    return hit.model_copy(
        update={
            "event_id": call.event_id,
            "tool_call_id": call.tool_call_id,
        }
    )


async def execute_tool_calls(tool_calls: list[ToolCall]) -> list[SearchHit]:
    if not tool_calls:
        return []

    results = await asyncio.gather(
        *(_execute_with_timeout(call) for call in tool_calls),
        return_exceptions=True,
    )

    hits: list[SearchHit] = []
    for call, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            logger.warning(
                "Retrieval tool failed: tool=%s error=%r",
                call.tool_name,
                result,
            )
            continue

        if not isinstance(result, list):
            logger.warning(
                "Retrieval tool returned an invalid result: tool=%s result=%r",
                call.tool_name,
                result,
            )
            continue

        for raw_hit in result:
            try:
                hit = SearchHit.model_validate(raw_hit)
            except Exception as error:
                logger.warning(
                    "Retrieval tool returned invalid SearchHit: tool=%s error=%r",
                    call.tool_name,
                    error,
                )
                continue
            hits.append(_with_event_and_tool_call(hit, call))

    return hits


__all__ = ["execute_tool_calls"]
