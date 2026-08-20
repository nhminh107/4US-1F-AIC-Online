from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any

from BackEnd.app.contracts.models import SearchHit

if TYPE_CHECKING:
    from BackEnd.app.retrieval.visual_retrieval import VisualRetrievalTools


_visual_tools: Any | None = None


def configure_visual_retrieval_tools(tools: Any | None) -> None:
    global _visual_tools
    _visual_tools = tools


def get_visual_retrieval_tools() -> Any:
    global _visual_tools
    if _visual_tools is None:
        from BackEnd.app.retrieval.visual_retrieval import (
            build_default_visual_retrieval_tools,
        )

        _visual_tools = build_default_visual_retrieval_tools()
    return _visual_tools


async def warmup_visual_retrieval_tools() -> None:
    """Load the visual backend before a retrieval timeout is applied."""

    await asyncio.to_thread(get_visual_retrieval_tools)


async def _run_visual_tool(method_name: str, **kwargs) -> list[SearchHit]:
    tools = get_visual_retrieval_tools()
    method = getattr(tools, method_name)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(method, **kwargs))


async def clip_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _run_visual_tool(
        "clip_search",
        query=query,
        top_k=top_k,
        event_id=event_id,
        tool_call_id=tool_call_id,
    )


async def frame_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _run_visual_tool(
        "frame_search",
        query=query,
        top_k=top_k,
        event_id=event_id,
        tool_call_id=tool_call_id,
    )


async def shot_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _run_visual_tool(
        "shot_search",
        query=query,
        top_k=top_k,
        event_id=event_id,
        tool_call_id=tool_call_id,
    )


__all__ = [
    "clip_search",
    "configure_visual_retrieval_tools",
    "frame_search",
    "get_visual_retrieval_tools",
    "shot_search",
    "warmup_visual_retrieval_tools",
]
