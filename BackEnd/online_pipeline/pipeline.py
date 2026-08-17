from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from online_pipeline.fast_path.runner import run_fast_path
from online_pipeline.intent_extractor.extractor import extract_intent
from online_pipeline.query_planner.executor import execute_tool_calls
from online_pipeline.query_planner.planner import run_query_planner
from online_pipeline.shared.contracts import SearchHit


logger = logging.getLogger(__name__)


@dataclass
class RawQuery:
    text: str
    image_ref: str | None = None
    feedback: str | None = None
    session_id: str = ""


async def run_pipeline(raw_query: RawQuery) -> list[SearchHit]:
    logger.info("Pipeline step 1: extracting intent")
    structured_query = await extract_intent(raw_query)

    logger.info("Pipeline step 2: running Fast Path and Query Planner")
    fast_hits, tool_calls = await asyncio.gather(
        run_fast_path(structured_query),
        run_query_planner(structured_query),
    )
    logger.info(
        "Pipeline step 2 complete: fast_path_hits=%d planner_tool_calls=%d",
        len(fast_hits),
        len(tool_calls),
    )

    logger.info("Pipeline step 3: executing planner tool calls")
    agent_hits = await execute_tool_calls(tool_calls)
    logger.info(
        "Pipeline step 3 complete: agent_hits=%d",
        len(agent_hits),
    )

    merged_hits = fast_hits + agent_hits
    logger.info(
        "Pipeline step 4 complete: total_search_hits=%d",
        len(merged_hits),
    )
    return merged_hits


__all__ = ["RawQuery", "run_pipeline"]
