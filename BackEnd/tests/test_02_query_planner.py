"""Interactive end-to-end smoke test: intent extraction, planning, and execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from BackEnd.app.contracts.models import RawQuery, StructuredQuery
from BackEnd.app.intent_extractor.extractor import extract_intent_sync
from BackEnd.app.query_planner import execute_tool_calls, run_query_planner
from BackEnd.app.retrieval_tools.text import close_text_search


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Intent Extractor, Query Planner, and Tool Executor against configured backends."
    )
    parser.add_argument("prompt", nargs="?", help="User prompt to process.")
    parser.add_argument(
        "--feedback",
        help="Optional correction or additional constraint for the prompt.",
    )
    parser.add_argument(
        "--session-id",
        default="manual-query-planner-test",
        help="Session identifier included in RawQuery.",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default="http://127.0.0.1:9200",
        help="Elasticsearch URL used by retrieval tools.",
    )
    return parser.parse_args()


def _print_json_list(title: str, items: list[object]) -> None:
    print(f"\n=== {title} ===")
    print(f"Total: {len(items)}")
    for index, item in enumerate(items, start=1):
        print(f"\n[{index}]")
        print(item.model_dump_json(indent=2))


async def _plan_and_execute(structured_query: StructuredQuery) -> None:
    try:
        tool_calls = await run_query_planner(structured_query)
        _print_json_list("Query Planner ToolCall[]", tool_calls)

        hits = await execute_tool_calls(tool_calls)
        _print_json_list("Tool Executor SearchHit[]", hits)
    finally:
        await close_text_search()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    os.environ["ELASTICSEARCH_URL"] = args.elasticsearch_url

    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        raise SystemExit("Prompt must not be empty.")

    feedback = args.feedback
    if feedback is None and args.prompt is None:
        feedback = input("Feedback (optional, press Enter to skip): ").strip() or None

    structured_query = extract_intent_sync(
        RawQuery(
            text=prompt,
            feedback=feedback,
            session_id=args.session_id,
        )
    )
    print("=== StructuredQuery ===")
    print(structured_query.model_dump_json(indent=2))
    asyncio.run(_plan_and_execute(structured_query))


if __name__ == "__main__":
    main()
