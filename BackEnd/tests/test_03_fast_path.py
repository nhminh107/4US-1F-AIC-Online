"""Interactive end-to-end smoke test: Intent Extractor -> Fast Path retrieval."""

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
from BackEnd.app.fast_path.runner import run_fast_path
from BackEnd.app.intent_extractor.extractor import extract_intent_sync
from BackEnd.app.retrieval_tools.text import close_text_search


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Intent Extractor and Fast Path against configured FAISS and Elasticsearch backends."
    )
    parser.add_argument("prompt", nargs="?", help="User prompt to process.")
    parser.add_argument(
        "--feedback",
        help="Optional correction or additional constraint for the prompt.",
    )
    parser.add_argument(
        "--session-id",
        default="manual-fast-path-test",
        help="Session identifier included in RawQuery.",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default="http://127.0.0.1:9200",
        help="Elasticsearch URL used by OCR retrieval.",
    )
    return parser.parse_args()


async def _run_fast_path(structured_query: StructuredQuery) -> None:
    try:
        hits = await run_fast_path(structured_query)
    finally:
        await close_text_search()
    print("\n=== Fast Path SearchHit[] ===")
    print(f"Total hits: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}]")
        print(hit.model_dump_json(indent=2))


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
    asyncio.run(_run_fast_path(structured_query))


if __name__ == "__main__":
    main()
