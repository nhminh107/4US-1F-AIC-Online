"""Interactive smoke test for the Intent Extractor using the configured LLM."""

from __future__ import annotations

import argparse
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from BackEnd.app.contracts.models import RawQuery
from BackEnd.app.intent_extractor.extractor import extract_intent_sync


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one prompt to the Intent Extractor and print StructuredQuery JSON."
    )
    parser.add_argument("prompt", nargs="?", help="User prompt to parse.")
    parser.add_argument(
        "--feedback",
        help="Optional correction or additional constraint for the prompt.",
    )
    parser.add_argument(
        "--session-id",
        default="manual-intent-extractor-test",
        help="Session identifier included in RawQuery.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        raise SystemExit("Prompt must not be empty.")

    feedback = args.feedback
    if feedback is None and args.prompt is None:
        feedback = input("Feedback (optional, press Enter to skip): ").strip() or None

    result = extract_intent_sync(
        RawQuery(
            text=prompt,
            feedback=feedback,
            session_id=args.session_id,
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
