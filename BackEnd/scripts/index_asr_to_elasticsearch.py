"""Index PostgreSQL transcript segments into the configured Elasticsearch ASR alias.

The operation is idempotent: each document uses ``asr:<segment_id>`` as its
Elasticsearch ID, so rerunning the script updates existing segments instead of
creating duplicates.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from typing import Any

from elasticsearch import Elasticsearch, helpers
from sqlalchemy import func, select

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Database.sql_models import TranscriptSegment
from BackEnd.app.retrieval_tools.text import ASR_INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index TranscriptSegment records from PostgreSQL into Elasticsearch."
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200"),
        help="Elasticsearch endpoint (default: ELASTICSEARCH_URL or local port 9200).",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("ASR_ELASTICSEARCH_INDEX", ASR_INDEX),
        help="Existing concrete index or single-target ASR alias.",
    )
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count source records without writing documents to Elasticsearch.",
    )
    return parser.parse_args()


def transcript_actions(
    manager: PostgreManager,
    *,
    index_name: str,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    statement = select(TranscriptSegment).order_by(TranscriptSegment.segment_id)
    with manager.session_factory() as session:
        segments = session.scalars(statement).yield_per(batch_size)
        for segment in segments:
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": f"asr:{segment.segment_id}",
                "_source": {
                    "doc_id": f"asr:{segment.segment_id}",
                    "entity_id": segment.segment_id,
                    "segment_id": segment.segment_id,
                    "video_id": segment.video_id,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "timestamp_ms": segment.start_ms,
                    "content": segment.text,
                    "language": segment.language,
                    "source_type": "transcript",
                },
            }


def validate_target(client: Elasticsearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        return
    if client.indices.exists_alias(name=index_name):
        aliases = client.indices.get_alias(name=index_name)
        if len(aliases) == 1:
            return
        raise RuntimeError(
            f"Alias {index_name!r} points to multiple indexes; select a concrete index "
            "with --index to avoid writing to an ambiguous target."
        )
    raise RuntimeError(
        f"Elasticsearch index or alias {index_name!r} does not exist. "
        "Create it with the expected transcript mapping before indexing."
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    manager = PostgreManager()
    with Elasticsearch(args.elasticsearch_url) as client:
        validate_target(client, args.index)
        if args.dry_run:
            with manager.session_factory() as session:
                total = session.scalar(select(func.count()).select_from(TranscriptSegment))
            print(f"Dry run: {total or 0} transcript segments would be indexed.")
            return

        succeeded = 0
        failed = 0
        for ok, result in helpers.streaming_bulk(
            client,
            transcript_actions(
                manager, index_name=args.index, batch_size=args.batch_size
            ),
            chunk_size=args.batch_size,
            raise_on_error=False,
            raise_on_exception=False,
        ):
            if ok:
                succeeded += 1
            else:
                failed += 1
                print(f"Indexing failed: {result}")

        client.indices.refresh(index=args.index)
        print(
            f"ASR indexing completed: indexed_or_updated={succeeded}, failed={failed}, "
            f"target={args.index}"
        )
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
