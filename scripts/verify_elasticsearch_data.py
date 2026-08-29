"""Verify Elasticsearch text index source coverage for runtime deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from elasticsearch import Elasticsearch

TEXT_SOURCE_TYPES = ("video_metadata", "ocr", "transcript", "object")


def verify_elasticsearch_data(
    *,
    index: str,
    elasticsearch_url: str | None = None,
) -> dict[str, Any]:
    url = elasticsearch_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    client = Elasticsearch(url, request_timeout=30)
    total = int(client.count(index=index, body={"query": {"match_all": {}}})["count"])
    source_counts = {
        source_type: int(
            client.count(
                index=index,
                body={"query": {"term": {"source_type": source_type}}},
            )["count"]
        )
        for source_type in sorted(TEXT_SOURCE_TYPES)
    }
    missing_sources = [
        source_type
        for source_type, count in source_counts.items()
        if count <= 0
    ]
    return {
        "ok": total > 0 and not missing_sources,
        "index": index,
        "total": total,
        "source_counts": source_counts,
        "missing_sources": missing_sources,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Elasticsearch text index data after Postgres sync."
    )
    parser.add_argument(
        "--index",
        default="aic_hcm2026_text_*",
        help="Physical index or alias pattern to verify.",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=None,
        help="Elasticsearch URL override. Defaults to ELASTICSEARCH_URL from .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = verify_elasticsearch_data(
        index=args.index,
        elasticsearch_url=args.elasticsearch_url,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

