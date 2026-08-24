"""Direct Elasticsearch text search utility for testing and smoke verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

SOURCE_INDEX_MAP = {
    "ocr": "aic_hcm2026_text_ocr_active",
    "transcript": "aic_hcm2026_text_transcript_active",
    "asr": "aic_hcm2026_text_transcript_active",
    "object": "aic_hcm2026_text_object_active",
    "metadata": "aic_hcm2026_text_metadata_active",
}


def search_es(
    query_text: str,
    *,
    source: str = "ocr",
    top_k: int = 5,
    elasticsearch_url: str | None = None,
) -> list[dict[str, Any]]:
    url = elasticsearch_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    client = Elasticsearch(url, request_timeout=30)
    index_name = SOURCE_INDEX_MAP.get(source.lower(), f"aic_hcm2026_text_{source}_active")

    body = {
        "size": top_k,
        "query": {
            "match": {
                "content": query_text,
            }
        },
    }

    try:
        response = client.search(index=index_name, body=body)
    except Exception:
        # Fallback to wildcard pattern if specific alias is missing
        body["query"] = {
            "bool": {
                "must": [{"match": {"content": query_text}}],
                "filter": [{"term": {"source_type": source}}],
            }
        }
        response = client.search(index="aic_hcm2026_text_*", body=body)

    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.get("_source", {})
        results.append({
            "id": hit.get("_id"),
            "score": hit.get("_score"),
            "video_id": src.get("video_id"),
            "start_ms": src.get("start_ms"),
            "end_ms": src.get("end_ms"),
            "content": src.get("content"),
            "source_type": src.get("source_type"),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Elasticsearch text indices directly.")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--source", type=str, default="ocr", help="Source type (ocr, transcript, object, etc.)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--elasticsearch-url", type=str, default=None, help="Elasticsearch URL override")

    args = parser.parse_args()
    results = search_es(
        args.query,
        source=args.source,
        top_k=args.top_k,
        elasticsearch_url=args.elasticsearch_url,
    )
    print(f"Found {len(results)} hits for query '{args.query}' in source '{args.source}':")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
