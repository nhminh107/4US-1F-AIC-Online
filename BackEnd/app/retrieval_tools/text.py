from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from BackEnd.app.contracts.models import SearchHit

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None  # type: ignore[assignment,misc]


_TEXT_FIELD = "text.keyword"
_VECTOR_FIELD = "embedding"

_es_client: Any | None = None
_text_embedder: Callable[[str], list[float]] | None = None


def configure_text_search(
    *,
    client: Any | None = None,
    embedder: Callable[[str], list[float]] | None = None,
) -> None:
    global _es_client, _text_embedder
    _es_client = client
    _text_embedder = embedder


def embed_text(query: str) -> list[float]:
    if _text_embedder is None:
        raise NotImplementedError("text embedder is not configured")
    return _text_embedder(query)


def _require_client() -> Any:
    global _es_client
    if _es_client is not None:
        return _es_client
    if AsyncElasticsearch is None:
        raise RuntimeError(
            "AsyncElasticsearch is not configured; install elasticsearch[async]"
        )

    elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
    _es_client = (
        AsyncElasticsearch(elasticsearch_url)
        if elasticsearch_url
        else AsyncElasticsearch()
    )
    return _es_client


def _build_query(
    query: str,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    if mode == "similarity":
        return {
            "knn": {
                "field": _VECTOR_FIELD,
                "query_vector": embed_text(query),
                "k": top_k,
                "num_candidates": max(top_k * 2, 10),
            },
            "size": top_k,
        }
    if mode == "exact":
        return {
            "query": {"term": {_TEXT_FIELD: query}},
            "size": top_k,
        }
    if mode == "fuzzy":
        return {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "fuzziness": "AUTO",
                    }
                }
            },
            "size": top_k,
        }
    raise ValueError(f"Unsupported text search mode: {mode}")


def _get_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _required_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    result = _get_value(value, field)
    if result is None:
        raise ValueError(f"Elasticsearch result has no {field!r}")
    return result


def _response_hits(response: Mapping[str, Any] | Any) -> list[Any]:
    hits = _required_value(response, "hits")
    return _required_value(hits, "hits")


async def _text_search(
    query: str,
    top_k: int,
    mode: str,
    event_id: str | None,
    tool_call_id: str | None,
    *,
    index_name: str,
    entity_type: str,
) -> list[SearchHit]:
    client = _require_client()
    request = _build_query(query, mode, top_k)
    response = await client.search(index=index_name, **request)

    results: list[SearchHit] = []
    for rank, hit in enumerate(_response_hits(response), start=1):
        source = _required_value(hit, "_source")
        results.append(
            SearchHit(
                source=index_name,
                entity_type=entity_type,
                entity_id=_required_value(hit, "_id"),
                video_id=_required_value(source, "video_id"),
                start_ms=_required_value(source, "start_ms"),
                end_ms=_required_value(source, "end_ms"),
                rank=rank,
                raw_score=_required_value(hit, "_score"),
                event_id=event_id,
                tool_call_id=tool_call_id,
            )
        )
    return results


async def ocr_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        tool_call_id,
        index_name="ocr_index",
        entity_type="ocr",
    )


async def asr_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        tool_call_id,
        index_name="asr_index",
        entity_type="asr",
    )


async def caption_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        tool_call_id,
        index_name="caption_index",
        entity_type="caption",
    )


__all__ = [
    "AsyncElasticsearch",
    "asr_search",
    "caption_search",
    "configure_text_search",
    "embed_text",
    "ocr_search",
]
