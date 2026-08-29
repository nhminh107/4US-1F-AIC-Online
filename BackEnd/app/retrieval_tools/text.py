from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from inspect import isawaitable
from typing import Any

from BackEnd.app.contracts.models import SearchHit

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None  # type: ignore[assignment,misc]


_TEXT_FIELD = "content"
_VECTOR_FIELD = "embedding"
OCR_INDEX = "aic_hcm2026_text_ocr_active"
ASR_INDEX = "aic_hcm2026_text_transcript_active"

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


async def close_text_search() -> None:
    """Close the shared Elasticsearch client when an application is shutting down."""

    global _es_client
    client = _es_client
    _es_client = None
    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if isawaitable(result):
            await result


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
    *,
    text_field: str = _TEXT_FIELD,
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if video_ids:
        filters.append({"terms": {"video_id": video_ids}})
    if end_ms is not None:
        filters.append({"range": {"start_ms": {"lte": end_ms}}})
    if start_ms is not None:
        filters.append({"range": {"end_ms": {"gte": start_ms}}})
    if mode == "similarity":
        request = {
            "knn": {
                "field": _VECTOR_FIELD,
                "query_vector": embed_text(query),
                "k": top_k,
                "num_candidates": max(top_k * 2, 10),
            },
            "size": top_k,
        }
        if filters:
            request["knn"]["filter"] = {"bool": {"filter": filters}}
        return request
    if mode in {"exact", "phrase"}:
        request = {
            "query": {"match_phrase": {text_field: query}},
            "size": top_k,
        }
    elif mode in {"fuzzy", "match"}:
        request = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {text_field: {"query": query, "boost": 3}}},
                        {"match": {text_field: {"query": query, "operator": "and", "boost": 1}}},
                        {"match": {text_field: {"query": query, "fuzziness": "AUTO", "boost": 0.5}}},
                        {"match": {f"{text_field}.shingle": {"query": query, "boost": 2}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": top_k,
        }
    else:
        raise ValueError(f"Unsupported text search mode: {mode}")
    if filters:
        request["query"] = {"bool": {"must": [request["query"]], "filter": filters}}
    return request


def _get_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    body = getattr(value, "body", None)
    if isinstance(body, Mapping):
        return body.get(field)
    return getattr(value, field, None)


def _required_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    result = _get_value(value, field)
    if result is None:
        raise ValueError(f"Elasticsearch result has no {field!r}")
    return result


def _response_hits(response: Mapping[str, Any] | Any) -> list[Any]:
    hits = _required_value(response, "hits")
    return _required_value(hits, "hits")


def _source_time_range(source: Mapping[str, Any] | Any) -> tuple[int, int]:
    timestamp_ms = _get_value(source, "timestamp_ms")
    start_ms = _get_value(source, "start_ms")
    end_ms = _get_value(source, "end_ms")
    resolved_start_ms = start_ms if start_ms is not None else timestamp_ms
    resolved_end_ms = end_ms if end_ms is not None else timestamp_ms
    if resolved_start_ms is None or resolved_end_ms is None:
        return 0, 0
    return int(resolved_start_ms), int(resolved_end_ms)


async def _text_search(
    query: str,
    top_k: int,
    mode: str,
    event_id: str | None,
    tool_call_id: str | None,
    *,
    index_name: str,
    entity_type: str,
    text_field: str = _TEXT_FIELD,
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[SearchHit]:
    client = _require_client()
    request = _build_query(
        query,
        mode,
        top_k,
        text_field=text_field,
        video_ids=video_ids,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    response = await client.search(index=index_name, **request)

    results: list[SearchHit] = []
    for rank, hit in enumerate(_response_hits(response), start=1):
        source = _required_value(hit, "_source")
        start_ms, end_ms = _source_time_range(source)
        results.append(
            SearchHit(
                source=index_name,
                entity_type=entity_type,
                entity_id=_get_value(source, "entity_id") or _required_value(hit, "_id"),
                video_id=_required_value(source, "video_id"),
                start_ms=start_ms,
                end_ms=end_ms,
                shot_id=_get_value(source, "shot_id"),
                clip_id=_get_value(source, "clip_id"),
                frame_id=_get_value(source, "frame_id"),
                rank=rank,
                raw_score=_required_value(hit, "_score"),
                text_content=_get_value(source, text_field),
                event_id=event_id,
                tool_call_id=tool_call_id,
            )
        )
    return results


async def ocr_search(
    query: str,
    top_k: int = 100,
    mode: str = "fuzzy",
    event_id: str | None = None,
    tool_call_id: str | None = None,
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        tool_call_id,
        index_name=OCR_INDEX,
        entity_type="ocr",
        text_field="content",
        video_ids=video_ids,
        start_ms=start_ms,
        end_ms=end_ms,
    )


async def asr_search(
    query: str,
    top_k: int = 100,
    mode: str = "fuzzy",
    event_id: str | None = None,
    tool_call_id: str | None = None,
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        tool_call_id,
        index_name=ASR_INDEX,
        entity_type="asr",
        text_field="content",
        video_ids=video_ids,
        start_ms=start_ms,
        end_ms=end_ms,
    )


# caption_search removed — no "caption_index" exists in Elasticsearch.
# All caption data lives in the unified aic_hcm2026_text_v1 index.


__all__ = [
    "AsyncElasticsearch",
    "ASR_INDEX",
    "asr_search",
    "close_text_search",
    "configure_text_search",
    "embed_text",
    "ocr_search",
]
