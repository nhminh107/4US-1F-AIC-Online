from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from online_pipeline.shared.contracts import SearchHit

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None  # type: ignore[assignment,misc]


es_client = AsyncElasticsearch() if AsyncElasticsearch is not None else None

_TEXT_FIELD = "text.keyword"
_VECTOR_FIELD = "embedding"


def embed_text(query: str) -> list[float]:
    """Placeholder for the shared text embedding model."""
    raise NotImplementedError("embed_text is not configured")


def _require_client() -> Any:
    if es_client is None:
        raise RuntimeError(
            "AsyncElasticsearch is not configured; install elasticsearch[async]"
        )
    return es_client


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
            )
        )
    return results


async def ocr_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        index_name="ocr_index",
        entity_type="ocr",
    )


async def asr_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        index_name="asr_index",
        entity_type="asr",
    )


async def caption_search(
    query: str,
    top_k: int = 100,
    mode: str = "similarity",
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _text_search(
        query,
        top_k,
        mode,
        event_id,
        index_name="caption_index",
        entity_type="caption",
    )


__all__ = [
    "AsyncElasticsearch",
    "asr_search",
    "caption_search",
    "embed_text",
    "es_client",
    "ocr_search",
]
