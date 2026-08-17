from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from online_pipeline.shared.contracts import SearchHit

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None  # type: ignore[assignment,misc]

import logging


logger = logging.getLogger(__name__)
es_client = AsyncElasticsearch() if AsyncElasticsearch is not None else None


def _get_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _required_value(value: Mapping[str, Any] | Any, field: str) -> Any:
    result = _get_value(value, field)
    if result is None:
        raise ValueError(f"Elasticsearch result has no {field!r}")
    return result


def _object_query(object_class: str) -> dict[str, Any]:
    return {
        "bool": {
            "filter": [
                {"term": {"object_class": object_class}},
            ]
        }
    }


def _object_aggregations(min_count: int, top_k: int) -> dict[str, Any]:
    return {
        "by_video": {
            "terms": {"field": "video_id", "size": top_k},
            "aggs": {
                "by_shot": {
                    "terms": {"field": "shot_id", "size": top_k},
                    "aggs": {
                        "representative": {"top_hits": {"size": 1}},
                        "minimum_count": {
                            "bucket_selector": {
                                "buckets_path": {"count": "_count"},
                                "script": f"params.count >= {min_count}",
                            }
                        },
                    },
                }
            },
        }
    }


def _track_query(object_class: str, relation: str) -> dict[str, Any]:
    return {
        "bool": {
            "filter": [
                {"term": {"object_class": object_class}},
                {"term": {"relation": relation}},
            ]
        }
    }


def _top_level_hits(response: Mapping[str, Any] | Any) -> list[Any]:
    hits = _get_value(response, "hits")
    if hits is None:
        return []
    return _get_value(hits, "hits") or []


def _aggregation_hits(value: Mapping[str, Any] | Any) -> list[Any]:
    if isinstance(value, list):
        results: list[Any] = []
        for child in value:
            results.extend(_aggregation_hits(child))
        return results
    if not isinstance(value, Mapping):
        return []

    hits = value.get("hits")
    if isinstance(hits, Mapping) and isinstance(hits.get("hits"), list):
        return list(hits["hits"])

    results: list[Any] = []
    for child in value.values():
        results.extend(_aggregation_hits(child))
    return results


def _response_hits(response: Mapping[str, Any] | Any) -> list[Any]:
    direct_hits = _top_level_hits(response)
    if direct_hits:
        return direct_hits
    aggregations = _get_value(response, "aggregations")
    return _aggregation_hits(aggregations)


def _to_search_hits(
    response: Mapping[str, Any] | Any,
    *,
    source: str,
    entity_type: str,
    event_id: str | None,
) -> list[SearchHit]:
    results: list[SearchHit] = []
    for rank, hit in enumerate(_response_hits(response), start=1):
        document = _required_value(hit, "_source")
        score = _get_value(hit, "_score")
        results.append(
            SearchHit(
                source=source,
                entity_type=entity_type,
                entity_id=_required_value(hit, "_id"),
                video_id=_required_value(document, "video_id"),
                start_ms=_required_value(document, "start_ms"),
                end_ms=_required_value(document, "end_ms"),
                rank=rank,
                raw_score=0.0 if score is None else score,
                event_id=event_id,
            )
        )
    return results


async def _search(
    *,
    index: str,
    query: dict[str, Any],
    source: str,
    entity_type: str,
    event_id: str | None,
    size: int,
    aggregations: dict[str, Any] | None = None,
) -> list[SearchHit]:
    if es_client is None:
        logger.info("object index not available")
        return []

    try:
        request: dict[str, Any] = {
            "index": index,
            "query": query,
            "size": size,
        }
        if aggregations is not None:
            request["aggs"] = aggregations
        response = await es_client.search(**request)
        return _to_search_hits(
            response,
            source=source,
            entity_type=entity_type,
            event_id=event_id,
        )
    except Exception:
        logger.info("object index not available")
        return []


async def object_search(
    object_class: str,
    top_k: int = 100,
    min_count: int = 1,
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _search(
        index="object_detection_index",
        query=_object_query(object_class),
        aggregations=_object_aggregations(min_count, top_k),
        size=0,
        source="object_detection",
        entity_type="object",
        event_id=event_id,
    )


async def track_search(
    object_class: str,
    top_k: int = 100,
    relation: str = "continuous_track",
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _search(
        index="tracking_index",
        query=_track_query(object_class, relation),
        size=top_k,
        source="tracking",
        entity_type="track",
        event_id=event_id,
    )


__all__ = ["AsyncElasticsearch", "es_client", "object_search", "track_search"]
