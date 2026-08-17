from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from online_pipeline.shared.contracts import SearchHit


def embed_text(query: str) -> list[float]:
    """Placeholder for the shared embedding model."""
    raise NotImplementedError("embed_text is not configured")


def search_faiss(
    index_name: str,
    vector: list[float],
    top_k: int,
) -> list[tuple[str, float]]:
    """Placeholder for the synchronous FAISS search boundary."""
    raise NotImplementedError("search_faiss is not configured")


async def resolve_entity(faiss_id: str, index_name: str) -> Mapping[str, Any]:
    """Placeholder for canonical ID resolution in the Shared Data Layer."""
    raise NotImplementedError("resolve_entity is not configured")


def _get_value(entity: Mapping[str, Any] | Any, field: str) -> Any:
    if isinstance(entity, Mapping):
        return entity.get(field)
    return getattr(entity, field, None)


def _required_value(entity: Mapping[str, Any] | Any, field: str) -> Any:
    value = _get_value(entity, field)
    if value is None:
        raise ValueError(f"resolve_entity returned no {field!r}")
    return value


def _unpack_faiss_result(result: Any) -> tuple[str, float]:
    if isinstance(result, Mapping):
        faiss_id = result.get("faiss_id", result.get("id"))
        raw_score = result.get("raw_score", result.get("score"))
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) < 2:
            raise ValueError("FAISS result must contain an id and a score")
        faiss_id, raw_score = result[0], result[1]
    else:
        faiss_id = getattr(result, "faiss_id", getattr(result, "id", None))
        raw_score = getattr(result, "raw_score", getattr(result, "score", None))

    if faiss_id is None or raw_score is None:
        raise ValueError("FAISS result must contain faiss_id and raw_score")
    return str(faiss_id), float(raw_score)


async def _visual_search(
    query: str,
    top_k: int,
    event_id: str | None,
    *,
    index_name: str,
    entity_type: str,
) -> list[SearchHit]:
    vector = embed_text(query)
    loop = asyncio.get_event_loop()
    faiss_results = await loop.run_in_executor(
        None,
        search_faiss,
        index_name,
        vector,
        top_k,
    )

    hits: list[SearchHit] = []
    for rank, result in enumerate(faiss_results, start=1):
        faiss_id, raw_score = _unpack_faiss_result(result)
        entity = await resolve_entity(faiss_id, index_name)
        hits.append(
            SearchHit(
                source=index_name,
                entity_type=entity_type,
                entity_id=_required_value(entity, "entity_id"),
                video_id=_required_value(entity, "video_id"),
                start_ms=_required_value(entity, "start_ms"),
                end_ms=_required_value(entity, "end_ms"),
                rank=rank,
                raw_score=raw_score,
                event_id=event_id,
            )
        )
    return hits


async def clip_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _visual_search(
        query,
        top_k,
        event_id,
        index_name="clip_embedding",
        entity_type="clip",
    )


async def frame_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _visual_search(
        query,
        top_k,
        event_id,
        index_name="frame_embedding",
        entity_type="frame",
    )


async def shot_search(
    query: str,
    top_k: int = 200,
    event_id: str | None = None,
) -> list[SearchHit]:
    return await _visual_search(
        query,
        top_k,
        event_id,
        index_name="shot_embedding",
        entity_type="shot",
    )


__all__ = [
    "clip_search",
    "embed_text",
    "frame_search",
    "resolve_entity",
    "search_faiss",
    "shot_search",
]
