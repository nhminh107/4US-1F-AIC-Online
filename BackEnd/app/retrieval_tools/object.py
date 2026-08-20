from __future__ import annotations

from collections import defaultdict
from typing import Any

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.contracts.models import SearchHit


_db_manager: Any | None = None


def configure_object_search_manager(manager: Any | None) -> None:
    """Inject a PostgreSQL manager for tests or application startup."""

    global _db_manager
    _db_manager = manager


def _get_manager() -> Any:
    global _db_manager
    if _db_manager is None:
        _db_manager = PostgreManager()
    return _db_manager


def _object_hits_from_rows(
    rows: list[tuple[Any, Any]],
    *,
    min_count: int,
    top_k: int,
    event_id: str | None,
    tool_call_id: str | None,
) -> list[SearchHit]:
    groups: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    for detection, frame in rows:
        # Official frames can have no shot_id. They must remain separate rather
        # than being merged into one artificial video-level group.
        group_id = frame.shot_id or frame.frame_id
        groups[(frame.video_id, group_id)].append((detection, frame))

    representatives: list[tuple[int, Any, Any]] = []
    for detections in groups.values():
        if len(detections) < min_count:
            continue
        detection, frame = max(
            detections,
            key=lambda item: (
                item[0].confidence,
                -item[1].timestamp_ms,
                -item[0].detection_id,
            ),
        )
        representatives.append((len(detections), detection, frame))

    representatives.sort(
        key=lambda item: (
            -item[0],
            -item[1].confidence,
            item[2].timestamp_ms,
            item[1].detection_id,
        )
    )
    return [
        SearchHit(
            source="postgresql_object_detection",
            entity_type="object_detection",
            entity_id=str(detection.detection_id),
            video_id=frame.video_id,
            shot_id=frame.shot_id,
            frame_id=frame.frame_id,
            start_ms=frame.timestamp_ms,
            end_ms=frame.timestamp_ms,
            rank=rank,
            raw_score=detection.confidence,
            event_id=event_id,
            tool_call_id=tool_call_id,
        )
        for rank, (_, detection, frame) in enumerate(representatives[:top_k], start=1)
    ]


async def object_search(
    object_class: str,
    top_k: int = 100,
    min_count: int = 1,
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    """Find representative frames with the requested object via PostgreSQL."""

    if top_k <= 0:
        return []
    if min_count <= 0:
        raise ValueError("min_count must be greater than zero")
    rows = _get_manager().search_object_detections(object_class)
    return _object_hits_from_rows(
        rows,
        min_count=min_count,
        top_k=top_k,
        event_id=event_id,
        tool_call_id=tool_call_id,
    )


async def track_search(
    object_class: str,
    top_k: int = 100,
    relation: str = "continuous_track",
    event_id: str | None = None,
    tool_call_id: str | None = None,
) -> list[SearchHit]:
    """Find persisted continuous object tracks via PostgreSQL."""

    if top_k <= 0 or relation != "continuous_track":
        return []
    rows = _get_manager().search_object_tracks(object_class)
    return [
        SearchHit(
            source="postgresql_object_track",
            entity_type="object_track",
            entity_id=str(track.track_id),
            video_id=shot.video_id,
            shot_id=shot.shot_id,
            start_ms=track.start_ms,
            end_ms=track.end_ms,
            rank=rank,
            raw_score=track.avg_confidence or 0.0,
            event_id=event_id,
            tool_call_id=tool_call_id,
        )
        for rank, (track, shot) in enumerate(rows[:top_k], start=1)
    ]


__all__ = [
    "configure_object_search_manager",
    "object_search",
    "track_search",
]
