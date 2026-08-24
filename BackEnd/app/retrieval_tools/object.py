from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.contracts.models import SearchHit


_db_manager: Any | None = None
_MIN_OBJECT_CONFIDENCE = 0.5
_DUPLICATE_IOU_THRESHOLD = 0.7


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
        if float(detection.confidence) < _MIN_OBJECT_CONFIDENCE:
            continue
        # Count means simultaneous objects, so detections may only be combined
        # inside the same frame. A shot-level group would count one tracked
        # object repeatedly across time as several simultaneous objects.
        groups[(frame.video_id, frame.frame_id)].append((detection, frame))

    representatives: list[tuple[int, Any, Any]] = []
    for raw_detections in groups.values():
        detections: list[tuple[Any, Any]] = []
        for item in sorted(raw_detections, key=lambda value: -value[0].confidence):
            if any(
                _box_iou(item[0], kept[0]) >= _DUPLICATE_IOU_THRESHOLD
                for kept in detections
            ):
                continue
            detections.append(item)
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


def _box_iou(left: Any, right: Any) -> float:
    fields = ("x_min", "x_max", "y_min", "y_max")
    if any(not hasattr(left, field) or not hasattr(right, field) for field in fields):
        return 0.0
    x_min = max(float(left.x_min), float(right.x_min))
    x_max = min(float(left.x_max), float(right.x_max))
    y_min = max(float(left.y_min), float(right.y_min))
    y_max = min(float(left.y_max), float(right.y_max))
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = max(0.0, float(left.x_max) - float(left.x_min)) * max(
        0.0, float(left.y_max) - float(left.y_min)
    )
    right_area = max(0.0, float(right.x_max) - float(right.x_min)) * max(
        0.0, float(right.y_max) - float(right.y_min)
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


async def object_search(
    object_class: str,
    top_k: int = 100,
    min_count: int = 1,
    event_id: str | None = None,
    tool_call_id: str | None = None,
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[SearchHit]:
    """Find representative frames with the requested object via PostgreSQL."""

    if top_k <= 0:
        return []
    if min_count <= 0:
        raise ValueError("min_count must be greater than zero")
    limit = max(top_k * 20, 1000)
    manager = _get_manager()

    def load_rows():
        try:
            return manager.search_object_detections(
                object_class,
                limit=limit,
                video_ids=video_ids,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        except TypeError:
            return manager.search_object_detections(object_class)
    rows = await asyncio.to_thread(load_rows)
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
    video_ids: list[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[SearchHit]:
    """Find persisted continuous object tracks via PostgreSQL."""

    if top_k <= 0 or relation != "continuous_track":
        return []
    limit = max(top_k * 5, 500)
    manager = _get_manager()

    def load_rows():
        try:
            return manager.search_object_tracks(
                object_class,
                limit=limit,
                video_ids=video_ids,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        except TypeError:
            return manager.search_object_tracks(object_class)
    rows = await asyncio.to_thread(load_rows)
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
