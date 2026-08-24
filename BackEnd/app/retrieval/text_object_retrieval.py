"""Deterministic text, object, and tracking retrieval tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from BackEnd.app.Database.sql_models import (
    ClassID,
    Frame,
    ObjectDetection,
    ObjectTrack,
    Shot,
)
from BackEnd.app.contracts.models import SearchHit

TextMode = Literal["exact", "fuzzy", "similarity"]
TextSource = Literal["ocr", "asr", "caption"]


class ElasticsearchClient(Protocol):
    """Subset of the Elasticsearch client used by these tools."""

    def search(self, *, index: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TextIndexConfig:
    ocr_index: str = "ocr_index"
    asr_index: str = "asr_index"
    caption_index: str = "caption_index"
    text_field: str = "content"


class TextRetrievalTools:
    """Query persisted OCR, ASR, and caption indexes and return SearchHit."""

    def __init__(
        self,
        client: ElasticsearchClient,
        config: TextIndexConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or TextIndexConfig()

    def ocr_search(self, query: str, **kwargs: Any) -> list[SearchHit]:
        return self._search("ocr", query, **kwargs)

    def asr_search(self, query: str, **kwargs: Any) -> list[SearchHit]:
        return self._search("asr", query, **kwargs)

    def caption_search(self, query: str, **kwargs: Any) -> list[SearchHit]:
        return self._search("caption", query, **kwargs)

    def _search(
        self,
        source: TextSource,
        query: str,
        *,
        mode: TextMode = "fuzzy",
        top_k: int = 100,
        event_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        index_name = getattr(self.config, f"{source}_index")
        response = self.client.search(
            index=index_name,
            query=self._build_query(query, mode),
            size=top_k,
        )
        raw_hits = response.get("hits", {}).get("hits", [])
        results: list[SearchHit] = []
        for rank, raw_hit in enumerate(raw_hits, start=1):
            document = raw_hit.get("_source", {})
            entity_id = str(document.get("entity_id") or raw_hit.get("_id") or "")
            video_id = str(document.get("video_id") or "")
            if not entity_id or not video_id:
                raise ValueError("Text index hit is missing entity_id or video_id")
            start_ms = int(document.get("start_ms", document.get("timestamp_ms", 0)))
            end_ms = int(document.get("end_ms", start_ms))
            results.append(
                SearchHit(
                    source=source,
                    entity_type=source,
                    entity_id=entity_id,
                    video_id=video_id,
                    shot_id=document.get("shot_id"),
                    clip_id=document.get("clip_id"),
                    frame_id=document.get("frame_id"),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    rank=rank,
                    raw_score=float(raw_hit.get("_score") or 0.0),
                    event_id=event_id,
                    tool_call_id=tool_call_id,
                )
            )
        return results

    def _build_query(self, query: str, mode: TextMode) -> dict[str, Any]:
        if mode == "exact":
            return {"term": {f"{self.config.text_field}.keyword": query}}
        if mode == "fuzzy":
            return {
                "match": {
                    self.config.text_field: {
                        "query": query,
                        "fuzziness": "AUTO",
                    }
                }
            }
        if mode == "similarity":
            return {"match": {self.config.text_field: {"query": query}}}
        raise ValueError(f"Unsupported text search mode: {mode}")


class ObjectTrackingRetrievalTools:
    """Query offline PostgreSQL detections/tracks without rerunning models."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def supported_object_classes(self) -> tuple[str, ...]:
        """Return the database-backed allow-list for the query planner."""

        with self.session_factory() as session:
            return tuple(sorted(session.scalars(select(ClassID.class_name)).all()))

    def object_search(
        self,
        object_class: str,
        *,
        top_k: int = 100,
        min_count: int = 1,
        min_confidence: float = 0.0,
        event_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> list[SearchHit]:
        self._validate_filters(object_class, top_k, min_confidence)
        if min_count <= 0:
            raise ValueError("min_count must be positive")
        with self.session_factory() as session:
            resolved_classes = self._resolve_object_classes(session, object_class)
            statement = (
                select(ObjectDetection, Frame, ClassID)
                .join(Frame, ObjectDetection.frame_id == Frame.frame_id)
                .join(ClassID, ObjectDetection.class_id == ClassID.class_id)
                .where(ClassID.class_name.in_(resolved_classes))
                .where(ObjectDetection.confidence >= min_confidence)
                .order_by(ObjectDetection.confidence.desc())
            )
            rows = session.execute(statement).all()

        grouped: dict[tuple[str, str | None, int], list[tuple[Any, Any, Any]]] = {}
        for detection, frame, object_type in rows:
            key = (frame.video_id, frame.shot_id, frame.timestamp_ms)
            grouped.setdefault(key, []).append((detection, frame, object_type))

        eligible = [items for items in grouped.values() if len(items) >= min_count]
        eligible.sort(key=lambda items: float(items[0][0].confidence), reverse=True)
        return [
            SearchHit(
                source="object_detection",
                entity_type="object_detection",
                entity_id=str(items[0][0].detection_id),
                video_id=items[0][1].video_id,
                shot_id=items[0][1].shot_id,
                frame_id=items[0][1].frame_id,
                start_ms=items[0][1].timestamp_ms,
                end_ms=items[0][1].timestamp_ms,
                rank=rank,
                raw_score=max(float(item[0].confidence) for item in items),
                event_id=event_id,
                tool_call_id=tool_call_id,
            )
            for rank, items in enumerate(eligible[:top_k], start=1)
        ]

    def track_search(
        self,
        object_class: str,
        *,
        top_k: int = 100,
        min_confidence: float = 0.0,
        min_duration_ms: int = 0,
        event_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> list[SearchHit]:
        self._validate_filters(object_class, top_k, min_confidence)
        if min_duration_ms < 0:
            raise ValueError("min_duration_ms must be non-negative")
        with self.session_factory() as session:
            resolved_classes = self._resolve_object_classes(session, object_class)
            statement = (
                select(ObjectTrack, Shot, ClassID)
                .join(Shot, ObjectTrack.shot_id == Shot.shot_id)
                .join(ClassID, ObjectTrack.class_id == ClassID.class_id)
                .where(ClassID.class_name.in_(resolved_classes))
                .where(ObjectTrack.avg_confidence >= min_confidence)
                .where(ObjectTrack.end_ms - ObjectTrack.start_ms >= min_duration_ms)
                .order_by(ObjectTrack.avg_confidence.desc())
                .limit(top_k)
            )
            rows = session.execute(statement).all()
        return [
            SearchHit(
                source="object_track",
                entity_type="object_track",
                entity_id=str(track.track_id),
                video_id=shot.video_id,
                shot_id=track.shot_id,
                start_ms=track.start_ms,
                end_ms=track.end_ms,
                rank=rank,
                raw_score=float(track.avg_confidence or 0.0),
                event_id=event_id,
                tool_call_id=tool_call_id,
            )
            for rank, (track, shot, _object_type) in enumerate(rows, start=1)
        ]

    @staticmethod
    def _validate_filters(object_class: str, top_k: int, confidence: float) -> None:
        if not object_class.strip():
            raise ValueError("object_class must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")

    @staticmethod
    def _resolve_object_classes(
        session: Session,
        requested_class: str,
    ) -> tuple[str, ...]:
        """Resolve planner output against real DB classes and person aliases."""

        available = tuple(session.scalars(select(ClassID.class_name)).all())
        by_normalized = {name.casefold(): name for name in available}
        normalized = requested_class.strip().casefold()
        person_queries = {"person", "people", "human", "người", "nguoi"}
        person_labels = {"person", "man", "woman", "boy", "girl", "human"}
        if normalized in person_queries:
            resolved = tuple(
                name for name in available if name.casefold() in person_labels
            )
        else:
            direct = by_normalized.get(normalized)
            resolved = (direct,) if direct is not None else ()
        if not resolved:
            raise ValueError(
                f"Unsupported object class {requested_class!r}. "
                "Use supported_object_classes() to constrain planner output."
            )
        return resolved


__all__ = [
    "ObjectTrackingRetrievalTools",
    "TextIndexConfig",
    "TextRetrievalTools",
]
