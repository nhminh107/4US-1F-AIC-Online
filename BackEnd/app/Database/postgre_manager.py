"""Read helpers for the video-analysis PostgreSQL database."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .sql_models import (
    Caption,
    ClassID,
    ClipEmbeddingRecord,
    ClipWindow,
    Frame,
    FrameEmbeddingRecord,
    OCR,
    ObjectDetection,
    ObjectTrack,
    Shot,
    ShotEmbeddingRecord,
    TrackObservation,
    TranscriptSegment,
    Video,
)


class Postgre_Manager:
    """Provide named read operations for each database table.

    ``DATABASE_URL`` must follow the format in ``.env.example``. It can be
    passed to the constructor directly or supplied as a process environment
    variable.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None and database_url is not None:
            raise ValueError("Provide either database_url or engine, not both.")

        if engine is None:
            database_url = database_url or os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError(
                    "DATABASE_URL is required. Configure it as shown in .env.example."
                )
            engine = create_engine(database_url, pool_pre_ping=True)

        self._engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Open a session for custom read-only SQLAlchemy queries."""
        with self._session_factory() as session:
            yield session

    # Video
    def get_video_by_id(self, video_id: str) -> Video | None:
        with self.session() as session:
            return session.get(Video, video_id)

    def get_videos(self) -> list[Video]:
        with self.session() as session:
            return list(session.scalars(select(Video).order_by(Video.video_id)))

    # Shot
    def get_shot_by_id(self, shot_id: str) -> Shot | None:
        with self.session() as session:
            return session.get(Shot, shot_id)

    def get_shots_by_video_id(self, video_id: str) -> list[Shot]:
        statement = select(Shot).where(Shot.video_id == video_id).order_by(Shot.shot_index)
        with self.session() as session:
            return list(session.scalars(statement))

    # Frame
    def get_frame_by_id(self, frame_id: str) -> Frame | None:
        with self.session() as session:
            return session.get(Frame, frame_id)

    def get_frames_by_video_id(self, video_id: str) -> list[Frame]:
        statement = select(Frame).where(Frame.video_id == video_id).order_by(Frame.timestamp_ms)
        with self.session() as session:
            return list(session.scalars(statement))

    def get_frames_by_shot_id(self, shot_id: str) -> list[Frame]:
        statement = select(Frame).where(Frame.shot_id == shot_id).order_by(Frame.timestamp_ms)
        with self.session() as session:
            return list(session.scalars(statement))

    def get_frames_by_ocr(self, ocr_text: str) -> list[Frame]:
        """Return distinct frames whose OCR text contains ``ocr_text``."""
        if not ocr_text.strip():
            raise ValueError("ocr_text must not be empty.")
        statement = (
            select(Frame)
            .join(OCR, OCR.frame_id == Frame.frame_id)
            .where(OCR.text.ilike(f"%{ocr_text}%"))
            .distinct()
            .order_by(Frame.timestamp_ms)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # ClassID
    def get_class_by_id(self, class_id: str) -> ClassID | None:
        with self.session() as session:
            return session.get(ClassID, class_id)

    def get_class_by_name(self, class_name: str) -> ClassID | None:
        statement = select(ClassID).where(ClassID.class_name == class_name)
        with self.session() as session:
            return session.scalars(statement).one_or_none()

    # OCR
    def get_ocr_by_id(self, frame_id: str, n: int) -> OCR | None:
        with self.session() as session:
            return session.get(OCR, (frame_id, n))

    def get_ocr_by_frame_id(self, frame_id: str) -> list[OCR]:
        statement = select(OCR).where(OCR.frame_id == frame_id).order_by(OCR.n)
        with self.session() as session:
            return list(session.scalars(statement))

    # ObjectDetection
    def get_object_detection_by_id(self, detection_id: int) -> ObjectDetection | None:
        with self.session() as session:
            return session.get(ObjectDetection, detection_id)

    def get_object_detections_by_frame_id(self, frame_id: str) -> list[ObjectDetection]:
        statement = (
            select(ObjectDetection)
            .where(ObjectDetection.frame_id == frame_id)
            .order_by(ObjectDetection.detection_id)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # FrameEmbeddingRecord
    def get_frame_embedding_by_id(
        self, faiss_id: int, index_version: int
    ) -> FrameEmbeddingRecord | None:
        with self.session() as session:
            return session.get(FrameEmbeddingRecord, (faiss_id, index_version))

    def get_frame_embeddings_by_frame_id(self, frame_id: str) -> list[FrameEmbeddingRecord]:
        statement = (
            select(FrameEmbeddingRecord)
            .where(FrameEmbeddingRecord.frame_id == frame_id)
            .order_by(FrameEmbeddingRecord.index_version)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # TranscriptSegment
    def get_transcript_segment_by_id(self, segment_id: str) -> TranscriptSegment | None:
        with self.session() as session:
            return session.get(TranscriptSegment, segment_id)

    def get_transcript_segments_by_video_id(self, video_id: str) -> list[TranscriptSegment]:
        statement = (
            select(TranscriptSegment)
            .where(TranscriptSegment.video_id == video_id)
            .order_by(TranscriptSegment.start_ms)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # ObjectTrack
    def get_object_track_by_id(self, track_id: int) -> ObjectTrack | None:
        with self.session() as session:
            return session.get(ObjectTrack, track_id)

    def get_object_tracks_by_shot_id(self, shot_id: str) -> list[ObjectTrack]:
        statement = (
            select(ObjectTrack)
            .where(ObjectTrack.shot_id == shot_id)
            .order_by(ObjectTrack.track_id)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # TrackObservation
    def get_track_observation_by_id(
        self, track_id: int, frame_idx: int
    ) -> TrackObservation | None:
        with self.session() as session:
            return session.get(TrackObservation, (track_id, frame_idx))

    def get_track_observations_by_track_id(self, track_id: int) -> list[TrackObservation]:
        statement = (
            select(TrackObservation)
            .where(TrackObservation.track_id == track_id)
            .order_by(TrackObservation.frame_idx)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # ClipWindow
    def get_clip_by_id(self, clip_id: str) -> ClipWindow | None:
        with self.session() as session:
            return session.get(ClipWindow, clip_id)

    def get_clips_by_shot_id(self, shot_id: str) -> list[ClipWindow]:
        statement = (
            select(ClipWindow)
            .where(ClipWindow.shot_id == shot_id)
            .order_by(ClipWindow.start_ms)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # ClipEmbeddingRecord
    def get_clip_embedding_by_id(
        self, faiss_id: int, index_version: int
    ) -> ClipEmbeddingRecord | None:
        with self.session() as session:
            return session.get(ClipEmbeddingRecord, (faiss_id, index_version))

    def get_clip_embeddings_by_clip_id(self, clip_id: str) -> list[ClipEmbeddingRecord]:
        statement = (
            select(ClipEmbeddingRecord)
            .where(ClipEmbeddingRecord.clip_id == clip_id)
            .order_by(ClipEmbeddingRecord.index_version)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # ShotEmbeddingRecord
    def get_shot_embedding_by_id(
        self, faiss_id: int, index_version: int
    ) -> ShotEmbeddingRecord | None:
        with self.session() as session:
            return session.get(ShotEmbeddingRecord, (faiss_id, index_version))

    def get_shot_embeddings_by_shot_id(self, shot_id: str) -> list[ShotEmbeddingRecord]:
        statement = (
            select(ShotEmbeddingRecord)
            .where(ShotEmbeddingRecord.shot_id == shot_id)
            .order_by(ShotEmbeddingRecord.index_version)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    # Caption
    def get_caption_by_id(self, caption_id: int) -> Caption | None:
        with self.session() as session:
            return session.get(Caption, caption_id)

    def get_captions_by_frame_id(self, frame_id: str) -> list[Caption]:
        statement = (
            select(Caption)
            .where(Caption.frame_id == frame_id)
            .order_by(Caption.created_at)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    def get_captions_by_clip_id(self, clip_id: str) -> list[Caption]:
        statement = (
            select(Caption)
            .where(Caption.clip_id == clip_id)
            .order_by(Caption.created_at)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    def get_captions_by_shot_id(self, shot_id: str) -> list[Caption]:
        statement = (
            select(Caption)
            .where(Caption.shot_id == shot_id)
            .order_by(Caption.created_at)
        )
        with self.session() as session:
            return list(session.scalars(statement))

    def dispose(self) -> None:
        """Close all pooled connections."""
        self._engine.dispose()
