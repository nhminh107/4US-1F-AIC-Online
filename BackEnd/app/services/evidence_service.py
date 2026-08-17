"""Read services for evidence and temporal context."""

from __future__ import annotations

from typing import Literal

from sqlalchemy import and_, or_, select

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Database.sql_models import ClipWindow, Frame, Shot
from BackEnd.app.contracts.models import (
    CanonicalEntityRef,
    EvidenceBundle,
    TemporalNeighbors,
)

EntityType = Literal["frame", "shot", "clip"]


def get_evidence_bundle(
    video_id: str,
    start_ms: int,
    end_ms: int,
    db_mng: PostgreManager,
) -> EvidenceBundle:
    """Load modality-specific evidence for one video time range."""

    result = db_mng.get_evidence_by_video_id_and_time(video_id, start_ms, end_ms)
    shots, clips, frames, ocr, asr, captions, objects, tracks = result

    return EvidenceBundle(
        video_id=video_id,
        start_ms=start_ms,
        end_ms=end_ms,
        shots=shots,
        clips=clips,
        frames=frames,
        ocr=ocr,
        asr=asr,
        captions=captions,
        objects=objects,
        tracks=tracks,
    )


def get_temporal_neighbors(
    entity_type: EntityType,
    entity_id: str,
    db_mng: PostgreManager,
    neighbor_count: int = 1,
) -> TemporalNeighbors:
    """Return previous and next entities of the same type in one video."""

    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive.")

    with db_mng.session_factory() as session:
        if entity_type == "frame":
            frame = session.get(Frame, entity_id)
            if frame is None:
                raise ValueError(f"Frame '{entity_id}' does not exist.")

            previous_frames = list(
                session.scalars(
                    select(Frame)
                    .where(
                        Frame.video_id == frame.video_id,
                        or_(
                            Frame.timestamp_ms < frame.timestamp_ms,
                            and_(
                                Frame.timestamp_ms == frame.timestamp_ms,
                                Frame.frame_idx < frame.frame_idx,
                            ),
                        ),
                    )
                    .order_by(Frame.timestamp_ms.desc(), Frame.frame_idx.desc())
                    .limit(neighbor_count)
                )
            )
            next_frames = list(
                session.scalars(
                    select(Frame)
                    .where(
                        Frame.video_id == frame.video_id,
                        or_(
                            Frame.timestamp_ms > frame.timestamp_ms,
                            and_(
                                Frame.timestamp_ms == frame.timestamp_ms,
                                Frame.frame_idx > frame.frame_idx,
                            ),
                        ),
                    )
                    .order_by(Frame.timestamp_ms, Frame.frame_idx)
                    .limit(neighbor_count)
                )
            )

            return TemporalNeighbors(
                previous=[
                    CanonicalEntityRef(
                        video_id=item.video_id,
                        shot_id=item.shot_id,
                        frame_id=item.frame_id,
                        start_ms=item.timestamp_ms,
                        end_ms=item.timestamp_ms,
                    )
                    for item in reversed(previous_frames)
                ],
                next=[
                    CanonicalEntityRef(
                        video_id=item.video_id,
                        shot_id=item.shot_id,
                        frame_id=item.frame_id,
                        start_ms=item.timestamp_ms,
                        end_ms=item.timestamp_ms,
                    )
                    for item in next_frames
                ],
            )

        if entity_type == "shot":
            shot = session.get(Shot, entity_id)
            if shot is None:
                raise ValueError(f"Shot '{entity_id}' does not exist.")

            previous_shots = list(
                session.scalars(
                    select(Shot)
                    .where(
                        Shot.video_id == shot.video_id,
                        Shot.shot_index < shot.shot_index,
                    )
                    .order_by(Shot.shot_index.desc())
                    .limit(neighbor_count)
                )
            )
            next_shots = list(
                session.scalars(
                    select(Shot)
                    .where(
                        Shot.video_id == shot.video_id,
                        Shot.shot_index > shot.shot_index,
                    )
                    .order_by(Shot.shot_index)
                    .limit(neighbor_count)
                )
            )

            return TemporalNeighbors(
                previous=[
                    CanonicalEntityRef(
                        video_id=item.video_id,
                        shot_id=item.shot_id,
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                    )
                    for item in reversed(previous_shots)
                ],
                next=[
                    CanonicalEntityRef(
                        video_id=item.video_id,
                        shot_id=item.shot_id,
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                    )
                    for item in next_shots
                ],
            )

        if entity_type == "clip":
            clip = session.get(ClipWindow, entity_id)
            if clip is None:
                raise ValueError(f"Clip '{entity_id}' does not exist.")

            shot = session.get(Shot, clip.shot_id)
            if shot is None:
                raise ValueError(f"Shot '{clip.shot_id}' does not exist.")

            previous_clips = list(
                session.scalars(
                    select(ClipWindow)
                    .join(Shot)
                    .where(
                        Shot.video_id == shot.video_id,
                        or_(
                            ClipWindow.start_ms < clip.start_ms,
                            and_(
                                ClipWindow.start_ms == clip.start_ms,
                                ClipWindow.clip_id < clip.clip_id,
                            ),
                        ),
                    )
                    .order_by(ClipWindow.start_ms.desc(), ClipWindow.clip_id.desc())
                    .limit(neighbor_count)
                )
            )
            next_clips = list(
                session.scalars(
                    select(ClipWindow)
                    .join(Shot)
                    .where(
                        Shot.video_id == shot.video_id,
                        or_(
                            ClipWindow.start_ms > clip.start_ms,
                            and_(
                                ClipWindow.start_ms == clip.start_ms,
                                ClipWindow.clip_id > clip.clip_id,
                            ),
                        ),
                    )
                    .order_by(ClipWindow.start_ms, ClipWindow.clip_id)
                    .limit(neighbor_count)
                )
            )

            return TemporalNeighbors(
                previous=[
                    CanonicalEntityRef(
                        video_id=shot.video_id,
                        shot_id=item.shot_id,
                        clip_id=item.clip_id,
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                    )
                    for item in reversed(previous_clips)
                ],
                next=[
                    CanonicalEntityRef(
                        video_id=shot.video_id,
                        shot_id=item.shot_id,
                        clip_id=item.clip_id,
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                    )
                    for item in next_clips
                ],
            )

    raise ValueError(f"Unsupported entity_type: '{entity_type}'.")
