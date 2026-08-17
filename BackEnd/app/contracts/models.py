"""Shared data contracts for the online retrieval pipeline."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from BackEnd.app.contracts.pipeline import (
    CaptionResult,
    ClipWindowMetadata,
    FrameMetadata,
    ObjectDetectionResult,
    ObjectTrackResult,
    OCRResult,
    ShotMetadata,
    TranscriptSegmentResult,
)


class ContractModel(BaseModel):
    """Base configuration shared by contracts exchanged between modules."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TimeRangeModel(ContractModel):
    """A millisecond interval or a point in time within a video."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class CanonicalEntityRef(TimeRangeModel):
    """Canonical reference that resolves an offline entity for online modules."""

    video_id: str = Field(min_length=1)
    shot_id: str | None = None
    clip_id: str | None = None
    frame_id: str | None = None


class Provenance(ContractModel):
    """Origin metadata attached to evidence produced by an offline run."""

    model_name: str | None = None
    model_version: str | None = None
    run_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class EvidenceItem(CanonicalEntityRef):
    """Typed evidence item returned by the shared evidence layer."""

    entity_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    text: str | None = None
    media_ref: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: Provenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalNeighbors(ContractModel):
    """Neighbor references around a known entity or candidate region."""

    previous: list[CanonicalEntityRef] = Field(default_factory=list)
    next: list[CanonicalEntityRef] = Field(default_factory=list)


class MediaReference(CanonicalEntityRef):
    """A local path or URL used by the UI to render a canonical entity."""

    entity_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    uri: str = Field(min_length=1)


class RawQuery(ContractModel):
    query_id: str | None = None
    session_id: str | None = None
    text: str = Field(min_length=1)
    image_ref: str | None = None
    video_ref: str | None = None
    feedback: str | None = None


class Event(ContractModel):
    event_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TemporalConstraint(ContractModel):
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)
    max_gap_ms: int | None = Field(default=None, gt=0)
    allow_overlap: bool = False


class StructuredQuery(ContractModel):
    query_id: str = Field(min_length=1)
    task: Literal["KIS", "VQA", "TRAKE"]
    question: str = ""
    visual_queries: list[str] = Field(default_factory=list)
    ocr_constraints: list[str] = Field(default_factory=list)
    asr_constraints: list[str] = Field(default_factory=list)
    object_constraints: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(
        default_factory=list,
        validation_alias=AliasChoices("temporal_constraints", "temporal_constraint"),
    )
    negative_constraints: list[str] = Field(default_factory=list)

    @property
    def temporal_constraint(self) -> list[TemporalConstraint]:
        """Backward-compatible alias for the previous singular field name."""

        return self.temporal_constraints


class ToolCall(ContractModel):
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    event_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class SearchHit(CanonicalEntityRef):
    tool_call_id: str | None = None
    event_id: str | None = None
    source: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float


class CandidateEvidence(ContractModel):
    source: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float


class CandidateRegion(TimeRangeModel):
    candidate_id: str = Field(min_length=1)
    event_id: str | None = None
    video_id: str = Field(min_length=1)
    evidence: list[CandidateEvidence] = Field(default_factory=list)


class ConstraintResult(ContractModel):
    hard_constraints_passed: bool
    negative_constraints_passed: bool


class RankedCandidateRegion(TimeRangeModel):
    candidate_id: str = Field(min_length=1)
    event_id: str | None = None
    video_id: str = Field(min_length=1)
    fusion_score: float
    constraint_result: ConstraintResult
    evidence: list[CandidateEvidence] = Field(default_factory=list)


class EvidenceBundle(TimeRangeModel):
    video_id: str = Field(min_length=1)
    shots: list[ShotMetadata] = Field(default_factory=list)
    clips: list[ClipWindowMetadata] = Field(default_factory=list)
    frames: list[FrameMetadata] = Field(default_factory=list)
    ocr: list[OCRResult] = Field(default_factory=list)
    asr: list[TranscriptSegmentResult] = Field(default_factory=list)
    captions: list[CaptionResult] = Field(default_factory=list)
    objects: list[ObjectDetectionResult] = Field(default_factory=list)
    tracks: list[ObjectTrackResult] = Field(default_factory=list)


class KISResult(TimeRangeModel):
    video_id: str = Field(min_length=1)
    representative_frame_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class VQAResult(ContractModel):
    answer: str = ""
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["answered", "uncertain"]


class TemporalEventResult(TimeRangeModel):
    event_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)


class TemporalSequence(ContractModel):
    video_id: str = Field(min_length=1)
    events: list[TemporalEventResult] = Field(default_factory=list)
    sequence_score: float


class VerifiedResult(ContractModel):
    status: Literal["accepted", "rejected", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    failed_constraints: list[str] = Field(default_factory=list)
