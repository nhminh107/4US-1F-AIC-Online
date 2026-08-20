"""HTTP request and response models for the online pipeline API."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from BackEnd.CONFIG import TOP_K_DEFAULTS
from BackEnd.app.api.config import VQA_MAX_CANDIDATES
from BackEnd.app.contracts.models import (
    ContractModel,
    KISResult,
    StructuredQuery,
    TemporalEventResult,
    ToolCall,
    VerifiedResult,
)
from BackEnd.app.trake.contracts import TrakeAlignerStatus


class TopKParameters(ContractModel):
    """Per-request retrieval limits; omitted values use project defaults."""

    clip_search: int = Field(default=TOP_K_DEFAULTS["clip_search"], ge=1, le=1000)
    frame_search: int = Field(default=TOP_K_DEFAULTS["frame_search"], ge=1, le=1000)
    shot_search: int = Field(default=TOP_K_DEFAULTS["shot_search"], ge=1, le=1000)
    ocr_search: int = Field(default=TOP_K_DEFAULTS["ocr_search"], ge=1, le=1000)
    asr_search: int = Field(default=TOP_K_DEFAULTS["asr_search"], ge=1, le=1000)
    object_search: int = Field(default=TOP_K_DEFAULTS["object_search"], ge=1, le=1000)
    track_search: int = Field(default=TOP_K_DEFAULTS["track_search"], ge=1, le=1000)
    result_top_k: int = Field(default=VQA_MAX_CANDIDATES, ge=1, le=100)

    def for_tool(self, tool_name: str) -> int:
        """Return the configured K for one retrieval tool name."""

        value = getattr(self, tool_name, None)
        if not isinstance(value, int):
            raise ValueError(f"Unsupported top-k tool: {tool_name}")
        return value


class QueryRequest(ContractModel):
    """One user query, optionally including correction feedback."""

    prompt: str = Field(min_length=1)
    feedback: str | None = None
    session_id: str | None = None
    top_k: TopKParameters = Field(default_factory=TopKParameters)


class VisualResult(KISResult):
    """KIS-style result enriched with the official frame shown by the UI.

    ``representative_frame_id`` remains the frame selected by the task handler.
    The three display fields refer to the official frame actually served.
    """

    display_frame_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    img_url: str = Field(min_length=1)


class TrakeEventResult(TemporalEventResult):
    """One aligned TRAKE event with a displayable official frame."""

    display_frame_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    img_url: str = Field(min_length=1)


class TrakeSequenceResult(ContractModel):
    video_id: str = Field(min_length=1)
    events: list[TrakeEventResult] = Field(default_factory=list)
    sequence_score: float


class VerificationItem(ContractModel):
    target_id: str = Field(min_length=1)
    result: VerifiedResult


class VerificationSummary(ContractModel):
    enabled: bool
    applied: bool
    reason: str | None = None
    items: list[VerificationItem] = Field(default_factory=list)


class QueryResponse(ContractModel):
    """Compact API output; raw SearchHit data is intentionally not returned."""

    query_id: str = Field(min_length=1)
    task: Literal["KIS", "VQA", "TRAKE"]
    structured_query: StructuredQuery
    top_k: TopKParameters
    execution_path: Literal[
        "fast_path",
        "query_planner",
        "query_planner_fallback",
    ]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    search_hit_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    results: list[VisualResult] | list[TrakeSequenceResult] = Field(
        default_factory=list
    )
    trake_status: TrakeAlignerStatus | None = None
    replan_required: bool = False
    missing_event_ids: list[str] = Field(default_factory=list)
    verification: VerificationSummary
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(ContractModel):
    status: Literal["ok"] = "ok"
    selective_verifier_enabled: bool


__all__ = [
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "TopKParameters",
    "TrakeEventResult",
    "TrakeSequenceResult",
    "VerificationItem",
    "VerificationSummary",
    "VisualResult",
]
