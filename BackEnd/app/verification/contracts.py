"""Internal contracts for Selective Verifier orchestration."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from BackEnd.app.contracts.models import ContractModel, TimeRangeModel
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    ClaimType,
    NextAction,
    VerificationLevel,
    VerificationStatus,
)


TaskName = Literal["KIS", "VQA", "TRAKE"]
EvidenceType = Literal["frame", "ocr", "asr", "caption", "object", "track"]


class ObjectCountSpec(ContractModel):
    operator: Literal["at_least", "exact", "at_most"]
    expected_count: int = Field(ge=1)
    object_label: str = Field(min_length=1)


class VerificationBudget(ContractModel):
    max_rounds: int = Field(default=1, ge=1)
    max_vlm_calls: int = Field(default=0, ge=0)
    max_reranker_calls: int = Field(default=0, ge=0)


class VerificationClaim(ContractModel):
    claim_id: str = Field(min_length=1)
    claim_type: ClaimType
    text: str = Field(min_length=1)
    importance: ClaimImportance = ClaimImportance.HARD
    current_status: ClaimStatus = ClaimStatus.NOT_CHECKED
    source_constraint_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationPlan(ContractModel):
    verification_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    task: TaskName
    target_result_id: str = Field(min_length=1)
    target_video_id: str | None = None
    target_start_ms: int | None = Field(default=None, ge=0)
    target_end_ms: int | None = Field(default=None, ge=0)
    claims: list[VerificationClaim] = Field(default_factory=list)
    focus_claim_ids: list[str] = Field(default_factory=list)
    required_evidence_types: list[EvidenceType] = Field(default_factory=list)
    candidate_window: dict[str, int] | None = None
    max_verification_level: VerificationLevel = VerificationLevel.DETERMINISTIC
    budget: VerificationBudget = Field(default_factory=VerificationBudget)


class TextEvidence(TimeRangeModel):
    evidence_id: str = Field(min_length=1)
    evidence_type: Literal["ocr", "asr", "caption"]
    text: str = ""


class FrameEvidence(TimeRangeModel):
    evidence_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    frame_path: str | None = None


class ObjectEvidence(TimeRangeModel):
    evidence_id: str = Field(min_length=1)
    frame_id: str | None = None
    class_id: str | None = None
    class_name: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TrackEvidence(TimeRangeModel):
    evidence_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    observation_count: int = Field(default=1, ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class VerificationEvidencePack(TimeRangeModel):
    verification_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_evidence: list[FrameEvidence] = Field(default_factory=list)
    text_evidence: list[TextEvidence] = Field(default_factory=list)
    object_evidence: list[ObjectEvidence] = Field(default_factory=list)
    track_evidence: list[TrackEvidence] = Field(default_factory=list)
    omitted_evidence_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> Self:
        ids = [
            item.evidence_id
            for items in (
                self.frame_evidence,
                self.text_evidence,
                self.object_evidence,
                self.track_evidence,
            )
            for item in items
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique across modalities")
        return self

    def evidence_ids(self) -> set[str]:
        return {
            item.evidence_id
            for items in (
                self.frame_evidence,
                self.text_evidence,
                self.object_evidence,
                self.track_evidence,
            )
            for item in items
        }


class VerificationContext(ContractModel):
    query_id: str = Field(min_length=1)
    task: TaskName
    target_result_id: str = Field(min_length=1)
    top_score: float | None = None
    second_score: float | None = None
    score_margin: float | None = None
    supporting_modalities: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    candidate_duration_ms: int | None = None
    answer_confidence: float | None = None
    answer_evidence_count: int | None = None
    weakest_event_score: float | None = None
    sequence_margin: float | None = None
    hard_unknown: int = 0
    hard_contradicted: int = 0


class VerificationGateDecision(ContractModel):
    should_verify: bool
    reasons: list[str] = Field(default_factory=list)
    direct_status: VerificationStatus | None = None


class PreflightResult(ContractModel):
    claim_results: list["ClaimVerificationResult"] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    terminal_status: VerificationStatus | None = None

    @property
    def hard_unknown(self) -> int:
        return sum(
            result.importance == ClaimImportance.HARD
            and result.status in {ClaimStatus.UNKNOWN, ClaimStatus.NOT_CHECKED}
            for result in self.claim_results
        )

    @property
    def hard_contradicted(self) -> int:
        return sum(
            result.importance == ClaimImportance.HARD
            and result.status == ClaimStatus.CONTRADICTED
            for result in self.claim_results
        )


class ClaimVerificationResult(ContractModel):
    claim_id: str = Field(min_length=1)
    status: ClaimStatus
    confidence: float = Field(ge=0, le=1)
    importance: ClaimImportance
    evidence_ids: list[str] = Field(default_factory=list)
    verifier_type: Literal["deterministic", "reranker", "vlm"]
    verifier_name: str = Field(min_length=1)
    verifier_version: str | None = None
    observation: str | None = None


class VerificationDetailResult(ContractModel):
    verification_id: str = Field(min_length=1)
    task: TaskName
    target_result_id: str = Field(min_length=1)
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1)
    verification_level: VerificationLevel
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    failed_constraint_ids: list[str] = Field(default_factory=list)
    uncertain_constraint_ids: list[str] = Field(default_factory=list)
    gate_reasons: list[str] = Field(default_factory=list)
    focus_claim_ids: list[str] = Field(default_factory=list)
    evidence_count_by_type: dict[str, int] = Field(default_factory=dict)
    omitted_evidence_count: int = Field(default=0, ge=0)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    next_action: NextAction = NextAction.RETURN_RESULT
