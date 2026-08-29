from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from BackEnd.app.contracts.models import ContractModel
from BackEnd.app.contracts.models import SearchHit, TemporalConstraint
from BackEnd.app.retrieval_v2.answer_spec import AnswerSpec


PromptRole = Literal["global", "rare_detail", "action", "context", "contrast"]
AtomModality = Literal["visual", "ocr", "asr", "object"]
AtomOperator = Literal["MUST", "SHOULD", "MUST_NOT"]
AtomRole = Literal["REQUIRED", "SUPPORTING"]
AtomGranularity = Literal["FRAME", "MOMENT", "SHOT", "VIDEO"]
AtomType = Literal["ENTITY", "ATTRIBUTE", "ACTION", "RELATION", "COUNT", "TEXT", "CONTEXT"]
AtomScope = Literal["VIDEO_ANCHOR", "EVENT", "ANSWER_EVIDENCE"]
ExecutionProfile = Literal["KIS_MOMENT", "KIS_SEQUENCE", "VQA", "TRAKE"]
EvidenceStatus = Literal["PASS", "FAIL", "UNKNOWN"]
RetrievalEvidenceStatus = Literal["RETRIEVED", "MISSING"]
ConstraintScope = Literal["PLAN", "VIDEO", "MOMENT_BAND", "FRAME", "SUBMISSION_ROW"]


class PromptVariant(ContractModel):
    role: PromptRole
    text: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=2.0)


class AtomLink(ContractModel):
    """Typed relation between two atoms (temporal, spatial, count)."""

    target_atom_id: str = Field(min_length=1)
    relation: Literal[
        "temporal_before",
        "temporal_after",
        "same_moment",
        "spatial_near",
        "count_constraint",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class QueryAtom(ContractModel):
    atom_id: str = Field(min_length=1)
    event_id: str | None = None
    scope: AtomScope = "EVENT"
    group_id: str | None = None
    text: str = Field(min_length=1)
    modality: AtomModality
    atom_type: AtomType = "CONTEXT"
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    count: int | None = Field(default=None, ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)
    relation: str | None = None
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Backward-compatible: kept for code that reads `atom.required`.
    required: bool = True
    operator: AtomOperator = "MUST"
    role: AtomRole = "REQUIRED"
    granularity: AtomGranularity = "MOMENT"
    discriminative_weight: float = Field(ge=0.0, le=2.0)
    prompt_variants: list[PromptVariant] = Field(default_factory=list)
    allowed_retrievers: list[str] = Field(default_factory=list)
    forbidden_retrievers: list[str] = Field(default_factory=list)
    links: list[AtomLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantic_role(self) -> "QueryAtom":
        if self.operator == "MUST_NOT" and self.required:
            raise ValueError("MUST_NOT atoms cannot be positive required atoms")
        if self.role == "SUPPORTING" and self.required:
            raise ValueError("SUPPORTING atoms cannot be required")
        return self


class RetrievalPlan(ContractModel):
    query_id: str = Field(min_length=1)
    task: Literal["KIS", "VQA", "TRAKE"]
    execution_profile: ExecutionProfile
    atoms: list[QueryAtom] = Field(default_factory=list)
    answer_spec: AnswerSpec | None = None
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "RetrievalPlan":
        atom_ids = [atom.atom_id for atom in self.atoms]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("RetrievalPlan atom_id values must be unique")
        known_atoms = set(atom_ids)
        for atom in self.atoms:
            unknown = [link.target_atom_id for link in atom.links if link.target_atom_id not in known_atoms]
            if unknown:
                raise ValueError(f"Atom links reference unknown atoms: {unknown}")
        event_ids = {atom.event_id for atom in self.atoms if atom.event_id is not None}
        for constraint in self.temporal_constraints:
            if constraint.before not in event_ids or constraint.after not in event_ids:
                raise ValueError("Temporal constraints must reference planned event IDs")
        return self


class ScoringProfile(ContractModel):
    """Versioned, externalized scoring weights for fusion ranking."""

    profile_id: str = "default_v1"
    W_required: float = Field(default=0.7, ge=0.0)
    W_supporting: float = Field(default=0.15, ge=0.0)
    W_temporal: float = Field(default=0.1, ge=0.0)
    W_review: float = Field(default=0.05, ge=0.0)
    missing_penalty_per_atom: float = Field(default=0.2, ge=0.0)
    negative_penalty: float = Field(default=0.5, ge=0.0)
    duplicate_penalty: float = Field(default=0.1, ge=0.0)
    epsilon: float = Field(default=0.01, ge=0.0)


class CoverageCell(ContractModel):
    atom_id: str = Field(min_length=1)
    # Retrieval support is not semantic verification. A FAISS/ES neighbor can
    # make this RETRIEVED while the verified tri-state remains UNKNOWN.
    retrieval_status: RetrievalEvidenceStatus = "MISSING"
    status: EvidenceStatus = "UNKNOWN"
    score: float = Field(ge=0.0)
    evidence_ids: list[str] = Field(default_factory=list)
    retriever_families: list[str] = Field(default_factory=list)
    prompt_roles: list[str] = Field(default_factory=list)
    family_scores: dict[str, float] = Field(default_factory=dict)


class MomentBand(ContractModel):
    band_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    event_id: str | None = None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    peak_ms: int = Field(ge=0)
    coverage: dict[str, CoverageCell] = Field(default_factory=dict)
    contradictions: dict[str, CoverageCell] = Field(default_factory=dict)
    constraint_decisions: list[ConstraintDecision] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    hits: list[SearchHit] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> "MomentBand":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if not self.start_ms <= self.peak_ms <= self.end_ms:
            raise ValueError("peak_ms must lie inside the moment band")
        return self


class ConstraintDecision(ContractModel):
    constraint_id: str = Field(min_length=1)
    status: EvidenceStatus
    scope: ConstraintScope
    reason_code: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CandidateBudget(ContractModel):
    raw_retrieval_target: int = Field(default=3_000, ge=1)
    raw_retrieval_max: int = Field(default=5_000, ge=1)
    unique_candidate_min: int = Field(default=1_600, ge=1)
    unique_candidate_max: int = Field(default=2_000, ge=1)
    moment_band_limit: int = Field(default=800, ge=1)
    video_shortlist_limit: int = Field(default=60, ge=1)
    local_retrieval_k: int = Field(default=1_500, ge=1)
    retry_retrieval_k: int = Field(default=800, ge=1)
    rerank_limit: int = Field(default=100, ge=1)
    review_limit: int = Field(default=7, ge=1, le=20)
    max_retry_rounds: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def validate_raw_budget(self) -> "CandidateBudget":
        if self.raw_retrieval_target > self.raw_retrieval_max:
            raise ValueError("raw_retrieval_target must not exceed raw_retrieval_max")
        if self.unique_candidate_min > self.unique_candidate_max:
            raise ValueError("unique_candidate_min must not exceed unique_candidate_max")
        return self


class SearchCall(ContractModel):
    call_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    event_id: str | None = None
    prompt_role: str | None = None
    retriever: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    video_ids: list[str] = Field(default_factory=list)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    min_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_time_scope(self) -> "SearchCall":
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.start_ms > self.end_ms
        ):
            raise ValueError("start_ms must not exceed end_ms")
        return self


class VideoHypothesis(ContractModel):
    video_id: str = Field(min_length=1)
    video_confidence: float = Field(ge=0.0, le=1.0)
    moment_confidence: float = Field(ge=0.0, le=1.0)
    coverage: dict[str, CoverageCell] = Field(default_factory=dict)
    band_ids: list[str] = Field(default_factory=list)
    lane_sources: list[Literal["moment", "video"]] = Field(default_factory=list)


class RetryDiagnosis(ContractModel):
    reason: Literal[
        "MISSING_REQUIRED_ATOM",
        "LOW_VIDEO_CONFIDENCE",
        "LOW_MOMENT_CONFIDENCE",
        "PROMPT_FAMILY_DISAGREEMENT",
        "WRONG_VIDEO",
        "WRONG_MOMENT",
        "MISSING_ACTION",
        "WRONG_RELATION_OR_COUNT",
        "PROMPT_TOO_BROAD",
        "CORRELATED_RETRIEVER_FAMILY",
    ]
    action: Literal[
        "RETRY_WEAK_ATOM",
        "BROADEN_VIDEO_SEARCH",
        "EXPAND_LOCAL_SEARCH",
        "DIVERSIFY_PROMPT",
        "REJECT_VIDEO_AND_BROADEN",
        "REJECT_BAND_AND_LOCAL_SEARCH",
        "RETRY_ACTION_PROMPT",
        "VERIFY_RELATION_OR_COUNT",
        "EXPAND_DISCRIMINATIVE_PROMPT",
        "SWITCH_RETRIEVER_FAMILY",
    ]
    atom_id: str | None = None
    video_id: str | None = None
    band_id: str | None = None


class CandidateReview(ContractModel):
    band_id: str = Field(min_length=1)
    verdict: Literal["match", "partial", "mismatch", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    atom_status: dict[str, EvidenceStatus] = Field(default_factory=dict)
    scope: ConstraintScope | None = None
    failure_reason: Literal[
        "wrong_video",
        "wrong_moment",
        "missing_action",
        "wrong_relation_or_count",
        "prompt_too_broad",
        "correlated_retriever_family",
    ] | None = None
    video_id: str | None = None
    notes: str | None = None


class SearchRound(ContractModel):
    round_index: int = Field(ge=0)
    phase: Literal["GLOBAL", "LOCAL", "RETRY"]
    calls: list[SearchCall] = Field(default_factory=list)
    hit_count: int = Field(ge=0)
    requested_k: int = Field(default=0, ge=0)
    unique_candidate_count: int = Field(default=0, ge=0)
    new_video_gain: int = Field(default=0, ge=0)
    new_moment_gain: int = Field(default=0, ge=0)


class SearchSessionState(ContractModel):
    query_id: str = Field(min_length=1)
    rounds: list[SearchRound] = Field(default_factory=list)
    diagnoses: list[RetryDiagnosis] = Field(default_factory=list)
    raw_hit_count: int = Field(ge=0)
    deduplicated_hit_count: int = Field(ge=0)
    hypotheses: list[VideoHypothesis] = Field(default_factory=list)
    reviews: list[CandidateReview] = Field(default_factory=list)
    stop_reason: Literal[
        "SUFFICIENT_EVIDENCE",
        "NO_RETRY_ACTION",
        "NO_CANDIDATE_GAIN",
        "MAX_RETRIES",
        "COMPLETED",
    ] = "COMPLETED"


class SearchControllerResult(ContractModel):
    plan: RetrievalPlan
    bands: list[MomentBand] = Field(default_factory=list)
    reranked_bands: list[MomentBand] = Field(default_factory=list)
    hypotheses: list[VideoHypothesis] = Field(default_factory=list)
    session: SearchSessionState


__all__ = [
    "AtomGranularity",
    "AtomLink",
    "AtomModality",
    "AtomOperator",
    "AtomRole",
    "AtomScope",
    "AtomType",
    "CandidateBudget",
    "CandidateReview",
    "ExecutionProfile",
    "ConstraintDecision",
    "ConstraintScope",
    "CoverageCell",
    "EvidenceStatus",
    "MomentBand",
    "PromptRole",
    "PromptVariant",
    "QueryAtom",
    "RetrievalPlan",
    "RetrievalEvidenceStatus",
    "RetryDiagnosis",
    "ScoringProfile",
    "SearchCall",
    "SearchControllerResult",
    "SearchRound",
    "SearchSessionState",
    "VideoHypothesis",
]
