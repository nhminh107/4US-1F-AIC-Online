"""Config models for Selective Verifier."""

from __future__ import annotations

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel


class GateConfig(ContractModel):
    min_score_margin: float = Field(default=0.03, ge=0)
    vqa_min_answer_confidence: float = Field(default=0.70, ge=0, le=1)
    trake_min_event_score: float = Field(default=0.55, ge=0)
    trake_min_sequence_margin: float = Field(default=0.03, ge=0)


class BudgetConfig(ContractModel):
    max_rounds: int = Field(default=1, ge=1)
    max_vlm_calls_per_result: int = Field(default=0, ge=0)
    max_reranker_calls_per_result: int = Field(default=0, ge=0)
    max_candidates_to_verify: int = Field(default=3, ge=1)


class EvidenceConfig(ContractModel):
    timeout_ms: int = Field(default=500, ge=1)
    max_frames: int = Field(default=8, ge=0)
    max_text_items: int = Field(default=12, ge=0)
    max_objects: int = Field(default=20, ge=0)
    max_tracks: int = Field(default=20, ge=0)
    neighbor_count: int = Field(default=1, ge=0)


class DeterministicConfig(ContractModel):
    ocr_fuzzy_threshold: float = Field(default=0.90, ge=0, le=1)
    object_min_confidence: float = Field(default=0.25, ge=0, le=1)
    object_missing_confidence: float = Field(default=0.50, ge=0, le=1)
    allow_object_absence_contradiction: bool = False


class VerificationConfig(ContractModel):
    enabled: bool = True
    gate: GateConfig = Field(default_factory=GateConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    deterministic: DeterministicConfig = Field(default_factory=DeterministicConfig)
