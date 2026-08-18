"""Configuration for the TRAKE temporal aligner."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel


class TrakeAlignerConfig(ContractModel):
    """Tunable settings for deterministic temporal sequence alignment."""

    top_k_sequences: int = Field(default=10, ge=1)
    beam_width: int | None = Field(default=50, ge=1)
    per_video_beam_width: int | None = Field(default=10, ge=1)
    paths_per_node: int = Field(default=5, ge=1)

    gap_mode: Literal["hard", "soft"] = "hard"
    default_min_gap_ms: int = Field(default=0, ge=0)
    default_max_gap_ms: int | None = Field(default=None, gt=0)

    allow_overlap: bool = False
    overlap_tolerance_ms: int | None = Field(default=None, ge=0)

    event_score_weight: float = 1.0
    transition_bonus: float = 0.0
    gap_penalty_weight: float = Field(default=0.0, ge=0)
    overlap_penalty_weight: float = Field(default=0.0, ge=0)

    future_connectivity_pruning: bool = True
    future_score_weight: float = Field(default=1.0, ge=0)
    first_occurrence_mode: Literal["disabled", "strict", "soft"] = "disabled"
    first_occurrence_soft_weight: float = Field(default=0.0, ge=0)
    max_candidates_per_event_per_video: int | None = Field(default=None, ge=1)
