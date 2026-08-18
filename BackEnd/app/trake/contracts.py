"""TRAKE-local contracts built around shared pipeline models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel, TemporalSequence

TrakeAlignerStatus = Literal[
    "success",
    "no_valid_sequence",
    "insufficient_candidates",
    "invalid_input",
]


class TrakeAlignerResult(ContractModel):
    """Result returned by the TRAKE temporal aligner."""

    status: TrakeAlignerStatus
    sequences: list[TemporalSequence] = Field(default_factory=list)
    missing_event_ids: list[str] = Field(default_factory=list)
    replan_required: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)

