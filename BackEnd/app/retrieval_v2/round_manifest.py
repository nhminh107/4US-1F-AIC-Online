"""Per-round configuration manifest (pipeline.md §2.5, §7.4).

Instead of hardcoding submission limits and scoring profiles, each competition
round is described by a ``RoundManifest`` loaded from config.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel
from BackEnd.app.retrieval_v2.contracts import ScoringProfile


class RoundManifest(ContractModel):
    """Competition round configuration."""

    round_id: str = Field(min_length=1)
    max_submissions: int = Field(default=4, ge=1, le=10)
    max_rows_per_submission: int = Field(default=100, ge=1)
    tasks: list[Literal["KIS", "VQA", "TRAKE"]] = Field(
        default_factory=lambda: ["KIS", "VQA", "TRAKE"]
    )
    scoring_profile: ScoringProfile = Field(default_factory=ScoringProfile)
    # Per-task answer variant budget (e.g., QA may allow 2-3 answer variants)
    answer_variant_budget: int = Field(default=1, ge=1, le=5)
    notes: str | None = None


# Default manifest used when no round-specific config is loaded
DEFAULT_MANIFEST = RoundManifest(
    round_id="default",
    max_submissions=4,
    notes="Default manifest; update per-round from competition announcements.",
)


__all__ = ["DEFAULT_MANIFEST", "RoundManifest"]
