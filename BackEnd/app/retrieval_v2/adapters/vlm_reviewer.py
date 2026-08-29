"""Task 12: VLM review adapter protocol and fallback implementation."""

from __future__ import annotations

from typing import Protocol

from BackEnd.app.retrieval_v2.contracts import (
    CandidateReview,
    MomentBand,
    RetrievalPlan,
)


class VLMReviewer(Protocol):
    """Protocol for Visual Language Model (VLM) candidate reviewers."""

    async def review(
        self,
        plan: RetrievalPlan,
        bands: list[MomentBand],
    ) -> list[CandidateReview]:
        """Review top moment bands and emit structured CandidateReview verdicts."""
        ...


class DeterministicFallbackReviewer:
    """Honest fallback: preserve candidates without claiming image inspection."""

    async def review(
        self,
        plan: RetrievalPlan,
        bands: list[MomentBand],
    ) -> list[CandidateReview]:
        return [
            CandidateReview(
                band_id=band.band_id,
                verdict="uncertain",
                confidence=0.0,
                video_id=band.video_id,
                atom_status={
                    atom_id: cell.status for atom_id, cell in band.coverage.items()
                },
                notes="No visual reviewer configured; retrieval evidence is unverified.",
            )
            for band in bands
        ]


__all__ = ["DeterministicFallbackReviewer", "VLMReviewer"]
