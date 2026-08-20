"""Prepare ranked visual evidence for a human VQA reviewer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from BackEnd.app.contracts.models import (
    EvidenceBundle,
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
)


@dataclass(frozen=True, slots=True)
class VQAModelAnswer:
    """Compatibility contract for the optional, inactive FPT VLM client."""

    answer: str
    confidence: float


class VQAModelClient(Protocol):
    """Compatibility protocol for a future automated VQA implementation."""

    def answer(
        self,
        *,
        question: str,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> VQAModelAnswer: ...


EvidenceLoader = Callable[[str, int, int], EvidenceBundle]


class VQAHandler:
    """Return KIS-style visual candidates for a human to answer a VQA question.

    Retrieval and fusion have already used ``StructuredQuery.question`` to rank
    regions. This handler deliberately does not answer the question or invoke a
    VLM: it selects representative frames that a human reviewer can inspect.
    """

    def __init__(self, *, max_candidates: int = 5) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = max_candidates

    def handle(
        self,
        query: StructuredQuery,
        candidates: Sequence[RankedCandidateRegion],
        evidence_loader: EvidenceLoader,
    ) -> list[KISResult]:
        if query.task != "VQA":
            raise ValueError("VQAHandler only accepts task='VQA'")
        if not query.question.strip():
            raise ValueError("VQA query must contain a question")

        results: list[KISResult] = []
        for candidate in self._ranked_eligible_candidates(candidates):
            bundle = evidence_loader(
                candidate.video_id,
                candidate.start_ms,
                candidate.end_ms,
            )
            frame_id = self._representative_frame_id(candidate, bundle)
            if frame_id is None:
                continue

            results.append(
                KISResult(
                    video_id=candidate.video_id,
                    start_ms=candidate.start_ms,
                    end_ms=candidate.end_ms,
                    representative_frame_id=frame_id,
                    score=candidate.fusion_score,
                    evidence_ids=self._evidence_ids(candidate, frame_id),
                )
            )
            if len(results) == self.max_candidates:
                break
        return results

    @staticmethod
    def _ranked_eligible_candidates(
        candidates: Sequence[RankedCandidateRegion],
    ) -> list[RankedCandidateRegion]:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.constraint_result.hard_constraints_passed
            and candidate.constraint_result.negative_constraints_passed
        ]
        return sorted(eligible, key=lambda item: item.fusion_score, reverse=True)

    @staticmethod
    def _representative_frame_id(
        candidate: RankedCandidateRegion,
        bundle: EvidenceBundle,
    ) -> str | None:
        if not bundle.frames:
            return None

        midpoint_ms = (candidate.start_ms + candidate.end_ms) / 2
        return min(
            bundle.frames,
            key=lambda frame: (abs(frame.timestamp_ms - midpoint_ms), frame.frame_id),
        ).frame_id

    @staticmethod
    def _evidence_ids(candidate: RankedCandidateRegion, frame_id: str) -> list[str]:
        identifiers = [frame_id]
        identifiers.extend(item.entity_id for item in candidate.evidence)
        return list(dict.fromkeys(identifiers))


__all__ = ["VQAHandler", "VQAModelAnswer", "VQAModelClient"]
