"""Grounded VQA handling over ranked online-pipeline evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from BackEnd.app.contracts.models import (
    EvidenceBundle,
    RankedCandidateRegion,
    StructuredQuery,
    VQAResult,
)


@dataclass(frozen=True, slots=True)
class VQAModelAnswer:
    answer: str
    confidence: float


class VQAModelClient(Protocol):
    def answer(
        self,
        *,
        question: str,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> VQAModelAnswer: ...


EvidenceLoader = Callable[[str, int, int], EvidenceBundle]


class VQAHandler:
    """Select bounded evidence, call a VLM, and return the shared VQAResult."""

    def __init__(self, client: VQAModelClient, *, max_candidates: int = 5) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.client = client
        self.max_candidates = max_candidates

    def handle(
        self,
        query: StructuredQuery,
        candidates: Sequence[RankedCandidateRegion],
        evidence_loader: EvidenceLoader,
    ) -> VQAResult:
        if query.task != "VQA":
            raise ValueError("VQAHandler only accepts task='VQA'")
        question = query.question.strip()
        if not question:
            raise ValueError("VQA query must contain a question")

        selected = sorted(candidates, key=lambda item: item.fusion_score, reverse=True)[
            : self.max_candidates
        ]
        bundles = [
            evidence_loader(item.video_id, item.start_ms, item.end_ms)
            for item in selected
        ]
        image_paths = self._image_paths(bundles)
        evidence_ids = self._evidence_ids(bundles)
        if not image_paths:
            return VQAResult(
                answer="Insufficient visual evidence to answer the question.",
                confidence=0.0,
                evidence_ids=evidence_ids,
                status="uncertain",
            )

        model_answer = self.client.answer(
            question=question,
            prompt=self._prompt(question, bundles),
            image_paths=image_paths,
        )
        answer = model_answer.answer.strip()
        confidence = min(max(float(model_answer.confidence), 0.0), 1.0)
        return VQAResult(
            answer=answer,
            confidence=confidence,
            evidence_ids=evidence_ids,
            status="answered" if answer and confidence > 0.0 else "uncertain",
        )

    @staticmethod
    def _image_paths(bundles: Sequence[EvidenceBundle]) -> list[Path]:
        paths: list[Path] = []
        seen: set[Path] = set()
        for bundle in bundles:
            for frame in bundle.frames:
                if frame.frame_path is None:
                    continue
                path = Path(frame.frame_path)
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                paths.append(path)
        return paths

    @staticmethod
    def _evidence_ids(bundles: Sequence[EvidenceBundle]) -> list[str]:
        identifiers: list[str] = []
        for bundle in bundles:
            identifiers.extend(frame.frame_id for frame in bundle.frames)
            identifiers.extend(
                str(caption.caption_id)
                for caption in bundle.captions
                if caption.caption_id is not None
            )
        return list(dict.fromkeys(identifiers))

    @staticmethod
    def _prompt(question: str, bundles: Sequence[EvidenceBundle]) -> str:
        ranges = ", ".join(
            f"{bundle.video_id}:{bundle.start_ms}-{bundle.end_ms}ms"
            for bundle in bundles
        )
        return (
            "Answer only from visible evidence in the supplied frames. "
            "If evidence is insufficient, say so and do not speculate. "
            f"Question: {question}\nCandidate ranges: {ranges}"
        )


__all__ = ["VQAHandler", "VQAModelAnswer", "VQAModelClient"]
