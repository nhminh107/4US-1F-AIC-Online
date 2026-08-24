"""QA/VQA answer specification and grounding contracts (pipeline.md §8).

Separates question analysis from answer generation.
``AnswerSpec`` describes what kind of answer is expected.
``GroundedAnswer`` carries the answer with evidence provenance.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel


class AnswerSpec(ContractModel):
    """Describes the expected answer type and source for a VQA question."""

    answer_type: Literal[
        "NUMBER", "TEXT", "COLOR", "PERSON", "PLACE", "BOOLEAN", "OTHER",
    ] = "OTHER"
    answer_source: Literal["VISUAL", "OCR", "ASR", "MIXED"] = "VISUAL"
    normalization: str = "short_vietnamese"
    max_length: int = Field(default=100, ge=1, le=200)


class GroundedAnswer(ContractModel):
    """A QA answer grounded in specific evidence."""

    answer_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    answer_source: Literal["VISUAL", "OCR", "ASR", "MIXED"] = "VISUAL"
    # Which band/candidate provided the evidence
    band_id: str | None = None
    video_id: str | None = None
    frame_id: str | None = None


class AnswerGenerator:
    """Rule-based answer generator for QA.

    MVP implementation: extracts answer from OCR/ASR evidence when answer_source
    matches, returns uncertain otherwise. VLM integration comes in Phase H.
    """

    def generate(
        self,
        question: str,
        spec: AnswerSpec,
        *,
        ocr_texts: list[str] | None = None,
        asr_texts: list[str] | None = None,
    ) -> GroundedAnswer | None:
        """Attempt to generate a grounded answer from available evidence.

        Returns None if insufficient evidence to answer confidently.
        """
        # OCR-sourced answers: return the most relevant OCR text
        if spec.answer_source == "OCR" and ocr_texts:
            # Simple heuristic: shortest non-empty OCR text as answer
            candidates = sorted(
                (t for t in ocr_texts if t.strip()),
                key=len,
            )
            if candidates:
                answer = candidates[0][: spec.max_length]
                return GroundedAnswer(
                    answer_text=answer,
                    confidence=0.6,
                    answer_source="OCR",
                )

        # ASR-sourced answers: return the most relevant ASR segment
        if spec.answer_source == "ASR" and asr_texts:
            candidates = sorted(
                (t for t in asr_texts if t.strip()),
                key=len,
            )
            if candidates:
                answer = candidates[0][: spec.max_length]
                return GroundedAnswer(
                    answer_text=answer,
                    confidence=0.5,
                    answer_source="ASR",
                )

        # Visual/Mixed/Other: needs VLM — return None for MVP
        return None


def infer_answer_spec(
    question: str,
    *,
    has_ocr: bool = False,
    has_asr: bool = False,
) -> AnswerSpec:
    """Infer a conservative answer contract without attempting the answer."""

    normalized = question.casefold()
    if any(token in normalized for token in ("how many", "bao nhiêu", "số lượng")):
        answer_type = "NUMBER"
    elif any(token in normalized for token in ("what color", "màu gì", "màu nào")):
        answer_type = "COLOR"
    elif any(token in normalized for token in ("who", "ai ", "người nào")):
        answer_type = "PERSON"
    elif any(token in normalized for token in ("where", "ở đâu", "địa điểm")):
        answer_type = "PLACE"
    elif any(token in normalized for token in ("yes or no", "có phải")):
        answer_type = "BOOLEAN"
    elif has_ocr:
        answer_type = "TEXT"
    else:
        answer_type = "OTHER"

    visual_reasoning = answer_type in {"NUMBER", "COLOR", "PERSON", "PLACE", "BOOLEAN"}
    if has_ocr and visual_reasoning:
        answer_source = "MIXED"
    elif has_ocr:
        answer_source = "OCR"
    elif has_asr:
        answer_source = "ASR"
    else:
        answer_source = "VISUAL"
    return AnswerSpec(answer_type=answer_type, answer_source=answer_source)


__all__ = ["AnswerGenerator", "AnswerSpec", "GroundedAnswer", "infer_answer_spec"]
