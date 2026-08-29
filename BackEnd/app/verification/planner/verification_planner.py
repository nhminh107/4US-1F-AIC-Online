"""Verification plan construction."""

from __future__ import annotations

from uuid import uuid4

from BackEnd.app.contracts.models import (
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.contracts import VerificationPlan
from BackEnd.app.verification.enums import ClaimImportance, ClaimStatus, ClaimType
from BackEnd.app.verification.planner.claim_builder import ClaimBuilder


class VerificationPlanner:
    def __init__(self, claim_builder: ClaimBuilder | None = None) -> None:
        self.claim_builder = claim_builder or ClaimBuilder()

    def build_plan(
        self,
        query: StructuredQuery,
        result: KISResult | VQAResult | TemporalSequence,
        ranked_candidates: list[RankedCandidateRegion] | None = None,
    ) -> VerificationPlan:
        claims = self.claim_builder.build_claims(query, result)
        focus_claim_ids = [
            claim.claim_id
            for claim in claims
            if claim.current_status in {ClaimStatus.NOT_CHECKED, ClaimStatus.UNKNOWN}
        ]
        target_result_id, target_video_id, target_start_ms, target_end_ms = (
            self._target_context(result, ranked_candidates or [])
        )
        return VerificationPlan(
            verification_id=f"ver-{uuid4().hex}",
            query_id=query.query_id,
            task=query.task,
            target_result_id=target_result_id,
            target_video_id=target_video_id,
            target_start_ms=target_start_ms,
            target_end_ms=target_end_ms,
            claims=claims,
            focus_claim_ids=focus_claim_ids,
            required_evidence_types=self._required_evidence_types(claims),
            candidate_window=(
                {"start_ms": target_start_ms, "end_ms": target_end_ms}
                if target_start_ms is not None and target_end_ms is not None
                else None
            ),
        )

    @staticmethod
    def _target_context(
        result: KISResult | VQAResult | TemporalSequence,
        ranked_candidates: list[RankedCandidateRegion],
    ) -> tuple[str, str | None, int | None, int | None]:
        if isinstance(result, KISResult):
            return (
                result.representative_frame_id,
                result.video_id,
                result.start_ms,
                result.end_ms,
            )
        if isinstance(result, TemporalSequence):
            target_result_id = (
                "|".join(event.candidate_id for event in result.events)
                or result.video_id
            )
            if not result.events:
                return target_result_id, result.video_id, None, None
            return (
                target_result_id,
                result.video_id,
                min(event.start_ms for event in result.events),
                max(event.end_ms for event in result.events),
            )
        if ranked_candidates:
            candidate = max(
                ranked_candidates,
                key=lambda ranked: ranked.fusion_score,
            )
            return (
                candidate.candidate_id,
                candidate.video_id,
                candidate.start_ms,
                candidate.end_ms,
            )
        return "vqa-result", None, None, None

    @staticmethod
    def _required_evidence_types(claims) -> list:
        evidence_types: set[str] = set()
        for claim in claims:
            if claim.claim_type == ClaimType.OCR_EXACT:
                evidence_types.add("ocr")
            elif claim.claim_type == ClaimType.ASR_EXACT:
                evidence_types.add("asr")
            elif claim.claim_type in {
                ClaimType.OBJECT_PRESENCE,
                ClaimType.OBJECT_COUNT,
            }:
                evidence_types.add("object")
            elif claim.claim_type in {
                ClaimType.NEGATIVE_CONSTRAINT,
                ClaimType.VQA_ANSWER_CLAIM,
            }:
                evidence_types.update({"ocr", "asr", "caption"})
            elif claim.claim_type == ClaimType.KIS_MOMENT:
                evidence_types.add("frame")
            elif claim.claim_type == ClaimType.TRAKE_EVENT:
                evidence_types.update({"frame", "object", "ocr", "asr"})
        return sorted(evidence_types)
