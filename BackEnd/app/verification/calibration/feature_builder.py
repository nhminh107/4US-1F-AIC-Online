"""Build verification gate features from existing pipeline contracts."""

from __future__ import annotations

from BackEnd.app.contracts.models import (
    KISResult,
    RankedCandidateRegion,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.contracts import PreflightResult, TaskName, VerificationContext


def build_verification_context(
    *,
    query_id: str,
    task: TaskName,
    result: KISResult | VQAResult | TemporalSequence,
    ranked_candidates: list[RankedCandidateRegion],
    preflight: PreflightResult | None = None,
) -> VerificationContext:
    sorted_candidates = sorted(
        ranked_candidates,
        key=lambda candidate: candidate.fusion_score,
        reverse=True,
    )
    top_score = sorted_candidates[0].fusion_score if sorted_candidates else None
    second_score = sorted_candidates[1].fusion_score if len(sorted_candidates) > 1 else None
    score_margin = (
        top_score - second_score
        if top_score is not None and second_score is not None
        else None
    )

    target_result_id = _target_result_id(result, sorted_candidates)
    modalities = sorted(
        {
            evidence.entity_type
            for candidate in sorted_candidates[:1]
            for evidence in candidate.evidence
        }
    )
    evidence_count = sum(len(candidate.evidence) for candidate in sorted_candidates[:1])

    answer_confidence = None
    answer_evidence_count = None
    weakest_event_score = None

    if isinstance(result, VQAResult):
        answer_confidence = result.confidence
        answer_evidence_count = len(result.evidence_ids)
    if isinstance(result, TemporalSequence):
        event_scores = [
            event.fusion_score for event in result.events if event.fusion_score is not None
        ]
        weakest_event_score = min(event_scores) if event_scores else None

    duration = None
    if isinstance(result, KISResult):
        duration = result.end_ms - result.start_ms
    elif isinstance(result, TemporalSequence) and result.events:
        duration = max(event.end_ms for event in result.events) - min(
            event.start_ms for event in result.events
        )

    return VerificationContext(
        query_id=query_id,
        task=task,
        target_result_id=target_result_id,
        top_score=top_score,
        second_score=second_score,
        score_margin=score_margin,
        supporting_modalities=modalities,
        evidence_count=evidence_count,
        candidate_duration_ms=duration,
        answer_confidence=answer_confidence,
        answer_evidence_count=answer_evidence_count,
        weakest_event_score=weakest_event_score,
        hard_unknown=preflight.hard_unknown if preflight is not None else 0,
        hard_contradicted=preflight.hard_contradicted if preflight is not None else 0,
    )


def _target_result_id(
    result: KISResult | VQAResult | TemporalSequence,
    ranked_candidates: list[RankedCandidateRegion],
) -> str:
    if isinstance(result, KISResult):
        return result.representative_frame_id
    if isinstance(result, TemporalSequence):
        return "|".join(event.candidate_id for event in result.events) or result.video_id
    if ranked_candidates:
        return ranked_candidates[0].candidate_id
    return "vqa-result"
