from __future__ import annotations

from typing import Protocol

from BackEnd.app.retrieval_v2.contracts import (
    CandidateReview,
    MomentBand,
    RetrievalPlan,
    RetryDiagnosis,
    QueryAtom,
)


class CandidateReviewer(Protocol):
    async def review(
        self,
        plan: RetrievalPlan,
        bands: list[MomentBand],
    ) -> list[CandidateReview]: ...


def apply_candidate_reviews(
    bands: list[MomentBand],
    reviews: list[CandidateReview],
    atoms: list[QueryAtom] | None = None,
) -> list[MomentBand]:
    by_band: dict[str, CandidateReview] = {}
    verdict_priority = {"uncertain": 0, "match": 1, "partial": 2, "mismatch": 3}
    for review in reviews:
        current = by_band.get(review.band_id)
        if current is None or (
            verdict_priority[review.verdict], review.confidence
        ) > (
            verdict_priority[current.verdict], current.confidence
        ):
            by_band[review.band_id] = review
    rescored: list[MomentBand] = []
    confirmed_matches: set[str] = set()
    for band in bands:
        review = by_band.get(band.band_id)
        if review is None or review.verdict == "uncertain":
            rescored.append(band)
            continue
        if review.verdict == "mismatch" and review.confidence >= 0.8:
            # A high-confidence visual mismatch is direct negative evidence.
            # It must not survive merely because the reviewer omitted the
            # optional per-atom status map.
            continue
        if review.verdict == "match":
            score = band.score * (1.0 + review.confidence)
            confirmed_matches.add(band.band_id)
        elif review.verdict == "partial":
            score = band.score * (1.0 + 0.2 * review.confidence)
        else:
            score = band.score * (1.0 - review.confidence)
        coverage = dict(band.coverage)
        contradictions = dict(band.contradictions)
        negative_ids = {
            atom.atom_id for atom in (atoms or []) if atom.operator == "MUST_NOT"
        }
        for atom_id, status in review.atom_status.items():
            target = contradictions if atom_id in negative_ids else coverage
            cell = target.get(atom_id)
            if cell is not None:
                target[atom_id] = cell.model_copy(update={"status": status})
        rescored.append(band.model_copy(update={
            "score": max(0.0, score),
            "coverage": coverage,
            "contradictions": contradictions,
        }))
    return sorted(
        rescored,
        key=lambda band: (
            0 if band.band_id in confirmed_matches else 1,
            -band.score,
            band.video_id,
            band.start_ms,
        ),
    )


_DIAGNOSIS_BY_FAILURE = {
    "wrong_video": ("WRONG_VIDEO", "REJECT_VIDEO_AND_BROADEN"),
    "wrong_moment": ("WRONG_MOMENT", "REJECT_BAND_AND_LOCAL_SEARCH"),
    "missing_action": ("MISSING_ACTION", "RETRY_ACTION_PROMPT"),
    "wrong_relation_or_count": ("WRONG_RELATION_OR_COUNT", "VERIFY_RELATION_OR_COUNT"),
    "prompt_too_broad": ("PROMPT_TOO_BROAD", "EXPAND_DISCRIMINATIVE_PROMPT"),
    "correlated_retriever_family": (
        "CORRELATED_RETRIEVER_FAMILY",
        "SWITCH_RETRIEVER_FAMILY",
    ),
}


def diagnose_candidate_reviews(
    reviews: list[CandidateReview],
) -> list[RetryDiagnosis]:
    diagnoses: list[RetryDiagnosis] = []
    for review in reviews:
        if review.verdict != "mismatch" or review.confidence < 0.8:
            continue
        mapping = _DIAGNOSIS_BY_FAILURE.get(review.failure_reason)
        if mapping is None:
            continue
        reason, action = mapping
        weak_atom = next(
            (atom_id for atom_id, status in review.atom_status.items() if status != "PASS"),
            None,
        )
        diagnoses.append(
            RetryDiagnosis(
                reason=reason,
                action=action,
                atom_id=weak_atom,
                video_id=review.video_id,
                band_id=review.band_id,
            )
        )
    return diagnoses


__all__ = [
    "CandidateReviewer",
    "apply_candidate_reviews",
    "diagnose_candidate_reviews",
]
