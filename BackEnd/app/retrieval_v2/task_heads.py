"""Deterministic task-specific selection and validation helpers.

These helpers deliberately consume already retrieved and reviewed evidence. They
do not search, mutate pipeline state, or infer facts that are absent from the
provided evidence IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.retrieval_v2.contracts import MomentBand


RepresentativeStrategy = Literal[
    "nearest_peak",
    "earliest",
    "latest",
    "highest_score",
]


@dataclass(frozen=True, slots=True)
class OfficialFrame:
    evidence_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    event_id: str | None = None
    official: bool = True
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.video_id:
            raise ValueError("evidence_id and video_id must not be empty")
        if self.frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("frame_idx and timestamp_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class KISMomentSelection:
    band_id: str
    video_id: str
    event_id: str | None
    frame: OfficialFrame
    score: float


@dataclass(frozen=True, slots=True)
class KISSequenceSelection:
    video_id: str
    items: tuple[KISMomentSelection, ...]
    score: float


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    evidence_id: str
    answer: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class GroundedQAResult:
    answer: str
    status: Literal["answered", "uncertain"]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TRAKEValidationResult:
    valid: bool
    frames: tuple[OfficialFrame, ...]
    reasons: tuple[str, ...]


def select_kis_moments(
    bands: Sequence[MomentBand],
    accepted_band_ids: set[str] | frozenset[str],
    official_frames: Sequence[OfficialFrame],
    *,
    representative_strategy: RepresentativeStrategy = "nearest_peak",
    limit: int = 100,
) -> list[KISMomentSelection]:
    """Select one official representative frame for every accepted band."""

    if limit < 1:
        return []
    _validate_strategy(representative_strategy)
    selections: list[KISMomentSelection] = []
    for band in bands:
        if band.band_id not in accepted_band_ids:
            continue
        candidates = _frames_for_band(band, official_frames)
        if not candidates:
            continue
        frame = _representative_frame(candidates, band, representative_strategy)
        selections.append(
            KISMomentSelection(
                band_id=band.band_id,
                video_id=band.video_id,
                event_id=band.event_id,
                frame=frame,
                score=band.score,
            )
        )
    return sorted(
        selections,
        key=lambda item: (-item.score, item.video_id, item.frame.frame_idx, item.band_id),
    )[:limit]


def select_kis_sequences(
    bands: Sequence[MomentBand],
    accepted_band_ids: set[str] | frozenset[str],
    official_frames: Sequence[OfficialFrame],
    event_ids: Sequence[str],
    *,
    temporal_constraints: Sequence[TemporalConstraint] = (),
    representative_strategy: RepresentativeStrategy = "nearest_peak",
    limit: int = 100,
) -> list[KISSequenceSelection]:
    """Build complete, accepted, same-video KIS sequences in event order."""

    if not event_ids or len(set(event_ids)) != len(event_ids) or limit < 1:
        return []
    _validate_strategy(representative_strategy)
    by_video: dict[str, dict[str, list[MomentBand]]] = {}
    expected = set(event_ids)
    for band in bands:
        if (
            band.band_id in accepted_band_ids
            and band.event_id in expected
        ):
            by_video.setdefault(band.video_id, {}).setdefault(
                band.event_id, []
            ).append(band)

    results: list[KISSequenceSelection] = []
    for video_id, by_event in sorted(by_video.items()):
        if any(not by_event.get(event_id) for event_id in event_ids):
            continue
        event_options = [
            sorted(by_event[event_id], key=lambda band: (-band.score, band.start_ms, band.band_id))
            for event_id in event_ids
        ]
        for sequence in product(*event_options):
            if not _bands_satisfy_temporal(sequence, event_ids, temporal_constraints):
                continue
            items: list[KISMomentSelection] = []
            for band in sequence:
                candidates = _frames_for_band(
                    band,
                    official_frames,
                    require_matching_event=True,
                )
                if not candidates:
                    break
                frame = _representative_frame(candidates, band, representative_strategy)
                items.append(
                    KISMomentSelection(
                        band_id=band.band_id,
                        video_id=video_id,
                        event_id=band.event_id,
                        frame=frame,
                        score=band.score,
                    )
                )
            if len(items) == len(event_ids):
                results.append(
                    KISSequenceSelection(
                        video_id=video_id,
                        items=tuple(items),
                        score=sum(item.score for item in items),
                    )
                )
    return sorted(
        results,
        key=lambda sequence: (
            -sequence.score,
            sequence.video_id,
            tuple(item.frame.frame_idx for item in sequence.items),
        ),
    )[:limit]


def aggregate_grounded_answer(
    claims: Sequence[AnswerClaim],
    allowed_evidence_ids: set[str] | frozenset[str],
) -> GroundedQAResult:
    """Return an answer only when all grounded, non-empty claims agree."""

    grounded: list[tuple[AnswerClaim, str]] = []
    for claim in claims:
        if claim.evidence_id not in allowed_evidence_ids:
            continue
        answer = _normalize_answer(claim.answer)
        if answer:
            grounded.append((claim, answer))

    evidence_ids = tuple(dict.fromkeys(claim.evidence_id for claim, _ in grounded))
    if not grounded:
        return GroundedQAResult(
            answer="uncertain",
            status="uncertain",
            evidence_ids=evidence_ids,
        )

    groups: dict[str, list[tuple[AnswerClaim, str]]] = {}
    for claim, answer in grounded:
        groups.setdefault(answer.casefold(), []).append((claim, answer))
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            -sum(max(0.05, claim.confidence) for claim, _ in group),
            -max(claim.confidence for claim, _ in group),
            group[0][1].casefold(),
        ),
    )
    best = ranked_groups[0]
    best_weight = sum(max(0.05, claim.confidence) for claim, _ in best)
    total_weight = sum(
        max(0.05, claim.confidence)
        for group in ranked_groups
        for claim, _ in group
    )
    second_weight = (
        sum(max(0.05, claim.confidence) for claim, _ in ranked_groups[1])
        if len(ranked_groups) > 1
        else 0.0
    )
    if best_weight / total_weight < 0.6 or best_weight <= second_weight * 1.25:
        return GroundedQAResult(
            answer="uncertain",
            status="uncertain",
            evidence_ids=evidence_ids,
        )
    _, chosen_answer = min(
        best,
        key=lambda item: (-item[0].confidence, item[0].evidence_id, item[1]),
    )
    return GroundedQAResult(
        answer=chosen_answer[:100].rstrip(),
        status="answered",
        evidence_ids=evidence_ids,
    )


def validate_trake_sequence(
    frames: Sequence[OfficialFrame],
    event_ids: Sequence[str],
    *,
    temporal_constraints: Sequence[TemporalConstraint] = (),
) -> TRAKEValidationResult:
    """Validate a TRAKE sequence without repairing or guessing missing events."""

    reasons: list[str] = []
    if len(set(event_ids)) != len(event_ids):
        reasons.append("DUPLICATE_EVENT_ID")

    expected = set(event_ids)
    extras = sorted(
        {frame.event_id for frame in frames if frame.event_id not in expected},
        key=lambda value: "" if value is None else value,
    )
    for event_id in extras:
        reasons.append(f"EXTRA_EVENT:{event_id if event_id is not None else 'NONE'}")

    ordered: list[OfficialFrame] = []
    for event_id in event_ids:
        matching = [frame for frame in frames if frame.event_id == event_id]
        if not matching:
            reasons.append(f"MISSING_EVENT:{event_id}")
            continue
        if len(matching) != 1:
            reasons.append(f"EVENT_FRAME_COUNT:{event_id}")
            continue
        frame = matching[0]
        ordered.append(frame)
        if not frame.official:
            reasons.append(f"UNOFFICIAL_FRAME:{event_id}")

    sequence_videos = {frame.video_id for frame in frames if frame.event_id in expected}
    if len(sequence_videos) > 1:
        reasons.append("CROSS_VIDEO_SEQUENCE")

    if len(ordered) == len(event_ids):
        if any(
            ordered[index].frame_idx >= ordered[index + 1].frame_idx
            for index in range(len(ordered) - 1)
        ):
            reasons.append("NON_INCREASING_FRAME_INDEX")
        reasons.extend(
            _temporal_frame_violations(ordered, temporal_constraints)
        )

    return TRAKEValidationResult(
        valid=not reasons,
        frames=tuple(ordered),
        reasons=tuple(reasons),
    )


def _frames_for_band(
    band: MomentBand,
    frames: Sequence[OfficialFrame],
    *,
    require_matching_event: bool = False,
) -> list[OfficialFrame]:
    return [
        frame
        for frame in frames
        if frame.official
        and frame.video_id == band.video_id
        and band.start_ms <= frame.timestamp_ms <= band.end_ms
        and (
            not require_matching_event
            or frame.event_id == band.event_id
        )
    ]


def _representative_frame(
    frames: Sequence[OfficialFrame],
    band: MomentBand,
    strategy: RepresentativeStrategy,
) -> OfficialFrame:
    if strategy == "earliest":
        key = lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.evidence_id)
    elif strategy == "latest":
        key = lambda frame: (-frame.timestamp_ms, -frame.frame_idx, frame.evidence_id)
    elif strategy == "highest_score":
        key = lambda frame: (-frame.score, abs(frame.timestamp_ms - band.peak_ms), frame.frame_idx)
    else:
        key = lambda frame: (abs(frame.timestamp_ms - band.peak_ms), frame.frame_idx, frame.evidence_id)
    return min(frames, key=key)


def _validate_strategy(strategy: str) -> None:
    if strategy not in {"nearest_peak", "earliest", "latest", "highest_score"}:
        raise ValueError(f"unsupported representative strategy: {strategy}")


def _bands_satisfy_temporal(
    bands: Sequence[MomentBand],
    event_ids: Sequence[str],
    constraints: Sequence[TemporalConstraint],
) -> bool:
    by_event = dict(zip(event_ids, bands, strict=True))
    constraints_by_pair = {
        (constraint.before, constraint.after): constraint
        for constraint in constraints
    }
    for index in range(len(event_ids) - 1):
        before_id = event_ids[index]
        after_id = event_ids[index + 1]
        before = by_event[before_id]
        after = by_event[after_id]
        constraint = constraints_by_pair.get((before_id, after_id))
        if after.start_ms < before.end_ms and (
            constraint is None or not constraint.allow_overlap
        ):
            return False
    for constraint in constraints:
        before = by_event.get(constraint.before)
        after = by_event.get(constraint.after)
        if before is None or after is None:
            continue
        gap_ms = after.start_ms - before.end_ms
        if gap_ms < 0 and not constraint.allow_overlap:
            return False
        if constraint.min_gap_ms is not None and gap_ms < constraint.min_gap_ms:
            return False
        if constraint.max_gap_ms is not None and gap_ms > constraint.max_gap_ms:
            return False
    return True


def _temporal_frame_violations(
    frames: Sequence[OfficialFrame],
    constraints: Sequence[TemporalConstraint],
) -> list[str]:
    by_event = {frame.event_id: frame for frame in frames}
    reasons: list[str] = []
    for constraint in constraints:
        before = by_event.get(constraint.before)
        after = by_event.get(constraint.after)
        if before is None or after is None:
            continue
        gap_ms = after.timestamp_ms - before.timestamp_ms
        if gap_ms < 0 and not constraint.allow_overlap:
            reasons.append(
                f"OVERLAP_NOT_ALLOWED:{constraint.before}:{constraint.after}"
            )
        if constraint.min_gap_ms is not None and gap_ms < constraint.min_gap_ms:
            reasons.append(
                f"MIN_GAP_VIOLATION:{constraint.before}:{constraint.after}"
            )
        if constraint.max_gap_ms is not None and gap_ms > constraint.max_gap_ms:
            reasons.append(
                f"MAX_GAP_VIOLATION:{constraint.before}:{constraint.after}"
            )
    return reasons


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.split())


__all__ = [
    "AnswerClaim",
    "GroundedQAResult",
    "KISMomentSelection",
    "KISSequenceSelection",
    "OfficialFrame",
    "RepresentativeStrategy",
    "TRAKEValidationResult",
    "aggregate_grounded_answer",
    "select_kis_moments",
    "select_kis_sequences",
    "validate_trake_sequence",
]
