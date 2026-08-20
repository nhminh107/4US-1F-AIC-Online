"""Temporal transition validation and scoring."""

from __future__ import annotations

from dataclasses import dataclass

from BackEnd.app.contracts.models import RankedCandidateRegion, TemporalConstraint
from BackEnd.app.trake.config import TrakeAlignerConfig


@dataclass(frozen=True, slots=True)
class Transition:
    """A valid transition between two event candidates."""

    gap_ms: int
    overlap_ms: int
    score: float
    gap_penalty: float
    overlap_penalty: float


def constraint_for(
    before_event_id: str,
    after_event_id: str,
    constraints: list[TemporalConstraint],
) -> TemporalConstraint | None:
    """Return the explicit constraint for one adjacent event pair, if present."""

    for constraint in constraints:
        if constraint.before == before_event_id and constraint.after == after_event_id:
            return constraint
    return None


def event_index_by_id(event_order: list[str]) -> dict[str, int]:
    """Map event IDs to their requested sequence positions."""

    return {event_id: index for index, event_id in enumerate(event_order)}


def is_adjacent_constraint(
    constraint: TemporalConstraint,
    event_index: dict[str, int],
) -> bool:
    return event_index[constraint.after] - event_index[constraint.before] == 1


def non_adjacent_constraints(
    constraints: list[TemporalConstraint],
    event_order: list[str],
) -> list[TemporalConstraint]:
    event_index = event_index_by_id(event_order)
    return [
        constraint
        for constraint in constraints
        if constraint.before in event_index
        and constraint.after in event_index
        and not is_adjacent_constraint(constraint, event_index)
    ]


def effective_min_gap_ms(
    constraint: TemporalConstraint | None,
    config: TrakeAlignerConfig,
) -> int:
    if constraint and constraint.min_gap_ms is not None:
        return constraint.min_gap_ms
    return config.default_min_gap_ms


def effective_max_gap_ms(
    constraint: TemporalConstraint | None,
    config: TrakeAlignerConfig,
) -> int | None:
    if constraint and constraint.max_gap_ms is not None:
        return constraint.max_gap_ms
    return config.default_max_gap_ms


def compute_transition(
    prev: RankedCandidateRegion,
    curr: RankedCandidateRegion,
    *,
    constraint: TemporalConstraint | None,
    config: TrakeAlignerConfig,
) -> Transition | None:
    """Validate and score a temporal transition."""

    return compute_temporal_transition(
        prev_video_id=prev.video_id,
        prev_start_ms=prev.start_ms,
        prev_end_ms=prev.end_ms,
        curr_video_id=curr.video_id,
        curr_start_ms=curr.start_ms,
        curr_end_ms=curr.end_ms,
        constraint=constraint,
        config=config,
    )


def compute_temporal_transition(
    *,
    prev_video_id: str,
    prev_start_ms: int,
    prev_end_ms: int,
    curr_video_id: str,
    curr_start_ms: int,
    curr_end_ms: int,
    constraint: TemporalConstraint | None,
    config: TrakeAlignerConfig,
) -> Transition | None:
    """Validate and score a transition between two time ranges."""

    if prev_video_id != curr_video_id:
        return None

    allow_overlap = constraint.allow_overlap if constraint else config.allow_overlap
    max_gap_ms = effective_max_gap_ms(constraint, config)
    min_gap_ms = effective_min_gap_ms(constraint, config)

    if curr_start_ms < prev_start_ms:
        return None

    overlap_ms = max(0, prev_end_ms - curr_start_ms)
    if overlap_ms > 0:
        if not allow_overlap:
            return None
        if (
            config.overlap_tolerance_ms is not None
            and overlap_ms > config.overlap_tolerance_ms
        ):
            return None

    gap_ms = curr_start_ms - prev_end_ms
    effective_gap_ms = max(0, gap_ms)
    if gap_ms >= 0 and gap_ms < min_gap_ms:
        return None
    if gap_ms < 0 and min_gap_ms > 0:
        return None

    if max_gap_ms is not None and effective_gap_ms > max_gap_ms:
        if config.gap_mode == "hard":
            return None

    gap_penalty = effective_gap_ms * config.gap_penalty_weight
    overlap_penalty = overlap_ms * config.overlap_penalty_weight
    score = config.transition_bonus - gap_penalty - overlap_penalty

    return Transition(
        gap_ms=gap_ms,
        overlap_ms=overlap_ms,
        score=score,
        gap_penalty=gap_penalty,
        overlap_penalty=overlap_penalty,
    )
