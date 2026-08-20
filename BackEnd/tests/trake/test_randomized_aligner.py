"""Deterministic randomized checks for TRAKE temporal alignment."""

from __future__ import annotations

from itertools import product
import math
from random import Random

import pytest

from BackEnd.app.contracts.models import (
    ConstraintResult,
    RankedCandidateRegion,
    TemporalConstraint,
)
from BackEnd.app.trake import TrakeAlignerConfig, TrakeTemporalAligner


EVENT_ORDER = ["E1", "E2", "E3"]
BASE_SEED = 20260818


def make_candidate(
    *,
    case_index: int,
    event_id: str | None,
    video_id: str,
    start_ms: int,
    duration_ms: int,
    score: float,
    suffix: str,
    hard_passed: bool = True,
) -> RankedCandidateRegion:
    return RankedCandidateRegion(
        candidate_id=f"case_{case_index:02d}_{video_id}_{event_id or 'none'}_{suffix}",
        event_id=event_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=start_ms + duration_ms,
        fusion_score=score,
        constraint_result=ConstraintResult(
            hard_constraints_passed=hard_passed,
            negative_constraints_passed=True,
        ),
    )


def build_random_case(case_index: int) -> tuple[
    list[RankedCandidateRegion],
    list[TemporalConstraint],
    str,
]:
    rng = Random(BASE_SEED + case_index)
    winning_video = f"video_{rng.randint(1, 4):02d}"
    max_gap_ms = rng.choice([6_000, 8_000, 10_000, 12_000])
    constraints = [
        TemporalConstraint(before="E1", after="E2", max_gap_ms=max_gap_ms),
        TemporalConstraint(before="E2", after="E3", max_gap_ms=max_gap_ms),
    ]

    candidates: list[RankedCandidateRegion] = []
    base = rng.randint(5, 80) * 1_000
    step = rng.randint(3_000, max_gap_ms)

    # Guaranteed valid high-quality chain.
    for event_index, event_id in enumerate(EVENT_ORDER):
        candidates.append(
            make_candidate(
                case_index=case_index,
                event_id=event_id,
                video_id=winning_video,
                start_ms=base + event_index * step,
                duration_ms=rng.choice([0, 500, 1_000, 1_500]),
                score=10.0 + event_index + case_index / 1000,
                suffix="winner",
            )
        )

    # Random feasible and infeasible noise across videos/events.
    for video_number in range(1, 5):
        video_id = f"video_{video_number:02d}"
        for event_id in EVENT_ORDER:
            for noise_index in range(rng.randint(1, 4)):
                start_ms = rng.randint(0, 180) * 1_000
                duration_ms = rng.choice([0, 500, 1_000, 2_000, 5_000])
                score = rng.uniform(0.05, 7.5)
                candidates.append(
                    make_candidate(
                        case_index=case_index,
                        event_id=event_id,
                        video_id=video_id,
                        start_ms=start_ms,
                        duration_ms=duration_ms,
                        score=score,
                        suffix=f"noise_{noise_index}",
                        hard_passed=rng.random() > 0.08,
                    )
                )

    # Invalid/irrelevant candidates should not affect the result.
    candidates.extend(
        [
            make_candidate(
                case_index=case_index,
                event_id="EX",
                video_id=winning_video,
                start_ms=base,
                duration_ms=1_000,
                score=99.0,
                suffix="unknown_event",
            ),
            make_candidate(
                case_index=case_index,
                event_id=None,
                video_id=winning_video,
                start_ms=base,
                duration_ms=1_000,
                score=99.0,
                suffix="missing_event",
            ),
            make_candidate(
                case_index=case_index,
                event_id="E1",
                video_id=winning_video,
                start_ms=base,
                duration_ms=1_000,
                score=math.nan,
                suffix="nan_score",
            ),
        ]
    )
    rng.shuffle(candidates)
    return candidates, constraints, winning_video


def is_valid_transition(
    previous: RankedCandidateRegion,
    current: RankedCandidateRegion,
    max_gap_ms: int | None,
    *,
    allow_overlap: bool = False,
) -> bool:
    if previous.video_id != current.video_id:
        return False
    if current.start_ms < previous.end_ms and not allow_overlap:
        return False
    effective_gap_ms = max(0, current.start_ms - previous.end_ms)
    return max_gap_ms is None or effective_gap_ms <= max_gap_ms


def brute_force_sequences(
    candidates: list[RankedCandidateRegion],
    constraints: list[TemporalConstraint],
) -> list[tuple[str, ...]]:
    constraint_by_pair = {
        (constraint.before, constraint.after): constraint
        for constraint in constraints
    }
    by_event: dict[str, list[RankedCandidateRegion]] = {event_id: [] for event_id in EVENT_ORDER}
    for candidate in candidates:
        if candidate.event_id not in by_event:
            continue
        if not math.isfinite(candidate.fusion_score):
            continue
        if not candidate.constraint_result.hard_constraints_passed:
            continue
        by_event[candidate.event_id].append(candidate)

    valid_sequences: list[tuple[float, int, int, str, str, tuple[str, ...]]] = []
    for path in product(*(by_event[event_id] for event_id in EVENT_ORDER)):
        event_by_id = {candidate.event_id: candidate for candidate in path}
        if any(
            not is_valid_transition(
                event_by_id[constraint.before],
                event_by_id[constraint.after],
                constraint.max_gap_ms,
                allow_overlap=constraint.allow_overlap,
            )
            for constraint in constraints
        ):
            continue

        first, second, third = path
        total_gap = (second.start_ms - first.end_ms) + (third.start_ms - second.end_ms)
        candidate_ids = tuple(candidate.candidate_id for candidate in path)
        valid_sequences.append(
            (
                -sum(candidate.fusion_score for candidate in path),
                first.start_ms,
                total_gap,
                first.video_id,
                "|".join(candidate_ids),
                candidate_ids,
            )
        )

    return [row[-1] for row in sorted(valid_sequences)]


@pytest.mark.parametrize("case_index", range(30))
def test_randomized_exact_alignment_matches_bruteforce_oracle(case_index: int) -> None:
    candidates, constraints, _winning_video = build_random_case(case_index)
    expected_candidate_ids = brute_force_sequences(candidates, constraints)[0]

    result = TrakeTemporalAligner(
        TrakeAlignerConfig(top_k_sequences=5, beam_width=None)
    ).align(candidates, EVENT_ORDER, constraints)

    assert result.status == "success"
    assert len(result.sequences) <= 5
    assert tuple(event.candidate_id for event in result.sequences[0].events) == (
        expected_candidate_ids
    )
    for sequence in result.sequences:
        assert [event.event_id for event in sequence.events] == EVENT_ORDER
        selected_by_id = {
            candidate.candidate_id: candidate
            for candidate in candidates
            if candidate.event_id in EVENT_ORDER
        }
        assert all(
            selected_by_id[event.candidate_id].video_id == sequence.video_id
            for event in sequence.events
        )


@pytest.mark.parametrize("case_index", range(30))
def test_randomized_exact_top_k_matches_bruteforce_oracle(case_index: int) -> None:
    candidates, constraints, _winning_video = build_random_case(case_index + 100)
    expected_candidate_ids = brute_force_sequences(candidates, constraints)[:5]

    result = TrakeTemporalAligner(
        TrakeAlignerConfig(top_k_sequences=5, beam_width=None)
    ).align(candidates, EVENT_ORDER, constraints)

    actual_candidate_ids = [
        tuple(event.candidate_id for event in sequence.events)
        for sequence in result.sequences
    ]
    assert actual_candidate_ids == expected_candidate_ids[: len(actual_candidate_ids)]


@pytest.mark.parametrize("case_index", range(10))
def test_randomized_non_adjacent_constraints_match_oracle(case_index: int) -> None:
    candidates, constraints, _winning_video = build_random_case(case_index + 200)
    constraints.append(
        TemporalConstraint(before="E1", after="E3", max_gap_ms=20_000)
    )
    expected_candidate_ids = brute_force_sequences(candidates, constraints)[:5]

    result = TrakeTemporalAligner(
        TrakeAlignerConfig(top_k_sequences=5, beam_width=None)
    ).align(candidates, EVENT_ORDER, constraints)

    actual_candidate_ids = [
        tuple(event.candidate_id for event in sequence.events)
        for sequence in result.sequences
    ]
    assert actual_candidate_ids == expected_candidate_ids[: len(actual_candidate_ids)]


@pytest.mark.parametrize("case_index", range(10))
def test_randomized_impossible_non_adjacent_constraint_returns_no_valid_sequence(
    case_index: int,
) -> None:
    candidates, constraints, _winning_video = build_random_case(case_index + 300)
    constraints.append(TemporalConstraint(before="E1", after="E3", max_gap_ms=1))

    assert brute_force_sequences(candidates, constraints) == []
    result = TrakeTemporalAligner(
        TrakeAlignerConfig(top_k_sequences=5, beam_width=None)
    ).align(candidates, EVENT_ORDER, constraints)

    assert result.status == "no_valid_sequence"
