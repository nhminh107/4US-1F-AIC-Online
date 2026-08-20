"""Timestamp-indexed successor lookup for TRAKE transitions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from BackEnd.app.contracts.models import RankedCandidateRegion, TemporalConstraint
from BackEnd.app.trake.config import TrakeAlignerConfig
from BackEnd.app.trake.scoring import (
    Transition,
    compute_transition,
    effective_max_gap_ms,
    effective_min_gap_ms,
)


@dataclass(frozen=True, slots=True)
class SuccessorLookupResult:
    successors: list[tuple[RankedCandidateRegion, Transition]]
    candidate_checks: int


class SuccessorIndex:
    """Lookup candidates in the next event layer by feasible timestamp window."""

    def __init__(
        self,
        by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    ) -> None:
        self._candidates: dict[tuple[str, str], list[RankedCandidateRegion]] = {}
        self._starts: dict[tuple[str, str], list[int]] = {}
        for video_id, event_map in by_video_event.items():
            for event_id, candidates in event_map.items():
                sorted_candidates = sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.start_ms,
                        candidate.end_ms,
                        candidate.candidate_id,
                    ),
                )
                key = (video_id, event_id)
                self._candidates[key] = sorted_candidates
                self._starts[key] = [
                    candidate.start_ms for candidate in sorted_candidates
                ]

    def valid_successors(
        self,
        previous: RankedCandidateRegion,
        target_event_id: str,
        *,
        constraint: TemporalConstraint | None,
        config: TrakeAlignerConfig,
    ) -> SuccessorLookupResult:
        key = (previous.video_id, target_event_id)
        candidates = self._candidates.get(key, [])
        starts = self._starts.get(key, [])
        if not candidates:
            return SuccessorLookupResult(successors=[], candidate_checks=0)

        low_start, high_start = self._time_window(previous, constraint, config)
        left = bisect_left(starts, low_start)
        right = len(candidates)
        if high_start is not None:
            right = bisect_right(starts, high_start)

        window_candidates = candidates[left:right]
        successors: list[tuple[RankedCandidateRegion, Transition]] = []
        for candidate in window_candidates:
            transition = compute_transition(
                previous,
                candidate,
                constraint=constraint,
                config=config,
            )
            if transition is not None:
                successors.append((candidate, transition))

        return SuccessorLookupResult(
            successors=successors,
            candidate_checks=len(window_candidates),
        )

    @staticmethod
    def _time_window(
        previous: RankedCandidateRegion,
        constraint: TemporalConstraint | None,
        config: TrakeAlignerConfig,
    ) -> tuple[int, int | None]:
        allow_overlap = constraint.allow_overlap if constraint else config.allow_overlap
        max_gap_ms = effective_max_gap_ms(constraint, config)
        min_gap_ms = effective_min_gap_ms(constraint, config)

        if min_gap_ms > 0:
            low_start = previous.end_ms + min_gap_ms
        elif allow_overlap:
            if config.overlap_tolerance_ms is None:
                low_start = previous.start_ms
            else:
                low_start = max(previous.start_ms, previous.end_ms - config.overlap_tolerance_ms)
        else:
            low_start = previous.end_ms

        high_start = None
        if max_gap_ms is not None and config.gap_mode == "hard":
            high_start = previous.end_ms + max_gap_ms

        return low_start, high_start
