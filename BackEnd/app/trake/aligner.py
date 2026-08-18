"""Public TRAKE temporal aligner."""

from __future__ import annotations

from BackEnd.app.contracts.models import (
    RankedCandidateRegion,
    TemporalConstraint,
    TemporalEventResult,
    TemporalSequence,
)
from BackEnd.app.trake.config import TrakeAlignerConfig
from BackEnd.app.trake.contracts import TrakeAlignerResult
from BackEnd.app.trake.dp import State, run_dp_beam
from BackEnd.app.trake.grouping import (
    TrakeDiagnostics,
    group_by_event,
    group_by_video_event,
    validate_candidates,
)
from BackEnd.app.trake.policies import sequence_candidate_key, sort_sequences
from BackEnd.app.trake.scoring import (
    compute_temporal_transition,
    non_adjacent_constraints,
)


class TrakeTemporalAligner:
    """Align ranked event candidates into same-video temporal sequences."""

    def __init__(self, config: TrakeAlignerConfig | None = None) -> None:
        self.config = config or TrakeAlignerConfig()

    def align(
        self,
        candidates: list[RankedCandidateRegion],
        event_order: list[str],
        temporal_constraints: list[TemporalConstraint],
    ) -> TrakeAlignerResult:
        diagnostics = TrakeDiagnostics(num_input_candidates=len(candidates))
        if not event_order:
            return TrakeAlignerResult(
                status="invalid_input",
                replan_required=False,
                diagnostics={**diagnostics.as_dict(), "reason": "empty_event_order"},
            )
        duplicate_event_ids = sorted(
            {event_id for event_id in event_order if event_order.count(event_id) > 1}
        )
        if duplicate_event_ids:
            return TrakeAlignerResult(
                status="invalid_input",
                replan_required=False,
                diagnostics={
                    **diagnostics.as_dict(),
                    "reason": "duplicate_event_id",
                    "duplicate_event_ids": duplicate_event_ids,
                },
            )
        normalized_event_order = list(event_order)
        known_event_ids = set(normalized_event_order)
        unknown_constraint_event_ids = sorted(
            {
                event_id
                for constraint in temporal_constraints
                for event_id in (constraint.before, constraint.after)
                if event_id not in known_event_ids
            }
        )
        if unknown_constraint_event_ids:
            return TrakeAlignerResult(
                status="invalid_input",
                replan_required=False,
                diagnostics={
                    **diagnostics.as_dict(),
                    "reason": "unknown_constraint_event",
                    "unknown_constraint_event_ids": unknown_constraint_event_ids,
                },
            )

        valid_candidates = validate_candidates(
            candidates,
            known_event_ids,
            diagnostics,
        )
        by_event = group_by_event(valid_candidates)
        missing_event_ids = [
            event_id
            for event_id in normalized_event_order
            if not by_event.get(event_id)
        ]
        diagnostics.missing_event_ids = missing_event_ids
        if missing_event_ids:
            return TrakeAlignerResult(
                status="insufficient_candidates",
                missing_event_ids=missing_event_ids,
                replan_required=True,
                diagnostics=diagnostics.as_dict(),
            )

        by_video_event = group_by_video_event(
            valid_candidates,
            self.config,
            diagnostics,
        )
        feasible_video_ids = sorted(
            video_id
            for video_id, event_map in by_video_event.items()
            if all(event_map.get(event_id) for event_id in normalized_event_order)
        )
        diagnostics.num_feasible_videos = len(feasible_video_ids)
        if not feasible_video_ids:
            return TrakeAlignerResult(
                status="no_valid_sequence",
                replan_required=True,
                diagnostics=diagnostics.as_dict(),
            )

        states = run_dp_beam(
            feasible_video_ids=feasible_video_ids,
            by_video_event=by_video_event,
            event_order=normalized_event_order,
            temporal_constraints=temporal_constraints,
            config=self.config,
            diagnostics=diagnostics,
        )
        if not states:
            return TrakeAlignerResult(
                status="no_valid_sequence",
                replan_required=True,
                diagnostics=diagnostics.as_dict(),
            )

        sequences = []
        state_by_sequence_key = {}
        for state in states:
            sequence = self._state_to_sequence(state)
            if not self._passes_non_adjacent_constraints(
                sequence,
                temporal_constraints,
                normalized_event_order,
            ):
                diagnostics.final_non_adjacent_constraint_reject_count += 1
                diagnostics.non_adjacent_constraint_reject_count += 1
                continue
            sequences.append(sequence)
            state_by_sequence_key[sequence_candidate_key(sequence)] = state

        sequences = sort_sequences(sequences, self.config)[: self.config.top_k_sequences]
        diagnostics.score_breakdowns = [
            self._state_score_breakdown(
                state_by_sequence_key[sequence_candidate_key(sequence)]
            )
            for sequence in sequences
        ]
        diagnostics.num_valid_sequences = len(sequences)
        if not sequences:
            return TrakeAlignerResult(
                status="no_valid_sequence",
                replan_required=True,
                diagnostics=diagnostics.as_dict(),
            )

        return TrakeAlignerResult(
            status="success",
            sequences=sequences,
            replan_required=False,
            diagnostics=diagnostics.as_dict(),
        )

    def _state_to_sequence(
        self,
        state: State,
    ) -> TemporalSequence:
        events = [
            TemporalEventResult(
                event_id=candidate.event_id or "",
                candidate_id=candidate.candidate_id,
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                fusion_score=candidate.fusion_score,
                evidence_ids=[
                    evidence.entity_id
                    for evidence in candidate.evidence
                ],
            )
            for candidate in state.path
        ]
        return TemporalSequence(
            video_id=state.video_id,
            events=events,
            sequence_score=state.score,
        )

    @staticmethod
    def _state_score_breakdown(state: State) -> dict[str, object]:
        return {
            "candidate_ids": [candidate.candidate_id for candidate in state.path],
            "event_score_sum": state.event_score_sum,
            "transition_score_sum": state.transition_score_sum,
            "gap_penalty": state.gap_penalty,
            "overlap_penalty": state.overlap_penalty,
            "total_gap_ms": state.total_gap_ms,
            "max_gap_ms": state.max_gap_ms,
            "final_sequence_score": state.score,
        }

    def _passes_non_adjacent_constraints(
        self,
        sequence: TemporalSequence,
        temporal_constraints: list[TemporalConstraint],
        event_order: list[str],
    ) -> bool:
        event_by_id = {event.event_id: event for event in sequence.events}
        for constraint in non_adjacent_constraints(temporal_constraints, event_order):
            before = event_by_id[constraint.before]
            after = event_by_id[constraint.after]
            transition = compute_temporal_transition(
                prev_video_id=sequence.video_id,
                prev_start_ms=before.start_ms,
                prev_end_ms=before.end_ms,
                curr_video_id=sequence.video_id,
                curr_start_ms=after.start_ms,
                curr_end_ms=after.end_ms,
                constraint=constraint,
                config=self.config,
            )
            if transition is None:
                return False
        return True
