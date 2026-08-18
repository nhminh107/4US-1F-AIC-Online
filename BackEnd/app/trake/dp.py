"""Dynamic programming and beam pruning for TRAKE."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from BackEnd.app.contracts.models import RankedCandidateRegion, TemporalConstraint
from BackEnd.app.trake.config import TrakeAlignerConfig
from BackEnd.app.trake.grouping import TrakeDiagnostics
from BackEnd.app.trake.policies import candidate_identity_key
from BackEnd.app.trake.scoring import (
    compute_transition,
    constraint_for,
    non_adjacent_constraints,
)
from BackEnd.app.trake.successors import SuccessorIndex


@dataclass(frozen=True, slots=True)
class State:
    """One partial temporal sequence ending at the last candidate in path."""

    video_id: str
    path: tuple[RankedCandidateRegion, ...]
    score: float
    beam_score: float
    event_score_sum: float
    transition_score_sum: float = 0.0
    total_gap_ms: int = 0
    max_gap_ms: int = 0
    gap_penalty: float = 0.0
    overlap_penalty: float = 0.0


def run_dp_beam(
    *,
    feasible_video_ids: list[str],
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    event_order: list[str],
    temporal_constraints: list[TemporalConstraint],
    config: TrakeAlignerConfig,
    diagnostics: TrakeDiagnostics,
) -> list[State]:
    """Return final states after exact DP or beam-pruned DP."""

    successor_index = SuccessorIndex(by_video_event)
    connectivity_memo: dict[tuple[str, str | None, str, int, int, int], bool] = {}
    states = _initial_states(
        feasible_video_ids,
        by_video_event,
        event_order[0],
        event_order,
        config,
    )
    states = _filter_dead_end_states(
        states,
        current_event_index=0,
        by_video_event=by_video_event,
        event_order=event_order,
        temporal_constraints=temporal_constraints,
        config=config,
        diagnostics=diagnostics,
        memo=connectivity_memo,
    )
    states = prune_states(states, config, diagnostics)

    for previous_event_id, current_event_id in zip(event_order, event_order[1:]):
        states_before_layer = len(states)
        next_states: list[State] = []
        candidate_checks = 0
        constraint = constraint_for(
            previous_event_id,
            current_event_id,
            temporal_constraints,
        )
        for state in states:
            previous_candidate = state.path[-1]
            lookup = successor_index.valid_successors(
                previous_candidate,
                current_event_id,
                constraint=constraint,
                config=config,
            )
            candidate_checks += lookup.candidate_checks
            diagnostics.candidate_checks += lookup.candidate_checks
            diagnostics.num_states_expanded += lookup.candidate_checks
            for current_candidate, transition in lookup.successors:
                diagnostics.num_edges += 1
                event_score = current_candidate.fusion_score * config.event_score_weight
                current_event_index = event_order.index(current_event_id)
                extended_state = _extend_state(
                    state,
                    current_candidate,
                    transition,
                    event_score,
                    current_event_index,
                    by_video_event,
                    event_order,
                    config=config,
                )
                if not _passes_ready_non_adjacent_constraints(
                    extended_state,
                    temporal_constraints,
                    event_order,
                    config,
                ):
                    diagnostics.early_non_adjacent_constraint_reject_count += 1
                    diagnostics.non_adjacent_constraint_reject_count += 1
                    continue
                next_states.append(extended_state)
        if not next_states:
            diagnostics.layers.append(
                _layer_diagnostics(
                    previous_event_id,
                    current_event_id,
                    states_before_layer,
                    candidate_checks,
                    0,
                    0,
                    0,
                )
            )
            return []
        before_dead_end_filter = len(next_states)
        next_states = _filter_dead_end_states(
            next_states,
            current_event_index=event_order.index(current_event_id),
            by_video_event=by_video_event,
            event_order=event_order,
            temporal_constraints=temporal_constraints,
            config=config,
            diagnostics=diagnostics,
            memo=connectivity_memo,
        )
        dead_end_pruned = before_dead_end_filter - len(next_states)
        if not next_states:
            diagnostics.layers.append(
                _layer_diagnostics(
                    previous_event_id,
                    current_event_id,
                    states_before_layer,
                    candidate_checks,
                    before_dead_end_filter,
                    dead_end_pruned,
                    0,
                )
            )
            return []
        before_beam_prune = len(next_states)
        states = prune_states(next_states, config, diagnostics)
        beam_pruned = before_beam_prune - len(states)
        diagnostics.layers.append(
            _layer_diagnostics(
                previous_event_id,
                current_event_id,
                states_before_layer,
                candidate_checks,
                before_dead_end_filter,
                dead_end_pruned,
                beam_pruned,
                states_out=len(states),
            )
        )

    return states


def _extend_state(
    state: State,
    current_candidate: RankedCandidateRegion,
    transition,
    event_score: float,
    current_event_index: int,
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    event_order: list[str],
    *,
    config: TrakeAlignerConfig,
) -> State:
    score = state.score + event_score + transition.score
    return State(
        video_id=state.video_id,
        path=(*state.path, current_candidate),
        score=score,
        beam_score=score
        + _future_upper_bound(
            state.video_id,
            current_event_index + 1,
            by_video_event,
            event_order,
            config,
        ),
        event_score_sum=state.event_score_sum + event_score,
        transition_score_sum=state.transition_score_sum + transition.score,
        total_gap_ms=state.total_gap_ms + max(0, transition.gap_ms),
        max_gap_ms=max(state.max_gap_ms, max(0, transition.gap_ms)),
        gap_penalty=state.gap_penalty + transition.gap_penalty,
        overlap_penalty=state.overlap_penalty + transition.overlap_penalty,
    )


def _layer_diagnostics(
    previous_event_id: str,
    current_event_id: str,
    states_in: int,
    candidate_checks: int,
    valid_edges: int,
    dead_end_pruned: int,
    beam_pruned: int,
    *,
    states_out: int = 0,
) -> dict[str, object]:
    return {
        "transition": f"{previous_event_id}->{current_event_id}",
        "states_in": states_in,
        "candidate_checks": candidate_checks,
        "valid_edges": valid_edges,
        "dead_end_pruned": dead_end_pruned,
        "beam_pruned": beam_pruned,
        "states_out": states_out,
    }


def _passes_ready_non_adjacent_constraints(
    state: State,
    temporal_constraints: list[TemporalConstraint],
    event_order: list[str],
    config: TrakeAlignerConfig,
) -> bool:
    candidate_by_event = {
        candidate.event_id: candidate
        for candidate in state.path
        if candidate.event_id is not None
    }
    current_event_id = state.path[-1].event_id
    for constraint in non_adjacent_constraints(temporal_constraints, event_order):
        if constraint.after != current_event_id:
            continue
        before = candidate_by_event.get(constraint.before)
        after = candidate_by_event.get(constraint.after)
        if before is None or after is None:
            continue
        if compute_transition(
            before,
            after,
            constraint=constraint,
            config=config,
        ) is None:
            return False
    return True


def prune_states(
    states: list[State],
    config: TrakeAlignerConfig,
    diagnostics: TrakeDiagnostics,
) -> list[State]:
    states = sorted(states, key=beam_rank_key)
    if config.beam_width is None:
        return states

    states = _limit_paths_per_terminal_node(states, config, diagnostics)
    per_video_limited: list[State] = []
    by_video: dict[str, list[State]] = defaultdict(list)
    for state in states:
        by_video[state.video_id].append(state)

    for video_id in sorted(by_video):
        video_states = by_video[video_id]
        if config.per_video_beam_width is not None:
            video_states = video_states[: config.per_video_beam_width]
        per_video_limited.extend(video_states)

    pruned = sorted(per_video_limited, key=beam_rank_key)[: config.beam_width]
    diagnostics.num_states_pruned += max(0, len(states) - len(pruned))
    return pruned


def beam_rank_key(state: State) -> tuple[float, float, int, int, str, str]:
    first_start = state.path[0].start_ms
    candidate_key = "||".join(candidate_identity_key(candidate) for candidate in state.path)
    return (
        -state.beam_score,
        -state.score,
        first_start,
        state.total_gap_ms,
        state.video_id,
        candidate_key,
    )


def _initial_states(
    feasible_video_ids: list[str],
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    first_event_id: str,
    event_order: list[str],
    config: TrakeAlignerConfig,
) -> list[State]:
    states: list[State] = []
    for video_id in feasible_video_ids:
        for candidate in by_video_event[video_id][first_event_id]:
            event_score = candidate.fusion_score * config.event_score_weight
            states.append(
                State(
                    video_id=video_id,
                    path=(candidate,),
                    score=event_score,
                    beam_score=event_score
                    + _future_upper_bound(
                        video_id,
                        1,
                        by_video_event,
                        event_order,
                        config,
                    ),
                    event_score_sum=event_score,
                )
            )
    return states


def _limit_paths_per_terminal_node(
    states: list[State],
    config: TrakeAlignerConfig,
    diagnostics: TrakeDiagnostics,
) -> list[State]:
    if config.paths_per_node < 1:
        return states

    kept: list[State] = []
    by_terminal_candidate: dict[str, list[State]] = defaultdict(list)
    for state in states:
        by_terminal_candidate[candidate_identity_key(state.path[-1])].append(state)

    for candidate_id in sorted(by_terminal_candidate):
        candidate_states = sorted(by_terminal_candidate[candidate_id], key=beam_rank_key)
        kept.extend(candidate_states[: config.paths_per_node])

    diagnostics.num_states_pruned += max(0, len(states) - len(kept))
    return sorted(kept, key=beam_rank_key)


def _filter_dead_end_states(
    states: list[State],
    *,
    current_event_index: int,
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    event_order: list[str],
    temporal_constraints: list[TemporalConstraint],
    config: TrakeAlignerConfig,
    diagnostics: TrakeDiagnostics,
    memo: dict[tuple[str, str | None, str, int, int, int], bool],
) -> list[State]:
    if not config.future_connectivity_pruning or current_event_index >= len(event_order) - 1:
        return states

    filtered = [
        state
        for state in states
        if _can_complete_from(
            state.path[-1],
            current_event_index,
            by_video_event,
            event_order,
            temporal_constraints,
            config,
            memo,
            diagnostics,
        )
    ]
    diagnostics.dead_end_state_count += len(states) - len(filtered)
    diagnostics.num_states_pruned += len(states) - len(filtered)
    return filtered


def _can_complete_from(
    candidate: RankedCandidateRegion,
    current_event_index: int,
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    event_order: list[str],
    temporal_constraints: list[TemporalConstraint],
    config: TrakeAlignerConfig,
    memo: dict[tuple[str, str | None, str, int, int, int], bool],
    diagnostics: TrakeDiagnostics,
) -> bool:
    if current_event_index >= len(event_order) - 1:
        return True

    memo_key = (
        candidate.video_id,
        candidate.event_id,
        candidate.candidate_id,
        candidate.start_ms,
        candidate.end_ms,
        current_event_index,
    )
    if memo_key in memo:
        diagnostics.connectivity_cache_hits += 1
        return memo[memo_key]
    diagnostics.connectivity_cache_misses += 1

    current_event_id = event_order[current_event_index]
    next_event_id = event_order[current_event_index + 1]
    constraint = constraint_for(current_event_id, next_event_id, temporal_constraints)
    next_candidates = by_video_event[candidate.video_id].get(next_event_id, [])
    for next_candidate in next_candidates:
        transition = compute_transition(
            candidate,
            next_candidate,
            constraint=constraint,
            config=config,
        )
        if transition is None:
            continue
        if _can_complete_from(
            next_candidate,
            current_event_index + 1,
            by_video_event,
            event_order,
            temporal_constraints,
            config,
            memo,
            diagnostics,
        ):
            memo[memo_key] = True
            return True

    memo[memo_key] = False
    return False


def _future_upper_bound(
    video_id: str,
    next_event_index: int,
    by_video_event: dict[str, dict[str, list[RankedCandidateRegion]]],
    event_order: list[str],
    config: TrakeAlignerConfig,
) -> float:
    if not event_order:
        return 0.0

    event_map = by_video_event.get(video_id, {})
    upper_bound = 0.0
    for event_id in event_order[next_event_index:]:
        event_candidates = event_map.get(event_id, [])
        if event_candidates:
            upper_bound += max(
                candidate.fusion_score for candidate in event_candidates
            ) * config.event_score_weight
    return upper_bound * config.future_score_weight
