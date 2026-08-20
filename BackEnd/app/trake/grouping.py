"""Candidate validation and grouping for TRAKE."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import defaultdict

from BackEnd.app.contracts.models import RankedCandidateRegion
from BackEnd.app.trake.config import TrakeAlignerConfig


@dataclass(slots=True)
class TrakeDiagnostics:
    """Mutable diagnostics collected during one alignment run."""

    num_input_candidates: int = 0
    ignored_missing_event_id_count: int = 0
    ignored_unknown_event_count: int = 0
    rejected_invalid_score_count: int = 0
    rejected_hard_constraint_count: int = 0
    duplicate_candidate_count: int = 0
    num_valid_candidates: int = 0
    num_feasible_videos: int = 0
    num_edges: int = 0
    candidate_checks: int = 0
    num_states_expanded: int = 0
    num_states_pruned: int = 0
    dead_end_state_count: int = 0
    non_adjacent_constraint_reject_count: int = 0
    early_non_adjacent_constraint_reject_count: int = 0
    final_non_adjacent_constraint_reject_count: int = 0
    connectivity_cache_hits: int = 0
    connectivity_cache_misses: int = 0
    pre_pruned_candidates: dict[str, int] = field(default_factory=dict)
    layers: list[dict[str, object]] = field(default_factory=list)
    num_valid_sequences: int = 0
    missing_event_ids: list[str] = field(default_factory=list)
    score_breakdowns: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "num_input_candidates": self.num_input_candidates,
            "ignored_missing_event_id_count": self.ignored_missing_event_id_count,
            "ignored_unknown_event_count": self.ignored_unknown_event_count,
            "rejected_invalid_score_count": self.rejected_invalid_score_count,
            "rejected_hard_constraint_count": self.rejected_hard_constraint_count,
            "duplicate_candidate_count": self.duplicate_candidate_count,
            "num_valid_candidates": self.num_valid_candidates,
            "num_feasible_videos": self.num_feasible_videos,
            "num_edges": self.num_edges,
            "candidate_checks": self.candidate_checks,
            "num_states_expanded": self.num_states_expanded,
            "num_states_pruned": self.num_states_pruned,
            "dead_end_state_count": self.dead_end_state_count,
            "non_adjacent_constraint_reject_count": (
                self.non_adjacent_constraint_reject_count
            ),
            "early_non_adjacent_constraint_reject_count": (
                self.early_non_adjacent_constraint_reject_count
            ),
            "final_non_adjacent_constraint_reject_count": (
                self.final_non_adjacent_constraint_reject_count
            ),
            "connectivity_cache_hits": self.connectivity_cache_hits,
            "connectivity_cache_misses": self.connectivity_cache_misses,
            "pre_pruned_candidates": self.pre_pruned_candidates,
            "layers": self.layers,
            "num_valid_sequences": self.num_valid_sequences,
            "missing_event_ids": self.missing_event_ids,
            "score_breakdowns": self.score_breakdowns,
        }


def candidate_rank_key(
    candidate: RankedCandidateRegion,
) -> tuple[float, int, int, str, str]:
    return (
        -candidate.fusion_score,
        candidate.start_ms,
        candidate.end_ms,
        candidate.video_id,
        candidate.candidate_id,
    )


def validate_candidates(
    candidates: list[RankedCandidateRegion],
    known_event_ids: set[str],
    diagnostics: TrakeDiagnostics,
) -> list[RankedCandidateRegion]:
    """Filter unusable candidates and deduplicate exact same regions."""

    deduplicated: dict[tuple[str, str, int, int], RankedCandidateRegion] = {}
    for candidate in candidates:
        if candidate.event_id is None:
            diagnostics.ignored_missing_event_id_count += 1
            continue
        if candidate.event_id not in known_event_ids:
            diagnostics.ignored_unknown_event_count += 1
            continue
        if not math.isfinite(candidate.fusion_score):
            diagnostics.rejected_invalid_score_count += 1
            continue
        if not candidate.constraint_result.hard_constraints_passed:
            diagnostics.rejected_hard_constraint_count += 1
            continue

        key = (
            candidate.video_id,
            candidate.event_id,
            candidate.start_ms,
            candidate.end_ms,
        )
        existing = deduplicated.get(key)
        if existing is None or candidate_rank_key(candidate) < candidate_rank_key(existing):
            if existing is not None:
                diagnostics.duplicate_candidate_count += 1
            deduplicated[key] = candidate
        else:
            diagnostics.duplicate_candidate_count += 1

    valid_candidates = sorted(
        deduplicated.values(),
        key=lambda item: (
            item.video_id,
            item.event_id or "",
            item.start_ms,
            item.end_ms,
            item.candidate_id,
        ),
    )
    diagnostics.num_valid_candidates = len(valid_candidates)
    return valid_candidates


def group_by_event(
    candidates: list[RankedCandidateRegion],
) -> dict[str, list[RankedCandidateRegion]]:
    grouped: dict[str, list[RankedCandidateRegion]] = defaultdict(list)
    for candidate in candidates:
        if candidate.event_id is not None:
            grouped[candidate.event_id].append(candidate)
    return dict(grouped)


def group_by_video_event(
    candidates: list[RankedCandidateRegion],
    config: TrakeAlignerConfig,
    diagnostics: TrakeDiagnostics | None = None,
) -> dict[str, dict[str, list[RankedCandidateRegion]]]:
    grouped: dict[str, dict[str, list[RankedCandidateRegion]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        if candidate.event_id is not None:
            grouped[candidate.video_id][candidate.event_id].append(candidate)

    for event_map in grouped.values():
        for event_id, event_candidates in event_map.items():
            sorted_candidates = sorted(event_candidates, key=candidate_rank_key)
            if config.max_candidates_per_event_per_video is not None:
                dropped_count = max(
                    0,
                    len(sorted_candidates) - config.max_candidates_per_event_per_video,
                )
                sorted_candidates = sorted_candidates[
                    : config.max_candidates_per_event_per_video
                ]
                if dropped_count:
                    if diagnostics is not None:
                        diagnostics.pre_pruned_candidates[
                            f"{sorted_candidates[0].video_id}:{event_id}"
                        ] = dropped_count
            event_map[event_id] = sorted_candidates
    return {video_id: dict(event_map) for video_id, event_map in grouped.items()}
