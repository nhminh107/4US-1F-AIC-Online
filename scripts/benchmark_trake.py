"""Synthetic benchmark for TRAKE exact DP vs DP+Beam pruning."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.contracts.models import ConstraintResult, RankedCandidateRegion
from BackEnd.app.trake import TrakeAlignerConfig, TrakeTemporalAligner


def _candidate(
    event_id: str,
    video_id: str,
    start_ms: int,
    score: float,
    candidate_id: str,
) -> RankedCandidateRegion:
    return RankedCandidateRegion(
        candidate_id=candidate_id,
        event_id=event_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=start_ms + 2_000,
        fusion_score=score,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )


def build_synthetic_candidates() -> list[RankedCandidateRegion]:
    """Create deterministic candidates with several complete chains."""

    candidates: list[RankedCandidateRegion] = []
    for video_index in range(1, 6):
        video_id = f"video_{video_index:02d}"
        base = video_index * 100_000
        for occurrence_index in range(5):
            offset = occurrence_index * 10_000
            score_base = 0.5 + video_index * 0.03 + occurrence_index * 0.01
            candidates.extend(
                [
                    _candidate(
                        "E1",
                        video_id,
                        base + offset,
                        score_base,
                        f"{video_id}_E1_{occurrence_index}",
                    ),
                    _candidate(
                        "E2",
                        video_id,
                        base + offset + 3_000,
                        score_base + 0.02,
                        f"{video_id}_E2_{occurrence_index}",
                    ),
                    _candidate(
                        "E3",
                        video_id,
                        base + offset + 6_000,
                        score_base + 0.04,
                        f"{video_id}_E3_{occurrence_index}",
                    ),
                ]
            )
    return candidates


def build_dead_end_candidates() -> list[RankedCandidateRegion]:
    """Create complete chains plus high-score E1 candidates that cannot finish."""

    candidates = build_synthetic_candidates()
    for video_index in range(1, 6):
        video_id = f"video_{video_index:02d}"
        candidates.append(
            _candidate(
                "E1",
                video_id,
                999_000 + video_index,
                2.0,
                f"{video_id}_E1_dead_high",
            )
        )
    return candidates


def build_dense_candidates() -> list[RankedCandidateRegion]:
    """Create a denser candidate set to exercise successor lookup."""

    candidates: list[RankedCandidateRegion] = []
    for video_index in range(1, 4):
        video_id = f"dense_{video_index:02d}"
        base = video_index * 100_000
        for occurrence_index in range(20):
            offset = occurrence_index * 1_000
            score_base = 0.2 + video_index * 0.05 + occurrence_index * 0.002
            candidates.extend(
                [
                    _candidate(
                        "E1",
                        video_id,
                        base + offset,
                        score_base,
                        f"{video_id}_E1_{occurrence_index}",
                    ),
                    _candidate(
                        "E2",
                        video_id,
                        base + offset + 3_000,
                        score_base + 0.02,
                        f"{video_id}_E2_{occurrence_index}",
                    ),
                    _candidate(
                        "E3",
                        video_id,
                        base + offset + 6_000,
                        score_base + 0.04,
                        f"{video_id}_E3_{occurrence_index}",
                    ),
                ]
            )
    return candidates


def run_benchmark() -> list[dict[str, object]]:
    event_order = ["E1", "E2", "E3"]
    rows: list[dict[str, object]] = []
    fixtures = {
        "baseline": build_synthetic_candidates(),
        "dead_end": build_dead_end_candidates(),
        "dense": build_dense_candidates(),
    }

    modes = [
        ("exact", None, True),
        ("beam_5_no_connectivity", 5, False),
        ("beam_5", 5, True),
        ("beam_10", 10, True),
        ("beam_20", 20, True),
        ("beam_50", 50, True),
        ("beam_100", 100, True),
    ]

    for fixture_name, candidates in fixtures.items():
        exact_top1_candidate_ids: tuple[str, ...] | None = None
        exact_top5_candidate_ids: list[tuple[str, ...]] | None = None
        for mode_name, beam_width, connectivity_pruning in modes:
            config = TrakeAlignerConfig(
                top_k_sequences=5,
                beam_width=beam_width,
                per_video_beam_width=10,
                future_connectivity_pruning=connectivity_pruning,
            )
            aligner = TrakeTemporalAligner(config)
            started_at = perf_counter()
            result = aligner.align(candidates, event_order, [])
            runtime_ms = (perf_counter() - started_at) * 1000
            top_candidate_ids = [
                tuple(event.candidate_id for event in sequence.events)
                for sequence in result.sequences
            ]
            if beam_width is None:
                exact_top1_candidate_ids = (
                    top_candidate_ids[0] if top_candidate_ids else ()
                )
                exact_top5_candidate_ids = top_candidate_ids

            rows.append(
                {
                    "fixture": fixture_name,
                    "mode": mode_name,
                    "top1_matches_exact": (
                        bool(top_candidate_ids)
                        and top_candidate_ids[0] == exact_top1_candidate_ids
                    ),
                    "top5_matches_exact": top_candidate_ids == exact_top5_candidate_ids,
                    "status": result.status,
                    "num_sequences": len(result.sequences),
                    "states_expanded": result.diagnostics["num_states_expanded"],
                    "states_pruned": result.diagnostics["num_states_pruned"],
                    "dead_end_states": result.diagnostics["dead_end_state_count"],
                    "candidate_checks": result.diagnostics["candidate_checks"],
                    "runtime_ms": round(runtime_ms, 3),
                }
            )
    return rows


def main() -> None:
    for row in run_benchmark():
        print(row)


if __name__ == "__main__":
    main()
