import math

from BackEnd.app.contracts.models import (
    CandidateEvidence,
    ConstraintResult,
    RankedCandidateRegion,
    TemporalConstraint,
)
from BackEnd.app.trake import TrakeAlignerConfig, TrakeTemporalAligner


def candidate(
    event_id: str | None,
    video_id: str,
    start_ms: int,
    end_ms: int,
    score: float,
    candidate_id: str,
    *,
    evidence_ids: list[str] | None = None,
) -> RankedCandidateRegion:
    evidence = [
        CandidateEvidence(
            source="clip",
            entity_type="clip",
            entity_id=evidence_id,
            rank=index + 1,
            raw_score=score,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for index, evidence_id in enumerate(evidence_ids or [])
    ]
    return RankedCandidateRegion(
        candidate_id=candidate_id,
        event_id=event_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=end_ms,
        fusion_score=score,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
        evidence=evidence,
    )


def aligner(**config_overrides) -> TrakeTemporalAligner:
    return TrakeTemporalAligner(TrakeAlignerConfig(**config_overrides))


def test_returns_success_for_simple_valid_chain() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 20_000, 22_000, 0.9, "A_E2"),
        candidate("E3", "video_A", 30_000, 32_000, 0.7, "A_E3"),
    ]

    result = aligner().align(candidates, ["E1", "E2", "E3"], [])

    assert result.status == "success"
    assert result.replan_required is False
    assert [event.candidate_id for event in result.sequences[0].events] == [
        "A_E1",
        "A_E2",
        "A_E3",
    ]
    assert result.sequences[0].video_id == "video_A"


def test_duplicate_event_order_is_invalid_input() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 20_000, 22_000, 0.9, "A_E2"),
    ]

    result = aligner().align(candidates, ["E1", "E2", "E1"], [])

    assert result.status == "invalid_input"
    assert result.diagnostics["reason"] == "duplicate_event_id"
    assert result.diagnostics["duplicate_event_ids"] == ["E1"]


def test_constraint_referencing_unknown_event_is_invalid_input() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 20_000, 22_000, 0.9, "A_E2"),
    ]

    result = aligner().align(
        candidates,
        ["E1", "E2"],
        [TemporalConstraint(before="E1", after="E9")],
    )

    assert result.status == "invalid_input"
    assert result.diagnostics["reason"] == "unknown_constraint_event"
    assert result.diagnostics["unknown_constraint_event_ids"] == ["E9"]


def test_missing_event_requests_replan() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E3", "video_A", 30_000, 32_000, 0.7, "A_E3"),
    ]

    result = aligner().align(candidates, ["E1", "E2", "E3"], [])

    assert result.status == "insufficient_candidates"
    assert result.missing_event_ids == ["E2"]
    assert result.replan_required is True
    assert result.sequences == []


def test_candidates_from_different_videos_do_not_form_sequence() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E2", "video_B", 20_000, 22_000, 0.9, "B_E2"),
    ]

    result = aligner().align(candidates, ["E1", "E2"], [])

    assert result.status == "no_valid_sequence"
    assert result.replan_required is True
    assert result.sequences == []


def test_wrong_temporal_order_is_invalid() -> None:
    candidates = [
        candidate("E1", "video_A", 30_000, 32_000, 0.8, "A_E1_late"),
        candidate("E2", "video_A", 20_000, 22_000, 0.9, "A_E2_early"),
    ]

    result = aligner().align(candidates, ["E1", "E2"], [])

    assert result.status == "no_valid_sequence"
    assert result.sequences == []


def test_max_gap_constraint_is_enforced() -> None:
    candidates = [
        candidate("E1", "video_A", 0, 10_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 100_000, 110_000, 0.9, "A_E2"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E2", max_gap_ms=60_000)]

    result = aligner().align(candidates, ["E1", "E2"], constraints)

    assert result.status == "no_valid_sequence"
    assert result.sequences == []


def test_non_adjacent_temporal_constraint_rejects_completed_sequence() -> None:
    candidates = [
        candidate("E1", "video_A", 0, 1_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 4_000, 5_000, 0.9, "A_E2"),
        candidate("E3", "video_A", 8_000, 9_000, 0.7, "A_E3"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E3", max_gap_ms=5_000)]

    result = aligner().align(candidates, ["E1", "E2", "E3"], constraints)

    assert result.status == "no_valid_sequence"
    assert result.replan_required is True
    assert result.diagnostics["non_adjacent_constraint_reject_count"] == 1
    assert result.diagnostics["early_non_adjacent_constraint_reject_count"] == 1
    assert result.diagnostics["final_non_adjacent_constraint_reject_count"] == 0


def test_non_adjacent_temporal_constraint_allows_valid_completed_sequence() -> None:
    candidates = [
        candidate("E1", "video_A", 0, 1_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 2_000, 3_000, 0.9, "A_E2"),
        candidate("E3", "video_A", 4_000, 5_000, 0.7, "A_E3"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E3", max_gap_ms=5_000)]

    result = aligner().align(candidates, ["E1", "E2", "E3"], constraints)

    assert result.status == "success"


def test_non_adjacent_constraint_uses_explicit_overlap_policy_over_global_default() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 15_000, 16_000, 0.8, "A_E2"),
        candidate("E3", "video_A", 18_000, 30_000, 0.8, "A_E3"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E3", allow_overlap=False)]

    result = aligner(allow_overlap=True).align(candidates, ["E1", "E2", "E3"], constraints)

    assert result.status == "no_valid_sequence"


def test_non_adjacent_constraint_uses_default_max_gap_when_constraint_omits_gap() -> None:
    candidates = [
        candidate("E1", "video_A", 0, 1_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 5_000, 6_000, 0.8, "A_E2"),
        candidate("E3", "video_A", 7_000, 8_000, 0.8, "A_E3"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E3")]

    result = aligner(default_max_gap_ms=5_000).align(
        candidates,
        ["E1", "E2", "E3"],
        constraints,
    )

    assert result.status == "no_valid_sequence"


def test_overlap_policy_rejects_overlap_by_default_and_allows_when_configured() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 18_000, 30_000, 0.9, "A_E2"),
    ]

    rejected = aligner().align(candidates, ["E1", "E2"], [])
    allowed = aligner().align(
        candidates,
        ["E1", "E2"],
        [TemporalConstraint(before="E1", after="E2", allow_overlap=True)],
    )

    assert rejected.status == "no_valid_sequence"
    assert allowed.status == "success"


def test_overlap_tolerance_limits_allowed_overlap() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 18_000, 30_000, 0.9, "A_E2"),
    ]
    constraints = [TemporalConstraint(before="E1", after="E2", allow_overlap=True)]

    within_tolerance = aligner(overlap_tolerance_ms=2_000).align(
        candidates,
        ["E1", "E2"],
        constraints,
    )
    exceeds_tolerance = aligner(overlap_tolerance_ms=1_999).align(
        candidates,
        ["E1", "E2"],
        constraints,
    )

    assert within_tolerance.status == "success"
    assert exceeds_tolerance.status == "no_valid_sequence"


def test_min_gap_is_enforced_for_non_overlapping_transitions() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 21_000, 30_000, 0.9, "A_E2"),
    ]

    result = aligner(default_min_gap_ms=2_000).align(
        candidates,
        ["E1", "E2"],
        [],
    )

    assert result.status == "no_valid_sequence"


def test_constraint_specific_min_gap_overrides_default_gap() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 21_000, 30_000, 0.9, "A_E2"),
    ]

    result = aligner(default_min_gap_ms=0).align(
        candidates,
        ["E1", "E2"],
        [TemporalConstraint(before="E1", after="E2", min_gap_ms=2_000)],
    )

    assert result.status == "no_valid_sequence"


def test_non_adjacent_constraint_specific_min_gap_is_enforced() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 20_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 21_000, 22_000, 0.8, "A_E2"),
        candidate("E3", "video_A", 21_000, 30_000, 0.9, "A_E3"),
    ]

    result = aligner(default_min_gap_ms=0).align(
        candidates,
        ["E1", "E2", "E3"],
        [TemporalConstraint(before="E1", after="E3", min_gap_ms=2_000)],
    )

    assert result.status == "no_valid_sequence"


def test_zero_duration_keyframe_candidate_is_valid() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 10_000, 0.8, "A_E1_frame"),
        candidate("E2", "video_A", 10_000, 10_000, 0.9, "A_E2_frame"),
    ]

    result = aligner().align(
        candidates,
        ["E1", "E2"],
        [TemporalConstraint(before="E1", after="E2", allow_overlap=True)],
    )

    assert result.status == "success"


def test_multiple_occurrences_return_top_k_by_sequence_score() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1_first"),
        candidate("E2", "video_A", 20_000, 22_000, 0.8, "A_E2_first"),
        candidate("E3", "video_A", 30_000, 32_000, 0.8, "A_E3_first"),
        candidate("E1", "video_A", 100_000, 102_000, 0.95, "A_E1_second"),
        candidate("E2", "video_A", 110_000, 112_000, 0.95, "A_E2_second"),
        candidate("E3", "video_A", 120_000, 122_000, 0.95, "A_E3_second"),
    ]

    result = aligner(top_k_sequences=2).align(candidates, ["E1", "E2", "E3"], [])

    assert result.status == "success"
    assert len(result.sequences) == 2
    assert [event.candidate_id for event in result.sequences[0].events] == [
        "A_E1_second",
        "A_E2_second",
        "A_E3_second",
    ]


def test_first_occurrence_strict_prefers_earliest_valid_sequence() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1_first"),
        candidate("E2", "video_A", 20_000, 22_000, 0.8, "A_E2_first"),
        candidate("E1", "video_A", 100_000, 102_000, 0.95, "A_E1_second"),
        candidate("E2", "video_A", 110_000, 112_000, 0.95, "A_E2_second"),
    ]

    result = aligner(first_occurrence_mode="strict").align(
        candidates,
        ["E1", "E2"],
        [],
    )

    assert result.status == "success"
    assert result.sequences[0].events[0].candidate_id == "A_E1_first"


def test_large_beam_matches_exact_top_one() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.5, "A_E1_low"),
        candidate("E2", "video_A", 20_000, 22_000, 0.5, "A_E2_low"),
        candidate("E1", "video_B", 10_000, 12_000, 0.9, "B_E1"),
        candidate("E2", "video_B", 20_000, 22_000, 0.9, "B_E2"),
    ]

    exact = aligner(beam_width=None).align(candidates, ["E1", "E2"], [])
    beam = aligner(beam_width=50, per_video_beam_width=10).align(
        candidates,
        ["E1", "E2"],
        [],
    )

    assert exact.sequences[0].events == beam.sequences[0].events


def test_connectivity_pruning_keeps_lower_score_completable_branch() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.99, "A_E1_dead_high"),
        candidate("E1", "video_A", 1_000, 2_000, 0.5, "A_E1_alive_low"),
        candidate("E2", "video_A", 4_000, 5_000, 0.9, "A_E2_alive"),
        candidate("E3", "video_A", 7_000, 8_000, 0.9, "A_E3_alive"),
    ]

    result = aligner(beam_width=1, per_video_beam_width=1).align(
        candidates,
        ["E1", "E2", "E3"],
        [],
    )

    assert result.status == "success"
    assert [event.candidate_id for event in result.sequences[0].events] == [
        "A_E1_alive_low",
        "A_E2_alive",
        "A_E3_alive",
    ]
    assert result.diagnostics["dead_end_state_count"] >= 1


def test_connectivity_memo_uses_full_candidate_identity_not_candidate_id_only() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.7, "shared_E1"),
        candidate("E1", "video_B", 10_000, 12_000, 0.9, "shared_E1"),
        candidate("E2", "video_A", 4_000, 5_000, 0.8, "A_E2"),
        candidate("E3", "video_A", 7_000, 8_000, 0.8, "A_E3"),
        candidate("E2", "video_B", 1_000, 2_000, 0.99, "B_E2_wrong_time"),
        candidate("E3", "video_B", 3_000, 4_000, 0.99, "B_E3_wrong_time"),
    ]

    result = aligner().align(candidates, ["E1", "E2", "E3"], [])

    assert result.status == "success"
    assert result.sequences[0].video_id == "video_A"


def test_paths_per_node_keeps_multiple_histories_for_same_terminal_candidate() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.8, "A_E1_first"),
        candidate("E1", "video_A", 2_000, 3_000, 0.7, "A_E1_second"),
        candidate("E2", "video_A", 5_000, 6_000, 0.9, "A_E2_shared"),
    ]

    one_path = aligner(
        beam_width=10,
        paths_per_node=1,
        top_k_sequences=5,
    ).align(candidates, ["E1", "E2"], [])
    two_paths = aligner(
        beam_width=10,
        paths_per_node=2,
        top_k_sequences=5,
    ).align(candidates, ["E1", "E2"], [])

    assert len(one_path.sequences) == 1
    assert len(two_paths.sequences) == 2


def test_score_breakdowns_match_returned_sequences_after_sort_and_top_k() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1_first"),
        candidate("E2", "video_A", 20_000, 22_000, 0.8, "A_E2_first"),
        candidate("E1", "video_A", 100_000, 102_000, 0.95, "A_E1_second"),
        candidate("E2", "video_A", 110_000, 112_000, 0.95, "A_E2_second"),
    ]

    result = aligner(top_k_sequences=1).align(candidates, ["E1", "E2"], [])

    assert len(result.diagnostics["score_breakdowns"]) == 1
    assert result.diagnostics["score_breakdowns"][0]["candidate_ids"] == [
        event.candidate_id for event in result.sequences[0].events
    ]


def test_duplicate_candidate_ids_do_not_cross_contaminate_score_breakdowns() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.9, "shared"),
        candidate("E2", "video_A", 3_000, 4_000, 0.9, "shared"),
        candidate("E1", "video_B", 10_000, 11_000, 0.7, "shared"),
        candidate("E2", "video_B", 12_000, 13_000, 0.7, "shared"),
    ]

    result = aligner(top_k_sequences=2).align(candidates, ["E1", "E2"], [])

    assert [sequence.video_id for sequence in result.sequences] == ["video_A", "video_B"]
    assert [
        breakdown["event_score_sum"]
        for breakdown in result.diagnostics["score_breakdowns"]
    ] == [1.8, 1.4]


def test_paths_per_node_groups_by_full_terminal_candidate_identity() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.9, "A_E1"),
        candidate("E2", "video_A", 3_000, 4_000, 0.9, "shared_terminal"),
        candidate("E1", "video_B", 1_000, 2_000, 0.8, "B_E1"),
        candidate("E2", "video_B", 3_000, 4_000, 0.8, "shared_terminal"),
    ]

    result = aligner(
        beam_width=10,
        paths_per_node=1,
        top_k_sequences=5,
    ).align(candidates, ["E1", "E2"], [])

    assert [sequence.video_id for sequence in result.sequences] == ["video_A", "video_B"]


def test_temporal_sequence_events_expose_fusion_score_and_evidence_ids() -> None:
    candidates = [
        candidate(
            "E1",
            "video_A",
            1_000,
            2_000,
            0.9,
            "A_E1",
            evidence_ids=["clip_1", "ocr_7"],
        ),
        candidate("E2", "video_A", 3_000, 4_000, 0.8, "A_E2"),
    ]

    result = aligner().align(candidates, ["E1", "E2"], [])

    assert result.status == "success"
    assert result.sequences[0].events[0].fusion_score == 0.9
    assert result.sequences[0].events[0].evidence_ids == ["clip_1", "ocr_7"]


def test_layer_diagnostics_report_transition_work() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 4_000, 5_000, 0.9, "A_E2"),
        candidate("E3", "video_A", 7_000, 8_000, 0.7, "A_E3"),
    ]

    result = aligner().align(candidates, ["E1", "E2", "E3"], [])

    layers = result.diagnostics["layers"]
    assert [layer["transition"] for layer in layers] == ["E1->E2", "E2->E3"]
    assert all("candidate_checks" in layer for layer in layers)
    assert all("valid_edges" in layer for layer in layers)


def test_invalid_scores_and_unknown_events_are_ignored_in_diagnostics() -> None:
    candidates = [
        candidate("E1", "video_A", 10_000, 12_000, 0.8, "A_E1"),
        candidate("E2", "video_A", 20_000, 22_000, 0.9, "A_E2"),
        candidate("E9", "video_A", 30_000, 32_000, 0.7, "unknown_event"),
        candidate("E1", "video_A", 40_000, 42_000, math.nan, "nan_score"),
    ]

    result = aligner().align(candidates, ["E1", "E2"], [])

    assert result.status == "success"
    assert result.diagnostics["ignored_unknown_event_count"] == 1
    assert result.diagnostics["rejected_invalid_score_count"] == 1


def test_score_only_pre_pruning_reports_dropped_candidates() -> None:
    candidates = [
        candidate("E1", "video_A", 1_000, 2_000, 0.9, "A_E1_high"),
        candidate("E1", "video_A", 2_000, 3_000, 0.8, "A_E1_low"),
        candidate("E2", "video_A", 5_000, 6_000, 0.9, "A_E2"),
    ]

    result = aligner(max_candidates_per_event_per_video=1).align(
        candidates,
        ["E1", "E2"],
        [],
    )

    assert result.status == "success"
    assert result.diagnostics["pre_pruned_candidates"] == {"video_A:E1": 1}
