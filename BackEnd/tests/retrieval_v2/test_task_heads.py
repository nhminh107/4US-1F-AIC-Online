from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.retrieval_v2.contracts import MomentBand
from BackEnd.app.retrieval_v2.task_heads import (
    AnswerClaim,
    OfficialFrame,
    aggregate_grounded_answer,
    select_kis_moments,
    select_kis_sequences,
    validate_trake_sequence,
)


def _band(
    band_id: str,
    video_id: str,
    start_ms: int,
    end_ms: int,
    peak_ms: int,
    score: float,
    *,
    event_id: str | None = None,
) -> MomentBand:
    return MomentBand(
        band_id=band_id,
        video_id=video_id,
        event_id=event_id,
        start_ms=start_ms,
        end_ms=end_ms,
        peak_ms=peak_ms,
        score=score,
    )


def _frame(
    evidence_id: str,
    video_id: str,
    frame_idx: int,
    timestamp_ms: int,
    *,
    event_id: str | None = None,
    official: bool = True,
    score: float = 0.0,
) -> OfficialFrame:
    return OfficialFrame(
        evidence_id=evidence_id,
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        event_id=event_id,
        official=official,
        score=score,
    )


def test_kis_moment_uses_only_accepted_bands_and_nearest_peak_frame():
    bands = [
        _band("accepted", "V1", 1_000, 3_000, 2_100, 0.8),
        _band("rejected", "V2", 1_000, 3_000, 2_000, 0.99),
    ]
    frames = [
        _frame("f1", "V1", 10, 1_500),
        _frame("f2", "V1", 11, 2_000),
        _frame("f3", "V2", 20, 2_000),
    ]

    selected = select_kis_moments(
        bands,
        accepted_band_ids={"accepted"},
        official_frames=frames,
        representative_strategy="nearest_peak",
    )

    assert [(item.band_id, item.frame.frame_idx) for item in selected] == [
        ("accepted", 11)
    ]


def test_kis_moment_representative_strategies_are_deterministic():
    band = _band("b1", "V1", 1_000, 4_000, 2_500, 0.8)
    frames = [
        _frame("late-low", "V1", 30, 3_000, score=0.2),
        _frame("early", "V1", 10, 1_000, score=0.4),
        _frame("best", "V1", 20, 2_000, score=0.9),
    ]

    earliest = select_kis_moments(
        [band], {"b1"}, frames, representative_strategy="earliest"
    )
    latest = select_kis_moments(
        [band], {"b1"}, frames, representative_strategy="latest"
    )
    highest = select_kis_moments(
        [band], {"b1"}, frames, representative_strategy="highest_score"
    )

    assert earliest[0].frame.frame_idx == 10
    assert latest[0].frame.frame_idx == 30
    assert highest[0].frame.frame_idx == 20


def test_kis_moment_skips_non_official_and_out_of_band_frames():
    band = _band("b1", "V1", 1_000, 2_000, 1_500, 0.8)
    frames = [
        _frame("unofficial", "V1", 10, 1_500, official=False),
        _frame("outside", "V1", 11, 3_000),
    ]

    assert select_kis_moments([band], {"b1"}, frames) == []


def test_kis_sequence_selects_complete_same_video_ordered_events():
    bands = [
        _band("e1-v1", "V1", 1_000, 2_000, 1_500, 0.8, event_id="E1"),
        _band("e2-v1", "V1", 4_000, 5_000, 4_500, 0.9, event_id="E2"),
        _band("e1-v2", "V2", 1_000, 2_000, 1_500, 0.99, event_id="E1"),
    ]
    frames = [
        _frame("v1-e1", "V1", 10, 1_500, event_id="E1"),
        _frame("v1-e2", "V1", 20, 4_500, event_id="E2"),
        _frame("v2-e1", "V2", 30, 1_500, event_id="E1"),
    ]

    sequences = select_kis_sequences(
        bands,
        accepted_band_ids={band.band_id for band in bands},
        official_frames=frames,
        event_ids=["E1", "E2"],
    )

    assert len(sequences) == 1
    assert sequences[0].video_id == "V1"
    assert [item.event_id for item in sequences[0].items] == ["E1", "E2"]


def test_grounded_qa_ignores_claims_outside_allowed_evidence():
    claims = [
        AnswerClaim(evidence_id="allowed", answer="  Ha Noi  ", confidence=0.7),
        AnswerClaim(evidence_id="foreign", answer="Hue", confidence=1.0),
    ]

    result = aggregate_grounded_answer(claims, allowed_evidence_ids={"allowed"})

    assert result.status == "answered"
    assert result.answer == "Ha Noi"
    assert result.evidence_ids == ("allowed",)


def test_grounded_qa_normalizes_whitespace_and_limits_answer_to_100_chars():
    long_answer = "  " + "word   " * 30

    result = aggregate_grounded_answer(
        [AnswerClaim(evidence_id="e1", answer=long_answer, confidence=0.9)],
        allowed_evidence_ids={"e1"},
    )

    assert result.status == "answered"
    assert len(result.answer) <= 100
    assert "  " not in result.answer


def test_grounded_qa_returns_uncertain_for_conflicting_grounded_claims():
    claims = [
        AnswerClaim(evidence_id="e1", answer="red", confidence=0.9),
        AnswerClaim(evidence_id="e2", answer="blue", confidence=0.8),
    ]

    result = aggregate_grounded_answer(claims, allowed_evidence_ids={"e1", "e2"})

    assert result.status == "uncertain"
    assert result.answer == "uncertain"
    assert result.evidence_ids == ("e1", "e2")


def test_grounded_qa_detects_conflict_after_the_output_length_boundary():
    shared_prefix = "x" * 100
    claims = [
        AnswerClaim(evidence_id="e1", answer=shared_prefix + " red"),
        AnswerClaim(evidence_id="e2", answer=shared_prefix + " blue"),
    ]

    result = aggregate_grounded_answer(claims, allowed_evidence_ids={"e1", "e2"})

    assert result.status == "uncertain"


def test_grounded_qa_returns_uncertain_without_grounded_claims():
    result = aggregate_grounded_answer(
        [AnswerClaim(evidence_id="foreign", answer="yes")],
        allowed_evidence_ids={"allowed"},
    )

    assert result.status == "uncertain"
    assert result.answer == "uncertain"
    assert result.evidence_ids == ()


def test_trake_accepts_exactly_one_official_increasing_frame_per_event():
    frames = [
        _frame("e1", "V1", 10, 1_000, event_id="E1"),
        _frame("e2", "V1", 20, 3_000, event_id="E2"),
        _frame("e3", "V1", 30, 5_000, event_id="E3"),
    ]

    result = validate_trake_sequence(frames, ["E1", "E2", "E3"])

    assert result.valid is True
    assert result.reasons == ()
    assert [frame.frame_idx for frame in result.frames] == [10, 20, 30]


def test_trake_rejects_missing_duplicate_unofficial_and_cross_video_events():
    frames = [
        _frame("e1-a", "V1", 10, 1_000, event_id="E1"),
        _frame("e1-b", "V1", 11, 1_100, event_id="E1"),
        _frame("e2", "V2", 20, 3_000, event_id="E2", official=False),
    ]

    result = validate_trake_sequence(frames, ["E1", "E2", "E3"])

    assert result.valid is False
    assert result.reasons == (
        "EVENT_FRAME_COUNT:E1",
        "UNOFFICIAL_FRAME:E2",
        "MISSING_EVENT:E3",
        "CROSS_VIDEO_SEQUENCE",
    )


def test_trake_rejects_non_increasing_official_frame_indices():
    frames = [
        _frame("e1", "V1", 20, 1_000, event_id="E1"),
        _frame("e2", "V1", 10, 3_000, event_id="E2"),
    ]

    result = validate_trake_sequence(frames, ["E1", "E2"])

    assert result.valid is False
    assert "NON_INCREASING_FRAME_INDEX" in result.reasons


def test_trake_enforces_temporal_min_max_and_overlap_constraints():
    constraints = [
        TemporalConstraint(before="E1", after="E2", min_gap_ms=1_000, max_gap_ms=3_000),
        TemporalConstraint(before="E2", after="E3", allow_overlap=True, max_gap_ms=500),
    ]
    too_close = [
        _frame("e1", "V1", 10, 1_000, event_id="E1"),
        _frame("e2", "V1", 20, 1_500, event_id="E2"),
        _frame("e3", "V1", 30, 2_500, event_id="E3"),
    ]

    result = validate_trake_sequence(
        too_close,
        ["E1", "E2", "E3"],
        temporal_constraints=constraints,
    )

    assert result.valid is False
    assert "MIN_GAP_VIOLATION:E1:E2" in result.reasons
    assert "MAX_GAP_VIOLATION:E2:E3" in result.reasons


def test_trake_overlap_is_rejected_unless_explicitly_allowed():
    frames = [
        _frame("e1", "V1", 10, 2_000, event_id="E1"),
        _frame("e2", "V1", 20, 1_500, event_id="E2"),
    ]

    rejected = validate_trake_sequence(
        frames,
        ["E1", "E2"],
        temporal_constraints=[
            TemporalConstraint(before="E1", after="E2", allow_overlap=False)
        ],
    )
    allowed = validate_trake_sequence(
        frames,
        ["E1", "E2"],
        temporal_constraints=[
            TemporalConstraint(before="E1", after="E2", allow_overlap=True)
        ],
    )

    assert "OVERLAP_NOT_ALLOWED:E1:E2" in rejected.reasons
    assert "OVERLAP_NOT_ALLOWED:E1:E2" not in allowed.reasons
