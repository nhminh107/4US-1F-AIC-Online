from BackEnd.app.contracts.models import (
    ConstraintResult,
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
    TemporalConstraint,
    TemporalSequence,
    TemporalEventResult,
    VQAResult,
)
from BackEnd.app.verification.enums import ClaimType
from BackEnd.app.verification.planner.verification_planner import VerificationPlanner


def test_planner_creates_claims_from_structured_constraints() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        ocr_constraints=["HCMC"],
        asr_constraints=["gold medal"],
        object_constraints=["person"],
        negative_constraints=["indoor"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    plan = VerificationPlanner().build_plan(query, result)

    claim_types = {claim.claim_type for claim in plan.claims}
    assert ClaimType.OCR_EXACT in claim_types
    assert ClaimType.ASR_EXACT in claim_types
    assert ClaimType.OBJECT_PRESENCE in claim_types
    assert ClaimType.NEGATIVE_CONSTRAINT in claim_types
    assert plan.focus_claim_ids
    assert plan.target_video_id == "video-1"
    assert plan.target_start_ms == 1000
    assert plan.target_end_ms == 2000


def test_planner_creates_vqa_answer_claim_with_ranked_candidate_window() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="The man receives a gold medal.",
        confidence=0.8,
        evidence_ids=["asr-1"],
        status="answered",
    )
    ranked_candidate = RankedCandidateRegion(
        candidate_id="candidate-1",
        video_id="video-9",
        start_ms=3000,
        end_ms=6000,
        fusion_score=0.91,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )

    plan = VerificationPlanner().build_plan(query, result, [ranked_candidate])

    assert any(claim.claim_type == ClaimType.VQA_ANSWER_CLAIM for claim in plan.claims)
    assert plan.target_video_id == "video-9"
    assert plan.target_start_ms == 3000
    assert plan.target_end_ms == 6000


def test_planner_creates_temporal_claims_for_trake_constraints() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="TRAKE",
        temporal_constraints=[
            TemporalConstraint(before="E1", after="E2", max_gap_ms=5000),
        ],
    )
    result = TemporalSequence(
        video_id="video-1",
        sequence_score=0.8,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="candidate-1",
                start_ms=10000,
                end_ms=12000,
            ),
            TemporalEventResult(
                event_id="E2",
                candidate_id="candidate-2",
                start_ms=14000,
                end_ms=15000,
            ),
        ],
    )

    plan = VerificationPlanner().build_plan(query, result)

    claim_types = {claim.claim_type for claim in plan.claims}
    assert ClaimType.TEMPORAL_ORDER in claim_types
    assert ClaimType.TEMPORAL_GAP in claim_types
    assert plan.target_video_id == "video-1"
    assert plan.target_start_ms == 10000
    assert plan.target_end_ms == 15000


def test_planner_creates_typed_object_count_claim() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        object_constraints=["at least 2 persons"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    plan = VerificationPlanner().build_plan(query, result)
    claim = plan.claims[0]
    count_spec = claim.metadata["count_spec"]

    assert claim.claim_type == ClaimType.OBJECT_COUNT
    assert count_spec.operator == "at_least"
    assert count_spec.expected_count == 2
    assert count_spec.object_label == "person"


def test_planner_marks_zero_object_count_as_unparsed() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        object_constraints=["at least 0 people"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    claim = VerificationPlanner().build_plan(query, result).claims[0]

    assert claim.claim_type == ClaimType.OBJECT_COUNT
    assert claim.metadata == {"count_parse_error": True}
