from BackEnd.app.contracts.models import (
    KISResult,
    StructuredQuery,
    TemporalConstraint,
    TemporalEventResult,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.enums import ClaimStatus, VerificationStatus
from BackEnd.app.verification.planner.verification_planner import VerificationPlanner
from BackEnd.app.verification.preflight import VerificationPreflight


def test_preflight_preserves_upstream_vqa_uncertainty() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="",
        confidence=0.0,
        evidence_ids=[],
        status="uncertain",
    )
    plan = VerificationPlanner().build_plan(query, result, [])

    preflight = VerificationPreflight().evaluate(query, result, plan)

    assert preflight.terminal_status == VerificationStatus.UNCERTAIN
    assert preflight.claim_results[0].claim_id == "upstream_result_uncertain"
    assert preflight.claim_results[0].status == ClaimStatus.UNKNOWN


def test_preflight_rejects_task_result_mismatch() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    plan = VerificationPlanner().build_plan(query, result, [])

    preflight = VerificationPreflight().evaluate(query, result, plan)

    assert preflight.terminal_status == VerificationStatus.UNCERTAIN
    assert preflight.claim_results[0].claim_id == "task_result_mismatch"


def test_preflight_rejects_reversed_temporal_order() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="TRAKE",
        temporal_constraints=[TemporalConstraint(before="E1", after="E2")],
    )
    result = TemporalSequence(
        video_id="video-1",
        sequence_score=0.8,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="candidate-1",
                start_ms=20000,
                end_ms=21000,
            ),
            TemporalEventResult(
                event_id="E2",
                candidate_id="candidate-2",
                start_ms=10000,
                end_ms=11000,
            ),
        ],
    )
    plan = VerificationPlanner().build_plan(query, result, [])

    preflight = VerificationPreflight().evaluate(query, result, plan)

    assert preflight.terminal_status == VerificationStatus.REJECTED
    assert preflight.claim_results[0].claim_id == "claim-temporal-order-E1-E2"
    assert preflight.claim_results[0].status == ClaimStatus.CONTRADICTED


def test_preflight_separates_temporal_order_from_gap_failure() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="TRAKE",
        temporal_constraints=[
            TemporalConstraint(before="E1", after="E2", max_gap_ms=1000)
        ],
    )
    result = TemporalSequence(
        video_id="video-1",
        sequence_score=0.8,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="candidate-1",
                start_ms=1000,
                end_ms=2000,
            ),
            TemporalEventResult(
                event_id="E2",
                candidate_id="candidate-2",
                start_ms=5000,
                end_ms=6000,
            ),
        ],
    )
    plan = VerificationPlanner().build_plan(query, result, [])

    preflight = VerificationPreflight().evaluate(query, result, plan)
    result_by_id = {item.claim_id: item for item in preflight.claim_results}

    assert result_by_id["claim-temporal-order-E1-E2"].status == ClaimStatus.SUPPORTED
    assert result_by_id["claim-temporal-gap-E1-E2"].status == ClaimStatus.CONTRADICTED
    assert preflight.terminal_status == VerificationStatus.REJECTED
