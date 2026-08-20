from BackEnd.app.verification.calibration.feature_builder import build_verification_context
from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import VerificationContext
from BackEnd.app.verification.enums import VerificationStatus
from BackEnd.app.verification.gate.rule_based_gate import RuleBasedVerificationGate
from BackEnd.app.contracts.models import (
    ConstraintResult,
    KISResult,
    RankedCandidateRegion,
    VQAResult,
)


def ranked_candidate(candidate_id: str, score: float) -> RankedCandidateRegion:
    return RankedCandidateRegion(
        candidate_id=candidate_id,
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        fusion_score=score,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )


def test_gate_requests_verification_for_small_top_margin() -> None:
    context = build_verification_context(
        query_id="query-1",
        task="KIS",
        result=KISResult(
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
            representative_frame_id="frame-1",
            score=0.9,
        ),
        ranked_candidates=[
            ranked_candidate("candidate-1", 0.90),
            ranked_candidate("candidate-2", 0.89),
        ],
    )

    decision = RuleBasedVerificationGate().decide(context)

    assert decision.should_verify is True
    assert "small_top1_top2_margin" in decision.reasons


def test_gate_requests_verification_for_low_vqa_confidence() -> None:
    context = build_verification_context(
        query_id="query-1",
        task="VQA",
        result=VQAResult(
            answer="gold medal",
            confidence=0.4,
            evidence_ids=["asr-1"],
            status="answered",
        ),
        ranked_candidates=[ranked_candidate("candidate-1", 0.90)],
    )

    decision = RuleBasedVerificationGate().decide(context)

    assert decision.should_verify is True
    assert "low_vqa_confidence" in decision.reasons


def test_gate_can_skip_when_no_risk_signal_is_present() -> None:
    config = VerificationConfig()
    context = build_verification_context(
        query_id="query-1",
        task="KIS",
        result=KISResult(
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
            representative_frame_id="frame-1",
            score=0.9,
        ),
        ranked_candidates=[
            ranked_candidate("candidate-1", 0.90),
            ranked_candidate("candidate-2", 0.70),
        ],
    )

    decision = RuleBasedVerificationGate(config).decide(context)

    assert decision.should_verify is False
    assert decision.reasons == []


def test_feature_builder_does_not_infer_hard_contradiction_from_candidate_constraints() -> None:
    context = build_verification_context(
        query_id="query-1",
        task="KIS",
        result=KISResult(
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
            representative_frame_id="frame-1",
            score=0.9,
        ),
        ranked_candidates=[
            RankedCandidateRegion(
                candidate_id="candidate-1",
                video_id="video-1",
                start_ms=1000,
                end_ms=2000,
                fusion_score=0.90,
                constraint_result=ConstraintResult(
                    hard_constraints_passed=False,
                    negative_constraints_passed=False,
                ),
            )
        ],
    )

    assert context.hard_unknown == 0
    assert context.hard_contradicted == 0


def test_gate_direct_rejects_only_explicit_hard_contradiction_context() -> None:
    decision = RuleBasedVerificationGate().decide(
        VerificationContext(
            query_id="query-1",
            task="KIS",
            target_result_id="frame-1",
            hard_contradicted=1,
        )
    )

    assert decision.should_verify is False
    assert decision.direct_status == VerificationStatus.REJECTED
    assert decision.reasons == ["hard_constraint_contradicted"]
