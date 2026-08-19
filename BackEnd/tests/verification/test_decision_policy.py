from BackEnd.app.verification.contracts import ClaimVerificationResult
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    VerificationLevel,
    VerificationStatus,
)
from BackEnd.app.verification.policy.decision_policy import DecisionPolicy


def claim_result(
    claim_id: str,
    status: ClaimStatus,
    confidence: float,
    importance: ClaimImportance = ClaimImportance.HARD,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim_id,
        status=status,
        confidence=confidence,
        importance=importance,
        verifier_type="deterministic",
        verifier_name="test",
    )


def test_accepted_confidence_uses_weakest_hard_support() -> None:
    detail = DecisionPolicy().decide(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        claim_results=[
            claim_result("claim-1", ClaimStatus.SUPPORTED, 0.95),
            claim_result("claim-2", ClaimStatus.SUPPORTED, 0.72),
        ],
    )

    assert detail.confidence == 0.72


def test_rejected_confidence_uses_strongest_hard_contradiction() -> None:
    detail = DecisionPolicy().decide(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        claim_results=[
            claim_result("claim-1", ClaimStatus.CONTRADICTED, 0.81),
            claim_result("claim-2", ClaimStatus.CONTRADICTED, 0.93),
        ],
    )

    assert detail.confidence == 0.93


def test_skipped_accept_uses_explicit_upstream_confidence() -> None:
    detail = DecisionPolicy().decide(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        claim_results=[],
        accepted_confidence=0.67,
        verification_level=VerificationLevel.SKIPPED,
    )

    assert detail.confidence == 0.67


def test_soft_unknown_does_not_block_supported_hard_claim() -> None:
    detail = DecisionPolicy().decide(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        claim_results=[
            claim_result("hard", ClaimStatus.SUPPORTED, 0.9),
            claim_result(
                "soft",
                ClaimStatus.UNKNOWN,
                0.0,
                importance=ClaimImportance.SOFT,
            ),
        ],
    )

    assert detail.status.value == "ACCEPTED"
    assert detail.confidence == 0.9


def test_empty_deterministic_result_is_uncertain() -> None:
    detail = DecisionPolicy().decide(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        claim_results=[],
        verification_level=VerificationLevel.DETERMINISTIC,
    )

    assert detail.status == VerificationStatus.UNCERTAIN
    assert detail.uncertain_constraint_ids == ["no_claim_results"]
