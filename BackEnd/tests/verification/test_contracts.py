import pytest
from pydantic import ValidationError

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    FrameEvidence,
    ObjectCountSpec,
    TextEvidence,
    VerificationClaim,
    VerificationDetailResult,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    ClaimType,
    VerificationStatus,
)


def test_verification_claim_uses_strict_tri_state_status() -> None:
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OCR_EXACT,
        text="HCMC",
        importance=ClaimImportance.HARD,
        current_status=ClaimStatus.NOT_CHECKED,
    )

    assert claim.current_status == ClaimStatus.NOT_CHECKED

    with pytest.raises(ValidationError):
        VerificationClaim(
            claim_id="claim-2",
            claim_type=ClaimType.OCR_EXACT,
            text="HCMC",
            importance=ClaimImportance.HARD,
            current_status="MAYBE",
        )


def test_detail_result_can_represent_need_replan_internally() -> None:
    claim_result = ClaimVerificationResult(
        claim_id="claim-1",
        status=ClaimStatus.UNKNOWN,
        confidence=0.0,
        importance=ClaimImportance.HARD,
        verifier_type="deterministic",
        verifier_name="ocr_exact",
    )
    result = VerificationDetailResult(
        verification_id="ver-1",
        task="KIS",
        target_result_id="candidate-1",
        status=VerificationStatus.NEED_REPLAN,
        confidence=0.0,
        verification_level="deterministic",
        claim_results=[claim_result],
        next_action="REPLAN",
    )

    assert result.status == VerificationStatus.NEED_REPLAN
    assert result.claim_results[0].status == ClaimStatus.UNKNOWN


def test_evidence_pack_rejects_duplicate_ids_across_modalities() -> None:
    with pytest.raises(ValidationError, match="evidence IDs must be unique"):
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="candidate-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
            frame_evidence=[
                FrameEvidence(
                    evidence_id="same-id",
                    frame_id="frame-1",
                    start_ms=1000,
                    end_ms=1000,
                )
            ],
            text_evidence=[
                TextEvidence(
                    evidence_id="same-id",
                    evidence_type="caption",
                    text="caption",
                    start_ms=1000,
                    end_ms=2000,
                )
            ],
        )


def test_object_count_spec_requires_positive_count() -> None:
    with pytest.raises(ValidationError):
        ObjectCountSpec(
            operator="at_least",
            expected_count=0,
            object_label="person",
        )
