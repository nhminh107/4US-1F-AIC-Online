"""Deterministic ASR claim checks."""

from __future__ import annotations

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType
from BackEnd.app.verification.deterministic.text_matching import contains_phrase


class AsrExactVerifier:
    verifier_name = "asr_exact"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.ASR_EXACT

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        for evidence in evidence_pack.text_evidence:
            if evidence.evidence_type == "asr" and contains_phrase(
                evidence.text,
                claim.text,
            ):
                return _result(claim, ClaimStatus.SUPPORTED, [evidence.evidence_id], 1.0)
        return _result(claim, ClaimStatus.UNKNOWN, [], 0.0)


def _result(
    claim: VerificationClaim,
    status: ClaimStatus,
    evidence_ids: list[str],
    confidence: float,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim.claim_id,
        status=status,
        confidence=confidence,
        importance=claim.importance,
        evidence_ids=evidence_ids,
        verifier_type="deterministic",
        verifier_name=AsrExactVerifier.verifier_name,
    )
