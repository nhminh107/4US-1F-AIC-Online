"""Deterministic negative constraint checks."""

from __future__ import annotations

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType
from BackEnd.app.verification.deterministic.text_matching import (
    contains_affirmative_phrase,
)


class NegativeConstraintVerifier:
    verifier_name = "negative_constraint"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        for evidence in evidence_pack.text_evidence:
            if contains_affirmative_phrase(evidence.text, claim.text):
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    status=ClaimStatus.CONTRADICTED,
                    confidence=1.0,
                    importance=claim.importance,
                    evidence_ids=[evidence.evidence_id],
                    verifier_type="deterministic",
                    verifier_name=self.verifier_name,
                )
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            status=ClaimStatus.UNKNOWN,
            confidence=0.0,
            importance=claim.importance,
            evidence_ids=[],
            verifier_type="deterministic",
            verifier_name=self.verifier_name,
        )
