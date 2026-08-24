"""Task 9: KIS local moment consistency verifier."""

from __future__ import annotations

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.deterministic.base import DeterministicClaimVerifier
from BackEnd.app.verification.enums import ClaimStatus, ClaimType


class KisMomentVerifier(DeterministicClaimVerifier):
    verifier_name = "kis_moment_consistency"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.KIS_MOMENT

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        start_ms = claim.metadata.get("start_ms")
        end_ms = claim.metadata.get("end_ms")
        rep_frame_id = claim.metadata.get("representative_frame_id")

        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.CONTRADICTED,
                confidence=1.0,
                importance=claim.importance,
                evidence_ids=[],
                verifier_type="deterministic",
                verifier_name=self.verifier_name,
                observation="Invalid time window: end_ms < start_ms",
            )

        supporting_ids = [
            f.evidence_id
            for f in evidence_pack.frame_evidence
            if (rep_frame_id and f.frame_id == rep_frame_id)
            or (start_ms is not None and end_ms is not None and start_ms <= f.start_ms <= end_ms)
        ]

        if supporting_ids or (start_ms is not None and end_ms is not None and start_ms <= end_ms):
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.SUPPORTED,
                confidence=0.9 if supporting_ids else 0.75,
                importance=claim.importance,
                evidence_ids=supporting_ids,
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


__all__ = ["KisMomentVerifier"]
