"""Task 11: TRAKE weakest-link event verifier."""

from __future__ import annotations

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.deterministic.base import DeterministicClaimVerifier
from BackEnd.app.verification.enums import ClaimStatus, ClaimType


class TrakeEventVerifier(DeterministicClaimVerifier):
    verifier_name = "trake_event_weakest_link"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.TRAKE_EVENT

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        start_ms = claim.metadata.get("start_ms")
        end_ms = claim.metadata.get("end_ms")
        event_id = claim.metadata.get("event_id")

        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.CONTRADICTED,
                confidence=1.0,
                importance=claim.importance,
                evidence_ids=[],
                verifier_type="deterministic",
                verifier_name=self.verifier_name,
                observation=f"Event {event_id} has invalid time window: end_ms < start_ms",
            )

        supporting_ids: list[str] = []
        if start_ms is not None and end_ms is not None:
            # Check frame evidence
            supporting_ids.extend(
                f.evidence_id
                for f in evidence_pack.frame_evidence
                if start_ms <= f.start_ms <= end_ms
            )
            # Check text evidence
            supporting_ids.extend(
                t.evidence_id
                for t in evidence_pack.text_evidence
                if max(start_ms, t.start_ms) <= min(end_ms, t.end_ms)
            )
            # Check object evidence
            supporting_ids.extend(
                o.evidence_id
                for o in evidence_pack.object_evidence
                if max(start_ms, o.start_ms) <= min(end_ms, o.end_ms)
            )

        if supporting_ids:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.SUPPORTED,
                confidence=0.9,
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


__all__ = ["TrakeEventVerifier"]
