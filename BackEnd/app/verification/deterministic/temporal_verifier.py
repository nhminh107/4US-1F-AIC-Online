"""Deterministic temporal claim checks for TRAKE sequences."""

from __future__ import annotations

from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType
from BackEnd.app.verification.temporal_rules import (
    evaluate_temporal_gap,
    evaluate_temporal_order,
)


class TemporalConstraintVerifier:
    verifier_name = "temporal_constraint"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type in {ClaimType.TEMPORAL_ORDER, ClaimType.TEMPORAL_GAP}

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        constraint = claim.metadata.get("constraint")
        events = claim.metadata.get("events", {})
        if not isinstance(constraint, TemporalConstraint):
            return self._result(claim, ClaimStatus.UNKNOWN, "missing temporal constraint")

        if claim.claim_type == ClaimType.TEMPORAL_ORDER:
            status, observation = evaluate_temporal_order(constraint, events)
        else:
            status, observation = evaluate_temporal_gap(constraint, events)
        return self._result(claim, status, observation)

    def _result(
        self,
        claim: VerificationClaim,
        status: ClaimStatus,
        observation: str,
    ) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            status=status,
            confidence=1.0 if status != ClaimStatus.UNKNOWN else 0.0,
            importance=claim.importance,
            evidence_ids=[],
            verifier_type="deterministic",
            verifier_name=self.verifier_name,
            observation=observation,
        )
