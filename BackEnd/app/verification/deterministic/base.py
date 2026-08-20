"""Base helpers for deterministic claim verifiers."""

from __future__ import annotations

from typing import Protocol

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)


class DeterministicClaimVerifier(Protocol):
    def supports(self, claim: VerificationClaim) -> bool:
        ...

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        ...
