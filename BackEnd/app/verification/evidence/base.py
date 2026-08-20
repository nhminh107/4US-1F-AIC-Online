"""Evidence provider interface for verifier-local evidence loading."""

from __future__ import annotations

from typing import Protocol

from BackEnd.app.verification.contracts import VerificationEvidencePack, VerificationPlan


class EvidenceProvider(Protocol):
    async def build_evidence_pack(
        self,
        plan: VerificationPlan,
    ) -> VerificationEvidencePack:
        ...
