"""Baseline VQA answer grounding verifier."""

from __future__ import annotations

from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType
from BackEnd.app.verification.deterministic.text_matching import contains_phrase


class VqaAnswerEvidenceVerifier:
    verifier_name = "vqa_answer_evidence"

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.VQA_ANSWER_CLAIM

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        referenced_ids = set(claim.metadata.get("evidence_ids", []))
        supporting_ids = sorted(
            evidence.evidence_id
            for evidence in evidence_pack.text_evidence
            if evidence.evidence_id in referenced_ids
            and contains_phrase(evidence.text, claim.text)
        )
        if supporting_ids:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.SUPPORTED,
                confidence=1.0,
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
