"""Deterministic OCR claim checks."""

from __future__ import annotations

from difflib import SequenceMatcher

from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.deterministic.text_matching import (
    contains_phrase,
    normalize_text,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType


class OcrExactVerifier:
    verifier_name = "ocr_exact"

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type == ClaimType.OCR_EXACT

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        needle = normalize_text(claim.text)
        if not needle:
            return _result(claim, ClaimStatus.UNKNOWN, [], confidence=0.0)
        for evidence in evidence_pack.text_evidence:
            if evidence.evidence_type != "ocr":
                continue
            haystack = normalize_text(evidence.text)
            if contains_phrase(haystack, needle):
                return _result(
                    claim,
                    ClaimStatus.SUPPORTED,
                    [evidence.evidence_id],
                    confidence=1.0,
                )
            if SequenceMatcher(None, needle, haystack).ratio() >= (
                self.config.deterministic.ocr_fuzzy_threshold
            ):
                return _result(
                    claim,
                    ClaimStatus.SUPPORTED,
                    [evidence.evidence_id],
                    confidence=self.config.deterministic.ocr_fuzzy_threshold,
                )
        return _result(claim, ClaimStatus.UNKNOWN, [], confidence=0.0)


def _result(
    claim: VerificationClaim,
    status: ClaimStatus,
    evidence_ids: list[str],
    *,
    confidence: float,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim.claim_id,
        status=status,
        confidence=confidence,
        importance=claim.importance,
        evidence_ids=evidence_ids,
        verifier_type="deterministic",
        verifier_name=OcrExactVerifier.verifier_name,
    )
