"""Map claim-level verification into final status."""

from __future__ import annotations

from BackEnd.app.contracts.models import VerifiedResult
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationDetailResult,
)
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    NextAction,
    VerificationLevel,
    VerificationStatus,
)


class DecisionPolicy:
    def decide(
        self,
        *,
        verification_id: str,
        task: str,
        target_result_id: str,
        claim_results: list[ClaimVerificationResult],
        gate_reasons: list[str] | None = None,
        verification_level: VerificationLevel = VerificationLevel.DETERMINISTIC,
        accepted_confidence: float = 0.8,
    ) -> VerificationDetailResult:
        hard_results = [
            result
            for result in claim_results
            if result.importance == ClaimImportance.HARD
        ]
        failed = [
            result.claim_id
            for result in hard_results
            if result.status == ClaimStatus.CONTRADICTED
        ]
        uncertain = [
            result.claim_id
            for result in hard_results
            if result.status in {ClaimStatus.UNKNOWN, ClaimStatus.NOT_CHECKED}
        ]
        if not claim_results and verification_level != VerificationLevel.SKIPPED:
            uncertain.append("no_claim_results")
        supporting = sorted(
            {
                evidence_id
                for result in claim_results
                if result.status == ClaimStatus.SUPPORTED
                for evidence_id in result.evidence_ids
            }
        )

        if failed:
            status = VerificationStatus.REJECTED
            confidence = max(
                result.confidence
                for result in hard_results
                if result.status == ClaimStatus.CONTRADICTED
            )
            next_action = NextAction.TRY_NEXT_CANDIDATE
        elif uncertain:
            status = VerificationStatus.UNCERTAIN
            confidence = 0.0
            next_action = NextAction.EXPAND_LOCAL_CONTEXT
        else:
            status = VerificationStatus.ACCEPTED
            supported_hard = [
                result.confidence
                for result in hard_results
                if result.status == ClaimStatus.SUPPORTED
            ]
            confidence = min(supported_hard) if supported_hard else accepted_confidence
            next_action = NextAction.RETURN_RESULT

        return VerificationDetailResult(
            verification_id=verification_id,
            task=task,  # type: ignore[arg-type]
            target_result_id=target_result_id,
            status=status,
            confidence=confidence,
            verification_level=verification_level,
            claim_results=claim_results,
            supporting_evidence_ids=supporting,
            failed_constraint_ids=failed,
            uncertain_constraint_ids=uncertain,
            gate_reasons=gate_reasons or [],
            next_action=next_action,
        )


def to_shared_verified_result(detail: VerificationDetailResult) -> VerifiedResult:
    status_map = {
        VerificationStatus.ACCEPTED: "accepted",
        VerificationStatus.REJECTED: "rejected",
        VerificationStatus.UNCERTAIN: "uncertain",
        VerificationStatus.NEED_REPLAN: "uncertain",
    }
    return VerifiedResult(
        status=status_map[detail.status],  # type: ignore[arg-type]
        confidence=detail.confidence,
        supporting_evidence_ids=detail.supporting_evidence_ids,
        failed_constraints=detail.failed_constraint_ids + detail.uncertain_constraint_ids,
    )
