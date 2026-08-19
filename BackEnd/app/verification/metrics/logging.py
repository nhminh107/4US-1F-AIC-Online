"""Build content-safe structured diagnostics for verification decisions."""

from __future__ import annotations

from typing import Any

from BackEnd.app.verification.contracts import VerificationDetailResult


def build_verification_log(detail: VerificationDetailResult) -> dict[str, Any]:
    return {
        "verification_id": detail.verification_id,
        "task": detail.task,
        "target_result_id": detail.target_result_id,
        "status": detail.status.value,
        "confidence": detail.confidence,
        "verification_level": detail.verification_level.value,
        "gate_reasons": list(detail.gate_reasons),
        "focus_claim_ids": list(detail.focus_claim_ids),
        "evidence_count_by_type": dict(detail.evidence_count_by_type),
        "omitted_evidence_count": detail.omitted_evidence_count,
        "claim_results": [
            {
                "claim_id": result.claim_id,
                "status": result.status.value,
                "confidence": result.confidence,
                "verifier_type": result.verifier_type,
                "verifier_name": result.verifier_name,
                "evidence_ids": list(result.evidence_ids),
            }
            for result in detail.claim_results
        ],
        "latency_ms": dict(detail.latency_ms),
        "next_action": detail.next_action.value,
    }
