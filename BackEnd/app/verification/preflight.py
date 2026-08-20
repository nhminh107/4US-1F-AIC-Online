"""Result-level safety checks that run before selective evidence loading."""

from __future__ import annotations

from BackEnd.app.contracts.models import (
    KISResult,
    StructuredQuery,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    PreflightResult,
    VerificationPlan,
)
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    VerificationStatus,
)
from BackEnd.app.verification.temporal_rules import (
    evaluate_temporal_gap,
    evaluate_temporal_order,
)


class VerificationPreflight:
    def evaluate(
        self,
        query: StructuredQuery,
        result: KISResult | VQAResult | TemporalSequence,
        plan: VerificationPlan,
    ) -> PreflightResult:
        if not _task_matches_result(query, result):
            return _terminal_unknown("task_result_mismatch")

        if isinstance(result, VQAResult) and (
            result.status == "uncertain" or not result.answer.strip()
        ):
            return _terminal_unknown("upstream_result_uncertain")

        if plan.claims and not _has_target_context(plan):
            return _terminal_unknown("missing_target_context")

        if not isinstance(result, TemporalSequence):
            return PreflightResult()

        events = {
            event.event_id: {
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
            }
            for event in result.events
        }
        claim_results: list[ClaimVerificationResult] = []
        for constraint in query.temporal_constraints:
            order_status, order_observation = evaluate_temporal_order(
                constraint,
                events,
            )
            claim_results.append(
                _claim_result(
                    f"claim-temporal-order-{constraint.before}-{constraint.after}",
                    order_status,
                    order_observation,
                )
            )
            if constraint.min_gap_ms is not None or constraint.max_gap_ms is not None:
                gap_status, gap_observation = evaluate_temporal_gap(constraint, events)
                claim_results.append(
                    _claim_result(
                        f"claim-temporal-gap-{constraint.before}-{constraint.after}",
                        gap_status,
                        gap_observation,
                    )
                )

        terminal_status = None
        if any(item.status == ClaimStatus.CONTRADICTED for item in claim_results):
            terminal_status = VerificationStatus.REJECTED
        elif any(item.status == ClaimStatus.UNKNOWN for item in claim_results):
            terminal_status = VerificationStatus.UNCERTAIN
        return PreflightResult(
            claim_results=claim_results,
            terminal_status=terminal_status,
        )


def _task_matches_result(
    query: StructuredQuery,
    result: KISResult | VQAResult | TemporalSequence,
) -> bool:
    return (
        (query.task == "KIS" and isinstance(result, KISResult))
        or (query.task == "VQA" and isinstance(result, VQAResult))
        or (query.task == "TRAKE" and isinstance(result, TemporalSequence))
    )


def _has_target_context(plan: VerificationPlan) -> bool:
    return (
        plan.target_video_id is not None
        and plan.target_start_ms is not None
        and plan.target_end_ms is not None
    )


def _terminal_unknown(issue: str) -> PreflightResult:
    return PreflightResult(
        claim_results=[_claim_result(issue, ClaimStatus.UNKNOWN, issue)],
        issues=[issue],
        terminal_status=VerificationStatus.UNCERTAIN,
    )


def _claim_result(
    claim_id: str,
    status: ClaimStatus,
    observation: str,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim_id,
        status=status,
        confidence=1.0 if status != ClaimStatus.UNKNOWN else 0.0,
        importance=ClaimImportance.HARD,
        verifier_type="deterministic",
        verifier_name="preflight",
        observation=observation,
    )
