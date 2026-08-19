"""Public Selective Verifier service."""

from __future__ import annotations

from asyncio import CancelledError, timeout
from collections.abc import Callable
from time import perf_counter
from typing import Any

from BackEnd.app.contracts.models import (
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
    TemporalSequence,
    VerifiedResult,
    VQAResult,
)
from BackEnd.app.verification.calibration.feature_builder import (
    build_verification_context,
)
from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationDetailResult,
    VerificationEvidencePack,
    VerificationPlan,
)
from BackEnd.app.verification.deterministic.asr_verifier import AsrExactVerifier
from BackEnd.app.verification.deterministic.base import DeterministicClaimVerifier
from BackEnd.app.verification.deterministic.negative_constraint_verifier import (
    NegativeConstraintVerifier,
)
from BackEnd.app.verification.deterministic.object_verifier import (
    ObjectPresenceVerifier,
)
from BackEnd.app.verification.deterministic.ocr_verifier import OcrExactVerifier
from BackEnd.app.verification.deterministic.temporal_verifier import (
    TemporalConstraintVerifier,
)
from BackEnd.app.verification.deterministic.vqa_answer_verifier import (
    VqaAnswerEvidenceVerifier,
)
from BackEnd.app.verification.evidence.base import EvidenceProvider
from BackEnd.app.verification.evidence.evidence_pack_builder import bound_evidence_pack
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    VerificationLevel,
    VerificationStatus,
)
from BackEnd.app.verification.gate.base import VerificationGate
from BackEnd.app.verification.gate.rule_based_gate import RuleBasedVerificationGate
from BackEnd.app.verification.metrics.logging import build_verification_log
from BackEnd.app.verification.planner.verification_planner import VerificationPlanner
from BackEnd.app.verification.policy.decision_policy import (
    DecisionPolicy,
    to_shared_verified_result,
)
from BackEnd.app.verification.preflight import VerificationPreflight


class VerificationService:
    """Verify KIS, VQA, and TRAKE task results against local evidence."""

    def __init__(
        self,
        evidence_provider: EvidenceProvider | None = None,
        *,
        config: VerificationConfig | None = None,
        gate: VerificationGate | None = None,
        planner: VerificationPlanner | None = None,
        deterministic_verifiers: list[DeterministicClaimVerifier] | None = None,
        decision_policy: DecisionPolicy | None = None,
        preflight: VerificationPreflight | None = None,
        diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or VerificationConfig()
        self.evidence_provider = evidence_provider
        self.gate = gate or RuleBasedVerificationGate(self.config)
        self.planner = planner or VerificationPlanner()
        self.deterministic_verifiers = (
            deterministic_verifiers
            if deterministic_verifiers is not None
            else [
                OcrExactVerifier(self.config),
                AsrExactVerifier(),
                ObjectPresenceVerifier(self.config),
                TemporalConstraintVerifier(),
                NegativeConstraintVerifier(),
                VqaAnswerEvidenceVerifier(),
            ]
        )
        self.decision_policy = decision_policy or DecisionPolicy()
        self.preflight = preflight or VerificationPreflight()
        self.diagnostic_sink = diagnostic_sink

    async def verify(
        self,
        structured_query: StructuredQuery,
        result: KISResult | VQAResult | TemporalSequence,
        ranked_candidates: list[RankedCandidateRegion],
    ) -> VerifiedResult:
        detail = await self.verify_detailed(structured_query, result, ranked_candidates)
        return to_shared_verified_result(detail)

    async def verify_detailed(
        self,
        structured_query: StructuredQuery,
        result: KISResult | VQAResult | TemporalSequence,
        ranked_candidates: list[RankedCandidateRegion],
    ) -> VerificationDetailResult:
        started = perf_counter()
        plan = self.planner.build_plan(structured_query, result, ranked_candidates)
        diagnostic_focus_ids = list(plan.focus_claim_ids)
        evidence_count_by_type = _empty_evidence_counts()
        omitted_evidence_count = 0
        latency_ms: dict[str, float] = {}

        def finalize(detail: VerificationDetailResult) -> VerificationDetailResult:
            latency_ms["total"] = (perf_counter() - started) * 1000
            enriched = detail.model_copy(
                update={
                    "focus_claim_ids": diagnostic_focus_ids,
                    "evidence_count_by_type": evidence_count_by_type,
                    "omitted_evidence_count": omitted_evidence_count,
                    "latency_ms": latency_ms,
                }
            )
            if self.diagnostic_sink is not None:
                try:
                    self.diagnostic_sink(build_verification_log(enriched))
                except Exception:
                    pass
            return enriched

        preflight = self.preflight.evaluate(structured_query, result, plan)
        if preflight.terminal_status is not None:
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=preflight.claim_results,
                verification_level=VerificationLevel.DETERMINISTIC,
                accepted_confidence=_upstream_confidence(result),
            ))

        if not self.config.enabled:
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=preflight.claim_results,
                gate_reasons=["verification_disabled"],
                verification_level=VerificationLevel.SKIPPED,
                accepted_confidence=_upstream_confidence(result),
            ))

        context = build_verification_context(
            query_id=structured_query.query_id,
            task=structured_query.task,
            result=result,
            ranked_candidates=ranked_candidates,
            preflight=preflight,
        )
        gate_decision = self.gate.decide(context)

        if gate_decision.direct_status == VerificationStatus.REJECTED:
            direct_reasons = gate_decision.reasons or ["gate_direct_rejection"]
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=[
                    _synthetic_claim_result(reason, ClaimStatus.CONTRADICTED)
                    for reason in direct_reasons
                ],
                gate_reasons=direct_reasons,
                verification_level=VerificationLevel.SKIPPED,
            ))

        if not gate_decision.should_verify:
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=preflight.claim_results,
                gate_reasons=gate_decision.reasons,
                verification_level=VerificationLevel.SKIPPED,
                accepted_confidence=_upstream_confidence(result),
            ))

        resolved_claim_ids = {item.claim_id for item in preflight.claim_results}
        focus_claim_ids = set(plan.focus_claim_ids) - resolved_claim_ids
        diagnostic_focus_ids = [
            claim_id for claim_id in plan.focus_claim_ids if claim_id in focus_claim_ids
        ]
        focus_claims = [
            claim for claim in plan.claims if claim.claim_id in focus_claim_ids
        ]
        if not focus_claims:
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=[
                    *preflight.claim_results,
                    _synthetic_claim_result(
                        "no_verifiable_focus_claim",
                        ClaimStatus.UNKNOWN,
                    ),
                ],
                gate_reasons=gate_decision.reasons,
                verification_level=VerificationLevel.DETERMINISTIC,
            ))

        if not any(self._verifier_for(claim) is not None for claim in focus_claims):
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=[
                    *preflight.claim_results,
                    *[
                        _unsupported_claim_result(claim)
                        for claim in focus_claims
                    ],
                ],
                gate_reasons=gate_decision.reasons,
                verification_level=VerificationLevel.DETERMINISTIC,
            ))

        if not _has_target_context(plan):
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=[
                    *preflight.claim_results,
                    _synthetic_claim_result("missing_target_context", ClaimStatus.UNKNOWN),
                ],
                gate_reasons=gate_decision.reasons,
                verification_level=VerificationLevel.DETERMINISTIC,
            ))

        evidence_started = perf_counter()
        try:
            async with timeout(self.config.evidence.timeout_ms / 1000):
                evidence_pack = await self._build_evidence_pack(plan, result)
        except CancelledError:
            raise
        except Exception:
            latency_ms["evidence"] = (perf_counter() - evidence_started) * 1000
            return finalize(self.decision_policy.decide(
                verification_id=plan.verification_id,
                task=structured_query.task,
                target_result_id=plan.target_result_id,
                claim_results=[
                    *preflight.claim_results,
                    _synthetic_claim_result(
                        "evidence_provider_unavailable",
                        ClaimStatus.UNKNOWN,
                    ),
                ],
                gate_reasons=gate_decision.reasons,
                verification_level=VerificationLevel.DETERMINISTIC,
            ))
        latency_ms["evidence"] = (perf_counter() - evidence_started) * 1000
        evidence_pack = bound_evidence_pack(
            evidence_pack,
            self.config,
            plan.required_evidence_types,
            {
                evidence_id
                for claim in focus_claims
                for evidence_id in claim.metadata.get("evidence_ids", [])
                if isinstance(evidence_id, str)
            },
        )
        evidence_count_by_type = _evidence_counts(evidence_pack)
        omitted_evidence_count = evidence_pack.omitted_evidence_count
        deterministic_started = perf_counter()
        claim_results = [
            *preflight.claim_results,
            *self._verify_claims(focus_claims, evidence_pack),
        ]
        latency_ms["deterministic"] = (perf_counter() - deterministic_started) * 1000
        return finalize(self.decision_policy.decide(
            verification_id=plan.verification_id,
            task=structured_query.task,
            target_result_id=plan.target_result_id,
            claim_results=claim_results,
            gate_reasons=gate_decision.reasons,
            verification_level=VerificationLevel.DETERMINISTIC,
        ))

    async def _build_evidence_pack(
        self,
        plan: VerificationPlan,
        result: KISResult | VQAResult | TemporalSequence,
    ) -> VerificationEvidencePack:
        if self.evidence_provider is not None:
            return await self.evidence_provider.build_evidence_pack(plan)

        if not _has_target_context(plan):
            raise ValueError("Verification plan does not contain canonical target context.")

        return VerificationEvidencePack(
            verification_id=plan.verification_id,
            candidate_id=plan.target_result_id,
            video_id=plan.target_video_id or "",
            start_ms=plan.target_start_ms or 0,
            end_ms=plan.target_end_ms or 0,
        )

    def _verify_claims(
        self,
        claims: list[VerificationClaim],
        evidence_pack: VerificationEvidencePack,
    ) -> list[ClaimVerificationResult]:
        results: list[ClaimVerificationResult] = []
        for claim in claims:
            verifier = self._verifier_for(claim)
            if verifier is None:
                results.append(_unsupported_claim_result(claim))
                continue
            results.append(verifier.verify(claim, evidence_pack))
        return results

    def _verifier_for(
        self,
        claim: VerificationClaim,
    ) -> DeterministicClaimVerifier | None:
        return next(
            (
                candidate
                for candidate in self.deterministic_verifiers
                if candidate.supports(claim)
            ),
            None,
        )


def _has_target_context(plan: VerificationPlan) -> bool:
    return (
        plan.target_video_id is not None
        and plan.target_start_ms is not None
        and plan.target_end_ms is not None
    )


def _synthetic_claim_result(
    claim_id: str,
    status: ClaimStatus,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim_id,
        status=status,
        confidence=1.0 if status == ClaimStatus.CONTRADICTED else 0.0,
        importance=ClaimImportance.HARD,
        verifier_type="deterministic",
        verifier_name="verification_service",
    )


def _unsupported_claim_result(
    claim: VerificationClaim,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim.claim_id,
        status=ClaimStatus.UNKNOWN,
        confidence=0.0,
        importance=claim.importance,
        verifier_type="deterministic",
        verifier_name="unsupported_claim_type",
    )


def _upstream_confidence(
    result: KISResult | VQAResult | TemporalSequence,
) -> float:
    if isinstance(result, KISResult):
        confidence = result.score
    elif isinstance(result, VQAResult):
        confidence = result.confidence
    else:
        confidence = result.sequence_score
    return max(0.0, min(1.0, confidence))


def _empty_evidence_counts() -> dict[str, int]:
    return {
        "frame": 0,
        "ocr": 0,
        "asr": 0,
        "caption": 0,
        "object": 0,
        "track": 0,
    }


def _evidence_counts(pack: VerificationEvidencePack) -> dict[str, int]:
    counts = _empty_evidence_counts()
    counts["frame"] = len(pack.frame_evidence)
    counts["object"] = len(pack.object_evidence)
    counts["track"] = len(pack.track_evidence)
    for evidence in pack.text_evidence:
        counts[evidence.evidence_type] += 1
    return counts
