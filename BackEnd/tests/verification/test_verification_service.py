import asyncio

import pytest

from BackEnd.app.contracts.models import (
    ConstraintResult,
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
    TemporalConstraint,
    TemporalEventResult,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.contracts import (
    ObjectEvidence,
    TextEvidence,
    TrackEvidence,
    VerificationEvidencePack,
)
from BackEnd.app.verification.evidence.base import EvidenceProvider
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    VerificationClaim,
    VerificationContext,
    VerificationGateDecision,
    VerificationPlan,
)
from BackEnd.app.verification.config import EvidenceConfig, VerificationConfig
from BackEnd.app.verification.enums import (
    ClaimImportance,
    ClaimStatus,
    ClaimType,
    VerificationStatus,
)
from BackEnd.app.verification.verification_service import VerificationService


class FakeEvidenceProvider(EvidenceProvider):
    def __init__(self, pack: VerificationEvidencePack) -> None:
        self.pack = pack
        self.build_count = 0

    async def build_evidence_pack(self, *args, **kwargs) -> VerificationEvidencePack:
        self.build_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.pack


class DirectRejectGate:
    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        return VerificationGateDecision(
            should_verify=False,
            direct_status=VerificationStatus.REJECTED,
            reasons=["hard_constraint_contradicted"],
        )


class ReasonlessDirectRejectGate:
    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        return VerificationGateDecision(
            should_verify=False,
            direct_status=VerificationStatus.REJECTED,
        )


class SkipGate:
    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        return VerificationGateDecision(should_verify=False)


class AlwaysVerifyGate:
    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        return VerificationGateDecision(
            should_verify=True,
            reasons=["test_requires_verification"],
        )


class FailingEvidenceProvider(EvidenceProvider):
    async def build_evidence_pack(self, *args, **kwargs) -> VerificationEvidencePack:
        raise RuntimeError("database unavailable")


class SlowEvidenceProvider(EvidenceProvider):
    async def build_evidence_pack(self, *args, **kwargs) -> VerificationEvidencePack:
        await asyncio.sleep(0.05)
        raise AssertionError("timeout should cancel the provider call")


class CancelledEvidenceProvider(EvidenceProvider):
    async def build_evidence_pack(self, *args, **kwargs) -> VerificationEvidencePack:
        raise asyncio.CancelledError


class EmptyFocusPlanner:
    def build_plan(self, query, result, ranked_candidates) -> VerificationPlan:
        return VerificationPlan(
            verification_id="ver-empty-focus",
            query_id=query.query_id,
            task=query.task,
            target_result_id=result.representative_frame_id,
            target_video_id=result.video_id,
            target_start_ms=result.start_ms,
            target_end_ms=result.end_ms,
            claims=[
                VerificationClaim(
                    claim_id="already-supported",
                    claim_type=ClaimType.OCR_EXACT,
                    text="HCMC",
                    importance=ClaimImportance.HARD,
                    current_status=ClaimStatus.SUPPORTED,
                )
            ],
            focus_claim_ids=[],
            required_evidence_types=["ocr"],
        )


class UnsupportedFocusPlanner:
    def build_plan(self, query, result, ranked_candidates) -> VerificationPlan:
        return VerificationPlan(
            verification_id="ver-unsupported-focus",
            query_id=query.query_id,
            task=query.task,
            target_result_id=result.representative_frame_id,
            target_video_id=result.video_id,
            target_start_ms=result.start_ms,
            target_end_ms=result.end_ms,
            claims=[
                VerificationClaim(
                    claim_id="claim-visual-1",
                    claim_type=ClaimType.VISUAL_ENTITY,
                    text="person",
                    importance=ClaimImportance.HARD,
                )
            ],
            focus_claim_ids=["claim-visual-1"],
            required_evidence_types=[],
        )


class MixedFocusPlanner:
    def build_plan(self, query, result, ranked_candidates) -> VerificationPlan:
        claims = [
            VerificationClaim(
                claim_id="claim-ocr-1",
                claim_type=ClaimType.OCR_EXACT,
                text="HCMC",
                importance=ClaimImportance.HARD,
            ),
            VerificationClaim(
                claim_id="claim-visual-1",
                claim_type=ClaimType.VISUAL_ENTITY,
                text="person",
                importance=ClaimImportance.HARD,
            ),
        ]
        return VerificationPlan(
            verification_id="ver-mixed-focus",
            query_id=query.query_id,
            task=query.task,
            target_result_id=result.representative_frame_id,
            target_video_id=result.video_id,
            target_start_ms=result.start_ms,
            target_end_ms=result.end_ms,
            claims=claims,
            focus_claim_ids=[claim.claim_id for claim in claims],
            required_evidence_types=["ocr"],
        )


class CapturingAsrVerifier:
    verifier_name = "capturing_asr"

    def __init__(self) -> None:
        self.seen_evidence_types = []

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_id == "claim-asr-1"

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        self.seen_evidence_types = [
            evidence.evidence_type for evidence in evidence_pack.text_evidence
        ]
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            status=ClaimStatus.SUPPORTED,
            confidence=1.0,
            importance=claim.importance,
            evidence_ids=[],
            verifier_type="deterministic",
            verifier_name=self.verifier_name,
        )


def test_service_accepts_supported_ocr_claim() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        ocr_constraints=["HCMC"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        text_evidence=[
            TextEvidence(
                evidence_id="ocr-1",
                evidence_type="ocr",
                text="HCMC",
                start_ms=1000,
                end_ms=1000,
            )
        ],
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
        ).verify(
            query,
            result,
            [],
        )
    )

    assert verified.status == "accepted"
    assert verified.supporting_evidence_ids == ["ocr-1"]


def test_service_rejects_temporal_contradiction() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="TRAKE",
        temporal_constraints=[TemporalConstraint(before="E1", after="E2")],
    )
    sequence = TemporalSequence(
        video_id="video-1",
        sequence_score=0.8,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="candidate-1",
                start_ms=20000,
                end_ms=21000,
            ),
            TemporalEventResult(
                event_id="E2",
                candidate_id="candidate-2",
                start_ms=10000,
                end_ms=11000,
            ),
        ],
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="sequence-1",
        video_id="video-1",
        start_ms=10000,
        end_ms=21000,
    )

    verified = asyncio.run(
        VerificationService(FakeEvidenceProvider(pack)).verify(
            query,
            sequence,
            [],
        )
    )

    assert verified.status == "rejected"
    assert "claim-temporal-order-E1-E2" in verified.failed_constraints


def test_service_direct_gate_rejection_has_failed_reason() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    verified = asyncio.run(
        VerificationService(gate=DirectRejectGate()).verify(query, result, [])
    )

    assert verified.status == "rejected"
    assert verified.confidence == 1.0
    assert verified.failed_constraints == ["hard_constraint_contradicted"]


def test_service_reasonless_direct_gate_rejection_still_rejects() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    verified = asyncio.run(
        VerificationService(gate=ReasonlessDirectRejectGate()).verify(
            query,
            result,
            [],
        )
    )

    assert verified.status == "rejected"
    assert verified.failed_constraints == ["gate_direct_rejection"]


def test_service_passes_vqa_ranked_candidate_window_to_provider() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="gold medal",
        confidence=0.9,
        evidence_ids=["asr-1"],
        status="answered",
    )
    candidate = RankedCandidateRegion(
        candidate_id="candidate-1",
        video_id="video-2",
        start_ms=3000,
        end_ms=5000,
        fusion_score=0.9,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-2",
        start_ms=3000,
        end_ms=5000,
        text_evidence=[
            TextEvidence(
                evidence_id="asr-1",
                evidence_type="asr",
                text="gold medal",
                start_ms=3000,
                end_ms=5000,
            )
        ],
    )
    provider = FakeEvidenceProvider(pack)

    verified = asyncio.run(
        VerificationService(provider, gate=AlwaysVerifyGate()).verify(
            query,
            result,
            [candidate],
        )
    )
    plan = provider.last_args[0]

    assert verified.status == "accepted"
    assert plan.target_video_id == "video-2"
    assert plan.target_start_ms == 3000
    assert plan.target_end_ms == 5000


def test_service_returns_uncertain_when_vqa_has_no_candidate_window() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="gold medal",
        confidence=0.9,
        evidence_ids=["asr-1"],
        status="answered",
    )

    verified = asyncio.run(VerificationService().verify(query, result, []))

    assert verified.status == "uncertain"
    assert "missing_target_context" in verified.failed_constraints


def test_service_bounds_custom_provider_pack_before_verification() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        asr_constraints=["gold medal"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="frame-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        text_evidence=[
            TextEvidence(
                evidence_id="ocr-1",
                evidence_type="ocr",
                text="noise",
                start_ms=1000,
                end_ms=2000,
            ),
            TextEvidence(
                evidence_id="asr-1",
                evidence_type="asr",
                text="gold medal",
                start_ms=1000,
                end_ms=2000,
            ),
            TextEvidence(
                evidence_id="asr-2",
                evidence_type="asr",
                text="extra",
                start_ms=1000,
                end_ms=2000,
            ),
        ],
    )
    verifier = CapturingAsrVerifier()
    config = VerificationConfig(
        evidence=EvidenceConfig(max_text_items=1, max_frames=0, max_objects=0)
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            config=config,
            gate=AlwaysVerifyGate(),
            deterministic_verifiers=[verifier],
        ).verify(query, result, [])
    )

    assert verified.status == "accepted"
    assert verifier.seen_evidence_types == ["asr"]


def test_service_accepts_object_claim_supported_by_track_evidence() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        object_constraints=["person"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="frame-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        track_evidence=[
            TrackEvidence(
                evidence_id="track-1",
                class_name="person",
                observation_count=2,
                confidence=0.8,
                start_ms=1000,
                end_ms=2000,
            )
        ],
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "accepted"
    assert verified.supporting_evidence_ids == ["track-1"]


def test_service_gate_skip_with_claims_does_not_load_evidence() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.91,
    )
    provider = FakeEvidenceProvider(
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="frame-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
        )
    )

    verified = asyncio.run(
        VerificationService(provider, gate=SkipGate()).verify(query, result, [])
    )

    assert verified.status == "accepted"
    assert provider.build_count == 0


def test_service_preserves_upstream_vqa_uncertainty() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="",
        confidence=0.0,
        evidence_ids=[],
        status="uncertain",
    )
    candidate = RankedCandidateRegion(
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        fusion_score=0.8,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )

    verified = asyncio.run(VerificationService().verify(query, result, [candidate]))

    assert verified.status == "uncertain"
    assert "upstream_result_uncertain" in verified.failed_constraints


def test_service_converts_evidence_provider_failure_to_uncertain() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    verified = asyncio.run(
        VerificationService(
            FailingEvidenceProvider(),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert "evidence_provider_unavailable" in verified.failed_constraints


def test_service_times_out_slow_evidence_provider_as_uncertain() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    config = VerificationConfig(
        evidence=EvidenceConfig(timeout_ms=1),
    )

    verified = asyncio.run(
        VerificationService(
            SlowEvidenceProvider(),
            config=config,
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert "evidence_provider_unavailable" in verified.failed_constraints


def test_service_preserves_evidence_provider_cancellation() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            VerificationService(
                CancelledEvidenceProvider(),
                gate=AlwaysVerifyGate(),
            ).verify(query, result, [])
        )


def test_service_disabled_skips_evidence_and_uses_upstream_confidence() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.73,
    )
    provider = FakeEvidenceProvider(
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="frame-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
        )
    )

    verified = asyncio.run(
        VerificationService(
            provider,
            config=VerificationConfig(enabled=False),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "accepted"
    assert verified.confidence == 0.73
    assert provider.build_count == 0


def test_service_empty_focus_does_not_expand_to_all_claims() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS", ocr_constraints=["HCMC"])
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )

    verified = asyncio.run(
        VerificationService(
            gate=AlwaysVerifyGate(),
            planner=EmptyFocusPlanner(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert "no_verifiable_focus_claim" in verified.failed_constraints


def test_service_unsupported_focus_does_not_load_evidence() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    provider = FakeEvidenceProvider(
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="frame-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
        )
    )

    verified = asyncio.run(
        VerificationService(
            provider,
            gate=AlwaysVerifyGate(),
            planner=UnsupportedFocusPlanner(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert verified.failed_constraints == ["claim-visual-1"]
    assert provider.build_count == 0


def test_service_mixed_focus_verifies_supported_claims() -> None:
    query = StructuredQuery(query_id="query-1", task="KIS")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    provider = FakeEvidenceProvider(
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="frame-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
            text_evidence=[
                TextEvidence(
                    evidence_id="ocr-1",
                    evidence_type="ocr",
                    text="HCMC",
                    start_ms=1000,
                    end_ms=1000,
                )
            ],
        )
    )

    verified = asyncio.run(
        VerificationService(
            provider,
            gate=AlwaysVerifyGate(),
            planner=MixedFocusPlanner(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert verified.supporting_evidence_ids == ["ocr-1"]
    assert verified.failed_constraints == ["claim-visual-1"]
    assert provider.build_count == 1


def test_service_rejects_task_result_mismatch_before_evidence_io() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    provider = FakeEvidenceProvider(
        VerificationEvidencePack(
            verification_id="ver-1",
            candidate_id="frame-1",
            video_id="video-1",
            start_ms=1000,
            end_ms=2000,
        )
    )

    verified = asyncio.run(
        VerificationService(provider, gate=AlwaysVerifyGate()).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert "task_result_mismatch" in verified.failed_constraints
    assert provider.build_count == 0


def test_service_does_not_accept_vqa_from_unrelated_referenced_text() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="gold medal",
        confidence=0.9,
        evidence_ids=["caption-1"],
        status="answered",
    )
    candidate = RankedCandidateRegion(
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        fusion_score=0.9,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        text_evidence=[
            TextEvidence(
                evidence_id="caption-1",
                evidence_type="caption",
                text="The athlete stands on the podium.",
                start_ms=1000,
                end_ms=2000,
            )
        ],
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [candidate])
    )

    assert verified.status == "uncertain"
    assert "claim-vqa-answer" in verified.failed_constraints
    assert verified.supporting_evidence_ids == []


def test_service_retains_referenced_vqa_evidence_before_text_cap() -> None:
    query = StructuredQuery(query_id="query-1", task="VQA", question="What medal?")
    result = VQAResult(
        answer="gold medal",
        confidence=0.9,
        evidence_ids=["caption-priority"],
        status="answered",
    )
    candidate = RankedCandidateRegion(
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        fusion_score=0.9,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        text_evidence=[
            TextEvidence(
                evidence_id="caption-unrelated",
                evidence_type="caption",
                text="The athlete stands on the podium.",
                start_ms=1000,
                end_ms=2000,
            ),
            TextEvidence(
                evidence_id="caption-priority",
                evidence_type="caption",
                text="The athlete receives a gold medal.",
                start_ms=1000,
                end_ms=2000,
            ),
        ],
    )
    config = VerificationConfig(
        evidence=EvidenceConfig(max_text_items=1),
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            config=config,
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [candidate])
    )

    assert verified.status == "accepted"
    assert verified.supporting_evidence_ids == ["caption-priority"]


def test_service_supports_object_count_only_with_same_frame_observations() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        object_constraints=["at least 2 people"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="frame-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        object_evidence=[
            ObjectEvidence(
                evidence_id="object-1",
                frame_id="frame-1",
                class_name="person",
                confidence=0.8,
                start_ms=1500,
                end_ms=1500,
            ),
            ObjectEvidence(
                evidence_id="object-2",
                frame_id="frame-1",
                class_name="person",
                confidence=0.9,
                start_ms=1500,
                end_ms=1500,
            ),
        ],
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "accepted"
    assert verified.supporting_evidence_ids == ["object-1", "object-2"]


def test_service_zero_object_count_is_uncertain_without_crashing() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        object_constraints=["at least 0 people"],
    )
    result = KISResult(
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        representative_frame_id="frame-1",
        score=0.9,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="frame-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
    )

    verified = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
        ).verify(query, result, [])
    )

    assert verified.status == "uncertain"
    assert verified.failed_constraints == ["claim-object-1"]
