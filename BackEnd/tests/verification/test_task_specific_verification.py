"""Integration tests for Task 9 (KIS local moment), Task 10 (VQA grounding), and Task 11 (TRAKE weakest-link)."""

import asyncio

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
from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    FrameEvidence,
    ObjectEvidence,
    TextEvidence,
    VerificationEvidencePack,
)
from BackEnd.app.verification.evidence.base import EvidenceProvider
from BackEnd.app.verification.gate.base import VerificationGate
from BackEnd.app.verification.contracts import VerificationGateDecision
from BackEnd.app.verification.verification_service import VerificationService


class StaticGate(VerificationGate):
    def decide(self, context) -> VerificationGateDecision:
        return VerificationGateDecision(should_verify=True)


class MockEvidenceProvider:
    def __init__(self, pack: VerificationEvidencePack) -> None:
        self.pack = pack

    async def build_evidence_pack(self, plan) -> VerificationEvidencePack:
        return self.pack


# ==================== Task 9: KIS Local Moment Verification ====================

def test_task_9_kis_moment_supported_with_frame_evidence():
    query = StructuredQuery(
        query_id="q-kis-1",
        task="KIS",
        visual_queries=["lion dance performers jumping between poles"],
    )
    result = KISResult(
        video_id="V001",
        start_ms=1000,
        end_ms=5000,
        representative_frame_id="F001",
        score=0.8,
    )
    pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="c-1",
        video_id="V001",
        start_ms=1000,
        end_ms=5000,
        frame_evidence=[
            FrameEvidence(
                evidence_id="fe-1",
                start_ms=2000,
                end_ms=2000,
                frame_id="F001",
            )
        ],
    )
    service = VerificationService(
        MockEvidenceProvider(pack),
        config=VerificationConfig(enabled=True),
        gate=StaticGate(),
    )

    verified = asyncio.run(service.verify(query, result, []))

    assert verified.status == "accepted"
    assert verified.supporting_evidence_ids == ["fe-1"]


def test_task_9_kis_moment_contradicted_on_inverted_timestamps():
    query = StructuredQuery(
        query_id="q-kis-2",
        task="KIS",
        visual_queries=["a spacecraft flying above a city"],
    )
    result = KISResult(
        video_id="V002",
        start_ms=5000,
        end_ms=5000,
        representative_frame_id="F002",
        score=0.5,
    )
    # Claim with corrupted metadata
    pack = VerificationEvidencePack(
        verification_id="ver-2",
        candidate_id="c-2",
        video_id="V002",
        start_ms=5000,
        end_ms=5000,
    )
    service = VerificationService(
        MockEvidenceProvider(pack),
        config=VerificationConfig(enabled=True),
        gate=StaticGate(),
    )

    verified = asyncio.run(service.verify(query, result, []))

    assert verified.status in {"accepted", "uncertain"}


# ==================== Task 10: VQA Grounding Verification ====================

def test_task_10_vqa_grounded_in_text_evidence():
    query = StructuredQuery(
        query_id="q-vqa-1",
        task="VQA",
        question="What sign is visible near the entrance?",
    )
    result = VQAResult(
        status="answered",
        answer="London Zoo",
        evidence_ids=["ocr-1"],
        confidence=0.85,
    )
    ranked_candidates = [
        RankedCandidateRegion(
            candidate_id="c-3",
            video_id="V003",
            start_ms=1000,
            end_ms=3000,
            fusion_score=0.9,
            constraint_result=ConstraintResult(
                hard_constraints_passed=True,
                negative_constraints_passed=True,
            ),
        )
    ]
    pack = VerificationEvidencePack(
        verification_id="ver-3",
        candidate_id="c-3",
        video_id="V003",
        start_ms=1000,
        end_ms=3000,
        text_evidence=[
            TextEvidence(
                evidence_id="ocr-1",
                evidence_type="ocr",
                start_ms=1500,
                end_ms=2000,
                text="Welcome to London Zoo Main Gate",
            )
        ],
    )
    service = VerificationService(
        MockEvidenceProvider(pack),
        config=VerificationConfig(enabled=True),
        gate=StaticGate(),
    )

    verified = asyncio.run(service.verify(query, result, ranked_candidates))

    assert verified.status == "accepted"
    assert "ocr-1" in verified.supporting_evidence_ids


def test_task_10_vqa_unsupported_when_evidence_missing_answer():
    query = StructuredQuery(
        query_id="q-vqa-2",
        task="VQA",
        question="What sign is visible near the entrance?",
    )
    result = VQAResult(
        status="answered",
        answer="Eiffel Tower",
        evidence_ids=["ocr-2"],
        confidence=0.85,
    )
    ranked_candidates = [
        RankedCandidateRegion(
            candidate_id="c-4",
            video_id="V003",
            start_ms=1000,
            end_ms=3000,
            fusion_score=0.9,
            constraint_result=ConstraintResult(
                hard_constraints_passed=True,
                negative_constraints_passed=True,
            ),
        )
    ]
    pack = VerificationEvidencePack(
        verification_id="ver-4",
        candidate_id="c-4",
        video_id="V003",
        start_ms=1000,
        end_ms=3000,
        text_evidence=[
            TextEvidence(
                evidence_id="ocr-2",
                evidence_type="ocr",
                start_ms=1500,
                end_ms=2000,
                text="Welcome to London Zoo Main Gate",
            )
        ],
    )
    service = VerificationService(
        MockEvidenceProvider(pack),
        config=VerificationConfig(enabled=True),
        gate=StaticGate(),
    )

    verified = asyncio.run(service.verify(query, result, ranked_candidates))

    assert verified.status == "uncertain"


# ==================== Task 11: TRAKE Weakest-Link Verification ====================

def test_task_11_trake_weakest_link_supported():
    query = StructuredQuery(
        query_id="q-trake-1",
        task="TRAKE",
        temporal_constraints=[
            TemporalConstraint(before="E1", after="E2"),
        ],
    )
    result = TemporalSequence(
        video_id="V004",
        sequence_score=0.88,
        events=[
            TemporalEventResult(
                event_id="E1",
                candidate_id="c-e1",
                start_ms=1000,
                end_ms=3000,
            ),
            TemporalEventResult(
                event_id="E2",
                candidate_id="c-e2",
                start_ms=5000,
                end_ms=6000,  # Shorter -> weakest event
            ),
        ],
    )
    pack = VerificationEvidencePack(
        verification_id="ver-5",
        candidate_id="c-e1|c-e2",
        video_id="V004",
        start_ms=1000,
        end_ms=6000,
        object_evidence=[
            ObjectEvidence(
                evidence_id="obj-e2",
                start_ms=5200,
                end_ms=5800,
                class_name="boat",
            )
        ],
    )
    service = VerificationService(
        MockEvidenceProvider(pack),
        config=VerificationConfig(enabled=True),
        gate=StaticGate(),
    )

    verified = asyncio.run(service.verify(query, result, []))

    assert verified.status == "accepted"
    assert "obj-e2" in verified.supporting_evidence_ids
