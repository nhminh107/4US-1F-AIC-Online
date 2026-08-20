import asyncio

from BackEnd.app.contracts.models import KISResult, StructuredQuery
from BackEnd.app.verification.contracts import TextEvidence, VerificationEvidencePack
from BackEnd.app.verification.metrics.logging import build_verification_log
from BackEnd.app.verification.verification_service import VerificationService
from BackEnd.tests.verification.test_verification_service import (
    AlwaysVerifyGate,
    FakeEvidenceProvider,
)


def test_service_emits_content_safe_detailed_diagnostics() -> None:
    query = StructuredQuery(
        query_id="query-1",
        task="KIS",
        ocr_constraints=["PRIVATE-TEXT"],
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
                text="PRIVATE-TEXT",
                start_ms=1000,
                end_ms=1000,
            )
        ],
    )
    payloads = []

    detail = asyncio.run(
        VerificationService(
            FakeEvidenceProvider(pack),
            gate=AlwaysVerifyGate(),
            diagnostic_sink=payloads.append,
        ).verify_detailed(query, result, [])
    )

    assert detail.focus_claim_ids == ["claim-ocr-1"]
    assert detail.evidence_count_by_type == {
        "frame": 0,
        "ocr": 1,
        "asr": 0,
        "caption": 0,
        "object": 0,
        "track": 0,
    }
    assert detail.latency_ms["total"] >= 0
    assert payloads == [build_verification_log(detail)]
    assert "PRIVATE-TEXT" not in repr(payloads[0])
