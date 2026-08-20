from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.verification.contracts import (
    FrameEvidence,
    ObjectEvidence,
    ObjectCountSpec,
    TextEvidence,
    TrackEvidence,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.deterministic.asr_verifier import AsrExactVerifier
from BackEnd.app.verification.deterministic.negative_constraint_verifier import (
    NegativeConstraintVerifier,
)
from BackEnd.app.verification.deterministic.object_verifier import ObjectPresenceVerifier
from BackEnd.app.verification.deterministic.ocr_verifier import OcrExactVerifier
from BackEnd.app.verification.deterministic.temporal_verifier import TemporalConstraintVerifier
from BackEnd.app.verification.deterministic.vqa_answer_verifier import (
    VqaAnswerEvidenceVerifier,
)
from BackEnd.app.verification.enums import ClaimImportance, ClaimStatus, ClaimType


def evidence_pack() -> VerificationEvidencePack:
    return VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=5000,
    )


def test_ocr_exact_match_is_supported() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="ocr-1",
                    evidence_type="ocr",
                    text="Welcome to HCMC",
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OCR_EXACT,
        text="HCMC",
        importance=ClaimImportance.HARD,
    )

    result = OcrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["ocr-1"]


def test_ocr_exact_does_not_match_inside_another_word() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="ocr-1",
                    evidence_type="ocr",
                    text="woman",
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OCR_EXACT,
        text="man",
        importance=ClaimImportance.HARD,
    )

    result = OcrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN
    assert result.evidence_ids == []


def test_ocr_whitespace_only_claim_is_unknown() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="ocr-1",
                    evidence_type="ocr",
                    text="any visible text",
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OCR_EXACT,
        text="   ",
        importance=ClaimImportance.HARD,
    )

    result = OcrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN
    assert result.evidence_ids == []


def test_ocr_fuzzy_match_remains_configurable() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="ocr-1",
                    evidence_type="ocr",
                    text="championshp",
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OCR_EXACT,
        text="championship",
        importance=ClaimImportance.HARD,
    )

    result = OcrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["ocr-1"]


def test_asr_exact_match_is_supported() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="asr-1",
                    evidence_type="asr",
                    text="he receives the gold medal",
                    start_ms=1000,
                    end_ms=2000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.ASR_EXACT,
        text="gold medal",
        importance=ClaimImportance.HARD,
    )

    result = AsrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["asr-1"]


def test_asr_exact_does_not_match_inside_another_word() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="asr-1",
                    evidence_type="asr",
                    text="the party starts now",
                    start_ms=1000,
                    end_ms=2000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.ASR_EXACT,
        text="art",
        importance=ClaimImportance.HARD,
    )

    result = AsrExactVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_missing_object_evidence_is_unknown_not_contradicted() -> None:
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="ice cream",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.UNKNOWN


def test_object_presence_is_supported_when_detection_matches() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    class_name="person",
                    confidence=0.91,
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="person",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["object-1"]


def test_object_presence_does_not_match_opaque_class_id_by_default() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    class_id="0",
                    class_name="0",
                    confidence=0.91,
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="person",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_object_presence_does_not_match_inside_another_label() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    class_name="woman",
                    confidence=0.91,
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="man",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_object_presence_can_be_supported_by_track_evidence() -> None:
    pack = evidence_pack().model_copy(
        update={
            "track_evidence": [
                TrackEvidence(
                    evidence_id="track-1",
                    class_name="person",
                    observation_count=3,
                    confidence=0.82,
                    start_ms=1000,
                    end_ms=4000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="person",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["track-1"]


def test_object_count_needs_enough_detections_in_one_frame() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    frame_id="frame-1",
                    class_name="person",
                    confidence=0.9,
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-count",
        claim_type=ClaimType.OBJECT_COUNT,
        text="at least 2 persons",
        importance=ClaimImportance.HARD,
        metadata={
            "count_spec": ObjectCountSpec(
                operator="at_least",
                expected_count=2,
                object_label="person",
            )
        },
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_object_count_uses_simultaneous_detections() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    frame_id="frame-1",
                    class_name="person",
                    confidence=0.9,
                    start_ms=1000,
                    end_ms=1000,
                ),
                ObjectEvidence(
                    evidence_id="object-2",
                    frame_id="frame-1",
                    class_name="person",
                    confidence=0.8,
                    start_ms=1000,
                    end_ms=1000,
                ),
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-count",
        claim_type=ClaimType.OBJECT_COUNT,
        text="at least 2 persons",
        importance=ClaimImportance.HARD,
        metadata={
            "count_spec": ObjectCountSpec(
                operator="at_least",
                expected_count=2,
                object_label="person",
            )
        },
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["object-1", "object-2"]


def test_object_count_does_not_sum_detections_across_frames() -> None:
    evidence_pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=2000,
        object_evidence=[
            ObjectEvidence(
                evidence_id="object-1",
                frame_id="frame-1",
                class_name="person",
                confidence=0.9,
                start_ms=1200,
                end_ms=1200,
            ),
            ObjectEvidence(
                evidence_id="object-2",
                frame_id="frame-2",
                class_name="person",
                confidence=0.9,
                start_ms=1800,
                end_ms=1800,
            ),
        ],
    )
    claim = VerificationClaim(
        claim_id="claim-object-count",
        claim_type=ClaimType.OBJECT_COUNT,
        text="at least 2 people",
        importance=ClaimImportance.HARD,
        metadata={
            "count_spec": ObjectCountSpec(
                operator="at_least",
                expected_count=2,
                object_label="person",
            )
        },
    )

    result = ObjectPresenceVerifier().verify(claim, evidence_pack)

    assert result.status == ClaimStatus.UNKNOWN
    assert result.evidence_ids == []


def test_object_count_does_not_sum_non_overlapping_tracks() -> None:
    evidence_pack = VerificationEvidencePack(
        verification_id="ver-1",
        candidate_id="candidate-1",
        video_id="video-1",
        start_ms=1000,
        end_ms=4000,
        track_evidence=[
            TrackEvidence(
                evidence_id="track-1",
                class_name="person",
                observation_count=3,
                confidence=0.9,
                start_ms=1000,
                end_ms=1900,
            ),
            TrackEvidence(
                evidence_id="track-2",
                class_name="person",
                observation_count=3,
                confidence=0.9,
                start_ms=2100,
                end_ms=3000,
            ),
        ],
    )
    claim = VerificationClaim(
        claim_id="claim-object-count",
        claim_type=ClaimType.OBJECT_COUNT,
        text="at least 2 people",
        importance=ClaimImportance.HARD,
        metadata={
            "count_spec": ObjectCountSpec(
                operator="at_least",
                expected_count=2,
                object_label="person",
            )
        },
    )

    result = ObjectPresenceVerifier().verify(claim, evidence_pack)

    assert result.status == ClaimStatus.UNKNOWN
    assert result.evidence_ids == []


def test_object_count_with_zero_metadata_is_unknown() -> None:
    claim = VerificationClaim(
        claim_id="claim-object-count",
        claim_type=ClaimType.OBJECT_COUNT,
        text="at least 0 people",
        importance=ClaimImportance.HARD,
        metadata={
            "count_spec": {
                "operator": "at_least",
                "expected_count": 0,
                "object_label": "person",
            }
        },
    )

    result = ObjectPresenceVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.UNKNOWN
    assert result.evidence_ids == []


def test_object_presence_ignores_low_confidence_evidence() -> None:
    pack = evidence_pack().model_copy(
        update={
            "object_evidence": [
                ObjectEvidence(
                    evidence_id="object-1",
                    class_name="person",
                    confidence=0.1,
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.OBJECT_PRESENCE,
        text="person",
        importance=ClaimImportance.HARD,
    )

    result = ObjectPresenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_vqa_answer_can_be_supported_by_referenced_caption_evidence() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="caption-1",
                    evidence_type="caption",
                    text="A man receives a gold medal.",
                    start_ms=1000,
                    end_ms=5000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-vqa-answer",
        claim_type=ClaimType.VQA_ANSWER_CLAIM,
        text="gold medal",
        importance=ClaimImportance.HARD,
        metadata={"evidence_ids": ["caption-1"]},
    )

    result = VqaAnswerEvidenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.SUPPORTED
    assert result.evidence_ids == ["caption-1"]


def test_vqa_answer_is_unknown_when_referenced_text_is_unrelated() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="caption-1",
                    evidence_type="caption",
                    text="A silver car is parked outside.",
                    start_ms=1000,
                    end_ms=5000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-vqa-answer",
        claim_type=ClaimType.VQA_ANSWER_CLAIM,
        text="gold medal",
        importance=ClaimImportance.HARD,
        metadata={"evidence_ids": ["caption-1"]},
    )

    result = VqaAnswerEvidenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_vqa_answer_is_unknown_for_visual_only_reference() -> None:
    pack = evidence_pack().model_copy(
        update={
            "frame_evidence": [
                FrameEvidence(
                    evidence_id="frame-1",
                    frame_id="frame-1",
                    start_ms=1000,
                    end_ms=1000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-vqa-answer",
        claim_type=ClaimType.VQA_ANSWER_CLAIM,
        text="gold medal",
        importance=ClaimImportance.HARD,
        metadata={"evidence_ids": ["frame-1"]},
    )

    result = VqaAnswerEvidenceVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_negative_constraint_is_contradicted_by_caption_evidence() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="caption-1",
                    evidence_type="caption",
                    text="The ceremony is indoor.",
                    start_ms=1000,
                    end_ms=5000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-negative-1",
        claim_type=ClaimType.NEGATIVE_CONSTRAINT,
        text="indoor",
        importance=ClaimImportance.HARD,
    )

    result = NegativeConstraintVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.CONTRADICTED
    assert result.evidence_ids == ["caption-1"]


def test_negative_constraint_negated_mention_remains_unknown() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="caption-1",
                    evidence_type="caption",
                    text="The ceremony is not indoor.",
                    start_ms=1000,
                    end_ms=5000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-negative-1",
        claim_type=ClaimType.NEGATIVE_CONSTRAINT,
        text="indoor",
        importance=ClaimImportance.HARD,
    )

    result = NegativeConstraintVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_negative_constraint_does_not_match_inside_another_word() -> None:
    pack = evidence_pack().model_copy(
        update={
            "text_evidence": [
                TextEvidence(
                    evidence_id="caption-1",
                    evidence_type="caption",
                    text="A carpet is visible.",
                    start_ms=1000,
                    end_ms=5000,
                )
            ]
        }
    )
    claim = VerificationClaim(
        claim_id="claim-negative-1",
        claim_type=ClaimType.NEGATIVE_CONSTRAINT,
        text="car",
        importance=ClaimImportance.HARD,
    )

    result = NegativeConstraintVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.UNKNOWN


def test_temporal_order_contradiction_is_deterministic() -> None:
    pack = evidence_pack()
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.TEMPORAL_ORDER,
        text="E1 before E2",
        importance=ClaimImportance.HARD,
        metadata={
            "constraint": TemporalConstraint(before="E1", after="E2"),
            "events": {
                "E1": {"start_ms": 20000, "end_ms": 21000},
                "E2": {"start_ms": 10000, "end_ms": 11000},
            },
        },
    )

    result = TemporalConstraintVerifier().verify(claim, pack)

    assert result.status == ClaimStatus.CONTRADICTED


def test_temporal_reversed_order_is_contradicted_even_when_overlap_allowed() -> None:
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.TEMPORAL_ORDER,
        text="E1 before E2",
        importance=ClaimImportance.HARD,
        metadata={
            "constraint": TemporalConstraint(before="E1", after="E2", allow_overlap=True),
            "events": {
                "E1": {"start_ms": 20000, "end_ms": 21000},
                "E2": {"start_ms": 10000, "end_ms": 11000},
            },
        },
    )

    result = TemporalConstraintVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.CONTRADICTED


def test_temporal_partial_overlap_can_be_supported_when_allowed() -> None:
    claim = VerificationClaim(
        claim_id="claim-1",
        claim_type=ClaimType.TEMPORAL_ORDER,
        text="E1 before E2",
        importance=ClaimImportance.HARD,
        metadata={
            "constraint": TemporalConstraint(before="E1", after="E2", allow_overlap=True),
            "events": {
                "E1": {"start_ms": 10000, "end_ms": 13000},
                "E2": {"start_ms": 12000, "end_ms": 15000},
            },
        },
    )

    result = TemporalConstraintVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.SUPPORTED


def test_temporal_order_does_not_apply_gap_limits() -> None:
    claim = VerificationClaim(
        claim_id="claim-order",
        claim_type=ClaimType.TEMPORAL_ORDER,
        text="E1 before E2",
        importance=ClaimImportance.HARD,
        metadata={
            "constraint": TemporalConstraint(before="E1", after="E2", max_gap_ms=1000),
            "events": {
                "E1": {"start_ms": 1000, "end_ms": 2000},
                "E2": {"start_ms": 5000, "end_ms": 6000},
            },
        },
    )

    result = TemporalConstraintVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.SUPPORTED


def test_temporal_gap_applies_gap_limits_after_order_passes() -> None:
    claim = VerificationClaim(
        claim_id="claim-gap",
        claim_type=ClaimType.TEMPORAL_GAP,
        text="E1 gap E2",
        importance=ClaimImportance.HARD,
        metadata={
            "constraint": TemporalConstraint(before="E1", after="E2", max_gap_ms=1000),
            "events": {
                "E1": {"start_ms": 1000, "end_ms": 2000},
                "E2": {"start_ms": 5000, "end_ms": 6000},
            },
        },
    )

    result = TemporalConstraintVerifier().verify(claim, evidence_pack())

    assert result.status == ClaimStatus.CONTRADICTED
