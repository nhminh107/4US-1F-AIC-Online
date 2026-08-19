"""Deterministic object presence checks."""

from __future__ import annotations

from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    ClaimVerificationResult,
    ObjectCountSpec,
    VerificationClaim,
    VerificationEvidencePack,
)
from BackEnd.app.verification.enums import ClaimStatus, ClaimType
from BackEnd.app.verification.deterministic.text_matching import contains_phrase


class ObjectPresenceVerifier:
    verifier_name = "object_presence"

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    def supports(self, claim: VerificationClaim) -> bool:
        return claim.claim_type in {ClaimType.OBJECT_PRESENCE, ClaimType.OBJECT_COUNT}

    def verify(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        if claim.claim_type == ClaimType.OBJECT_COUNT:
            return self._verify_count(claim, evidence_pack)

        for evidence in [*evidence_pack.object_evidence, *evidence_pack.track_evidence]:
            confidence = self._confidence(evidence.confidence)
            if (
                confidence >= self.config.deterministic.object_min_confidence
                and contains_phrase(evidence.class_name, claim.text)
            ):
                return self._result(
                    claim,
                    ClaimStatus.SUPPORTED,
                    confidence,
                    [evidence.evidence_id],
                )

        status = (
            ClaimStatus.CONTRADICTED
            if self.config.deterministic.allow_object_absence_contradiction
            else ClaimStatus.UNKNOWN
        )
        return self._result(claim, status, 0.0, [])

    def _verify_count(
        self,
        claim: VerificationClaim,
        evidence_pack: VerificationEvidencePack,
    ) -> ClaimVerificationResult:
        raw_spec = claim.metadata.get("count_spec")
        try:
            spec = (
                raw_spec
                if isinstance(raw_spec, ObjectCountSpec)
                else ObjectCountSpec.model_validate(raw_spec)
            )
        except (TypeError, ValueError):
            return self._result(claim, ClaimStatus.UNKNOWN, 0.0, [])
        if spec.expected_count < 1 or spec.operator != "at_least":
            return self._result(claim, ClaimStatus.UNKNOWN, 0.0, [])

        detection_groups: dict[str, list] = {}
        for evidence in evidence_pack.object_evidence:
            confidence = self._confidence(evidence.confidence)
            if (
                evidence.frame_id is not None
                and confidence >= self.config.deterministic.object_min_confidence
                and contains_phrase(evidence.class_name, spec.object_label)
            ):
                detection_groups.setdefault(evidence.frame_id, []).append(evidence)
        best_detections = max(detection_groups.values(), key=len, default=[])

        matching_tracks = [
            evidence
            for evidence in evidence_pack.track_evidence
            if self._confidence(evidence.confidence)
            >= self.config.deterministic.object_min_confidence
            and contains_phrase(evidence.class_name, spec.object_label)
        ]
        best_tracks = max(
            (
                [
                    candidate
                    for candidate in matching_tracks
                    if candidate.start_ms <= track.start_ms <= candidate.end_ms
                ]
                for track in matching_tracks
            ),
            key=len,
            default=[],
        )
        supporting = (
            best_detections if len(best_detections) >= len(best_tracks) else best_tracks
        )
        if len(supporting) < spec.expected_count:
            return self._result(claim, ClaimStatus.UNKNOWN, 0.0, [])

        selected = supporting[: spec.expected_count]
        confidence = min(self._confidence(item.confidence) for item in selected)
        return self._result(
            claim,
            ClaimStatus.SUPPORTED,
            confidence,
            [item.evidence_id for item in selected],
        )

    def _confidence(self, confidence: float | None) -> float:
        if confidence is None:
            return self.config.deterministic.object_missing_confidence
        return confidence

    def _result(
        self,
        claim: VerificationClaim,
        status: ClaimStatus,
        confidence: float,
        evidence_ids: list[str],
    ) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            status=status,
            confidence=confidence,
            importance=claim.importance,
            evidence_ids=evidence_ids,
            verifier_type="deterministic",
            verifier_name=self.verifier_name,
        )
