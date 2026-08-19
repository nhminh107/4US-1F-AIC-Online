"""Small helper for tests and adapters that need to bound evidence packs."""

from __future__ import annotations

from typing import Protocol, TypeVar

from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import VerificationEvidencePack


class _EvidenceItem(Protocol):
    evidence_id: str


_EvidenceT = TypeVar("_EvidenceT", bound=_EvidenceItem)


def bound_evidence_pack(
    pack: VerificationEvidencePack,
    config: VerificationConfig | None = None,
    required_evidence_types: list[str] | None = None,
    priority_evidence_ids: set[str] | None = None,
) -> VerificationEvidencePack:
    resolved_config = config or VerificationConfig()
    allowed_types = set(required_evidence_types or [])
    priority_ids = priority_evidence_ids or set()
    object_family_allowed = (
        not allowed_types or "object" in allowed_types or "track" in allowed_types
    )
    text_evidence = [
        item
        for item in pack.text_evidence
        if not allowed_types or item.evidence_type in allowed_types
    ]
    frame_evidence = [
        item for item in pack.frame_evidence if not allowed_types or "frame" in allowed_types
    ]
    object_evidence = [
        item
        for item in pack.object_evidence
        if not allowed_types or "object" in allowed_types
    ]
    track_evidence = [item for item in pack.track_evidence if object_family_allowed]

    text_evidence = _prioritize(text_evidence, priority_ids)
    frame_evidence = _prioritize(frame_evidence, priority_ids)
    object_evidence = _prioritize(object_evidence, priority_ids)
    track_evidence = _prioritize(track_evidence, priority_ids)

    omitted = pack.omitted_evidence_count
    omitted += len(pack.text_evidence) - len(text_evidence)
    omitted += len(pack.frame_evidence) - len(frame_evidence)
    omitted += len(pack.object_evidence) - len(object_evidence)
    omitted += len(pack.track_evidence) - len(track_evidence)
    omitted += max(0, len(text_evidence) - resolved_config.evidence.max_text_items)
    omitted += max(0, len(frame_evidence) - resolved_config.evidence.max_frames)
    omitted += max(0, len(object_evidence) - resolved_config.evidence.max_objects)
    omitted += max(0, len(track_evidence) - resolved_config.evidence.max_tracks)

    return pack.model_copy(
        update={
            "text_evidence": text_evidence[: resolved_config.evidence.max_text_items],
            "frame_evidence": frame_evidence[: resolved_config.evidence.max_frames],
            "object_evidence": object_evidence[
                : resolved_config.evidence.max_objects
            ],
            "track_evidence": track_evidence[: resolved_config.evidence.max_tracks],
            "omitted_evidence_count": omitted,
        }
    )


def _prioritize(items: list[_EvidenceT], priority_ids: set[str]) -> list[_EvidenceT]:
    if not priority_ids:
        return items
    return sorted(items, key=lambda item: item.evidence_id not in priority_ids)
