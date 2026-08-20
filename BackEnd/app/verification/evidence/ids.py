"""Canonical evidence identity rules for Selective Verifier packs."""

from __future__ import annotations

from typing import Any


def frame_evidence_id(item: Any) -> str:
    return item.frame_id


def ocr_evidence_id(item: Any) -> str:
    return f"ocr-{item.frame_id}-{item.n}"


def asr_evidence_id(item: Any) -> str:
    return item.segment_id


def caption_evidence_id(item: Any, index: int) -> str:
    if item.caption_id is not None:
        return f"caption-{item.caption_id}"
    target_id = item.frame_id or item.clip_id or item.shot_id or "unknown"
    return f"caption-{target_id}-{index}"


def object_evidence_id(item: Any, index: int) -> str:
    if item.detection_id is not None:
        return f"object-{item.detection_id}"
    return f"object-{item.frame_id}-{item.class_id}-{index}"


def track_evidence_id(item: Any, index: int) -> str:
    if item.track_id is not None:
        return f"track-{item.track_id}"
    return f"track-{item.shot_id}-{item.class_id}-{index}"
