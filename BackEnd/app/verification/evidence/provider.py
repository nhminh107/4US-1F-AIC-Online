"""Adapter from existing evidence services to verifier evidence packs."""

from __future__ import annotations

from asyncio import to_thread
from typing import Any

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Database.sql_models import ClassID
from BackEnd.app.services.evidence_service import get_evidence_bundle
from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    FrameEvidence,
    ObjectEvidence,
    TextEvidence,
    TrackEvidence,
    VerificationEvidencePack,
    VerificationPlan,
)
from BackEnd.app.verification.evidence.evidence_pack_builder import bound_evidence_pack
from BackEnd.app.verification.evidence.ids import (
    asr_evidence_id,
    caption_evidence_id,
    frame_evidence_id,
    object_evidence_id,
    ocr_evidence_id,
    track_evidence_id,
)


class DatabaseEvidenceProvider:
    def __init__(
        self,
        db_mng: PostgreManager,
        *,
        config: VerificationConfig | None = None,
        class_name_by_id: dict[str, str] | None = None,
    ) -> None:
        self.db_mng = db_mng
        self.config = config or VerificationConfig()
        self.class_name_by_id = class_name_by_id or {}
        self._class_name_cache: dict[str, str] = {}

    async def build_evidence_pack(
        self,
        plan: VerificationPlan,
    ) -> VerificationEvidencePack:
        if (
            plan.target_video_id is None
            or plan.target_start_ms is None
            or plan.target_end_ms is None
        ):
            raise ValueError("Verification plan does not contain canonical target context.")

        start_ms = plan.target_start_ms
        end_ms = plan.target_end_ms
        video_id = plan.target_video_id
        modalities = _query_modalities(plan.required_evidence_types)
        limits = _query_limits(modalities, self.config)
        priority_evidence_ids = {
            evidence_id
            for claim in plan.claims
            for evidence_id in claim.metadata.get("evidence_ids", [])
            if isinstance(evidence_id, str)
        }
        bundle = await to_thread(
            get_evidence_bundle,
            video_id,
            start_ms,
            end_ms,
            self.db_mng,
            modalities=modalities,
            limits=limits,
            priority_evidence_ids=priority_evidence_ids,
        )
        class_names = await to_thread(
            self._resolve_class_names,
            [*bundle.objects, *bundle.tracks],
        )
        text_evidence = [
            TextEvidence(
                evidence_id=ocr_evidence_id(item),
                evidence_type="ocr",
                text=item.text,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            for item in bundle.ocr
        ]
        text_evidence.extend(
            TextEvidence(
                evidence_id=asr_evidence_id(item),
                evidence_type="asr",
                text=item.text,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
            )
            for item in bundle.asr
        )
        text_evidence.extend(
            TextEvidence(
                evidence_id=caption_evidence_id(item, index),
                evidence_type="caption",
                text=item.caption_text,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            for index, item in enumerate(bundle.captions, start=1)
        )
        frame_evidence = [
            FrameEvidence(
                evidence_id=frame_evidence_id(item),
                frame_id=item.frame_id,
                frame_path=str(item.frame_path) if item.frame_path is not None else None,
                start_ms=item.timestamp_ms,
                end_ms=item.timestamp_ms,
            )
            for item in bundle.frames
        ]
        object_evidence = [
            ObjectEvidence(
                evidence_id=object_evidence_id(item, index),
                frame_id=item.frame_id,
                class_id=item.class_id,
                class_name=class_names[item.class_id],
                confidence=item.confidence,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            for index, item in enumerate(bundle.objects, start=1)
        ]
        track_evidence = [
            TrackEvidence(
                evidence_id=track_evidence_id(item, index),
                class_name=class_names[item.class_id],
                observation_count=item.observation_count,
                confidence=item.avg_confidence,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
            )
            for index, item in enumerate(bundle.tracks, start=1)
        ]
        pack = VerificationEvidencePack(
            verification_id=plan.verification_id,
            candidate_id=plan.target_result_id,
            video_id=bundle.video_id,
            start_ms=bundle.start_ms,
            end_ms=bundle.end_ms,
            frame_evidence=frame_evidence,
            text_evidence=text_evidence,
            object_evidence=object_evidence,
            track_evidence=track_evidence,
        )
        return bound_evidence_pack(
            pack,
            self.config,
            plan.required_evidence_types,
            priority_evidence_ids,
        )

    def _resolve_class_names(self, items: list[Any]) -> dict[str, str]:
        return {item.class_id: self._class_name_for(item) for item in items}

    def _class_name_for(self, item: Any) -> str:
        class_id = item.class_id
        if class_id in self.class_name_by_id:
            return self.class_name_by_id[class_id]
        if class_id in self._class_name_cache:
            return self._class_name_cache[class_id]

        object_class = getattr(item, "object_class", None)
        class_name = getattr(object_class, "class_name", None)
        if class_name:
            self._class_name_cache[class_id] = class_name
            return class_name

        session_factory = getattr(self.db_mng, "session_factory", None)
        if session_factory is not None:
            with session_factory() as session:
                class_record = session.get(ClassID, class_id)
                if class_record is not None:
                    self._class_name_cache[class_id] = class_record.class_name
                    return class_record.class_name

        self._class_name_cache[class_id] = class_id
        return class_id


def _query_modalities(required_evidence_types: list[str]) -> set[str]:
    modalities = set(required_evidence_types)
    if "object" in modalities:
        modalities.add("track")
    return modalities


def _query_limits(
    modalities: set[str] | None,
    config: VerificationConfig,
) -> dict[str, int] | None:
    if modalities is None:
        return None
    limits_by_type = {
        "frame": config.evidence.max_frames,
        "ocr": config.evidence.max_text_items,
        "asr": config.evidence.max_text_items,
        "caption": config.evidence.max_text_items,
        "object": config.evidence.max_objects,
        "track": config.evidence.max_tracks,
    }
    return {
        modality: limits_by_type[modality]
        for modality in modalities
        if modality in limits_by_type
    }
