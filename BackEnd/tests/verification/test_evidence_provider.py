import asyncio
import threading
from types import SimpleNamespace

import BackEnd.app.verification.evidence.provider as provider_module
from BackEnd.app.contracts.models import EvidenceBundle
from BackEnd.app.contracts.pipeline import (
    CaptionResult,
    ObjectDetectionResult,
    ObjectTrackResult,
    TranscriptSegmentResult,
)
from BackEnd.app.verification.config import EvidenceConfig, VerificationConfig
from BackEnd.app.verification.contracts import VerificationPlan
from BackEnd.app.verification.evidence.provider import DatabaseEvidenceProvider


def test_database_provider_uses_canonical_video_and_window(monkeypatch) -> None:
    calls = []

    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        calls.append((video_id, start_ms, end_ms, db_mng))
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
    )
    db_mng = SimpleNamespace()

    pack = asyncio.run(DatabaseEvidenceProvider(db_mng).build_evidence_pack(plan))

    assert calls == [("video-42", 1000, 2000, db_mng)]
    assert pack.video_id == "video-42"


def test_database_provider_empty_requirements_request_no_modalities(monkeypatch) -> None:
    calls = []

    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        calls.append(kwargs)
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=[],
    )

    asyncio.run(DatabaseEvidenceProvider(SimpleNamespace()).build_evidence_pack(plan))

    assert calls[0]["modalities"] == set()
    assert calls[0]["limits"] == {}


def test_database_provider_resolves_object_class_name_from_mapping(monkeypatch) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            objects=[
                ObjectDetectionResult(
                    frame_id="frame-1",
                    class_id="0",
                    confidence=0.9,
                    x_min=0.1,
                    x_max=0.2,
                    y_min=0.1,
                    y_max=0.2,
                    detection_id=7,
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
    )

    pack = asyncio.run(
        DatabaseEvidenceProvider(
            SimpleNamespace(),
            class_name_by_id={"0": "person"},
        ).build_evidence_pack(plan)
    )

    assert pack.object_evidence[0].class_id == "0"
    assert pack.object_evidence[0].class_name == "person"


def test_database_provider_maps_caption_evidence(monkeypatch) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            captions=[
                CaptionResult(
                    caption_id=11,
                    caption_text="A man receives a gold medal.",
                    model_name="captioner",
                    frame_id="frame-1",
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="VQA",
        target_result_id="candidate-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["caption"],
    )

    pack = asyncio.run(DatabaseEvidenceProvider(SimpleNamespace()).build_evidence_pack(plan))

    assert len(pack.text_evidence) == 1
    assert pack.text_evidence[0].evidence_id == "caption-11"
    assert pack.text_evidence[0].evidence_type == "caption"
    assert pack.text_evidence[0].text == "A man receives a gold medal."


def test_database_provider_filters_and_bounds_evidence(monkeypatch) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            asr=[
                TranscriptSegmentResult(
                    segment_id="asr-1",
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text="first",
                ),
                TranscriptSegmentResult(
                    segment_id="asr-2",
                    video_id=video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text="second",
                ),
            ],
            captions=[
                CaptionResult(
                    caption_id=11,
                    caption_text="caption",
                    model_name="captioner",
                    frame_id="frame-1",
                )
            ],
            objects=[
                ObjectDetectionResult(
                    frame_id="frame-1",
                    class_id="0",
                    confidence=0.9,
                    x_min=0.1,
                    x_max=0.2,
                    y_min=0.1,
                    y_max=0.2,
                    detection_id=7,
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["asr"],
    )
    config = VerificationConfig(
        evidence=EvidenceConfig(max_text_items=1, max_frames=0, max_objects=0)
    )

    pack = asyncio.run(
        DatabaseEvidenceProvider(SimpleNamespace(), config=config).build_evidence_pack(plan)
    )

    assert [item.evidence_id for item in pack.text_evidence] == ["asr-1"]
    assert pack.object_evidence == []
    assert pack.omitted_evidence_count == 3


def test_database_provider_maps_track_evidence_with_resolved_class_name(monkeypatch) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            tracks=[
                ObjectTrackResult(
                    track_id=4,
                    shot_id="shot-1",
                    class_id="0",
                    start_ms=1200,
                    end_ms=1800,
                    observation_count=3,
                    model_name="detector",
                    model_version="1",
                    tracker_name="tracker",
                    tracker_version="1",
                    sampling_fps=1.0,
                    mapping_version="1",
                    avg_confidence=0.8,
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["track"],
    )

    pack = asyncio.run(
        DatabaseEvidenceProvider(
            SimpleNamespace(),
            class_name_by_id={"0": "person"},
        ).build_evidence_pack(plan)
    )

    assert len(pack.track_evidence) == 1
    assert pack.track_evidence[0].evidence_id == "track-4"
    assert pack.track_evidence[0].class_name == "person"
    assert pack.track_evidence[0].observation_count == 3


def test_object_required_type_preserves_track_evidence(monkeypatch) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            tracks=[
                ObjectTrackResult(
                    track_id=4,
                    shot_id="shot-1",
                    class_id="0",
                    start_ms=1200,
                    end_ms=1800,
                    observation_count=3,
                    model_name="detector",
                    model_version="1",
                    tracker_name="tracker",
                    tracker_version="1",
                    sampling_fps=1.0,
                    mapping_version="1",
                    avg_confidence=0.8,
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["object"],
    )

    pack = asyncio.run(
        DatabaseEvidenceProvider(
            SimpleNamespace(),
            class_name_by_id={"0": "person"},
        ).build_evidence_pack(plan)
    )

    assert [item.evidence_id for item in pack.track_evidence] == ["track-4"]


def test_object_detection_without_persisted_id_gets_distinct_fallback_ids(
    monkeypatch,
) -> None:
    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            objects=[
                ObjectDetectionResult(
                    frame_id="frame-1",
                    class_id="0",
                    confidence=0.9,
                    x_min=0.1,
                    x_max=0.2,
                    y_min=0.1,
                    y_max=0.2,
                ),
                ObjectDetectionResult(
                    frame_id="frame-1",
                    class_id="0",
                    confidence=0.8,
                    x_min=0.3,
                    x_max=0.4,
                    y_min=0.3,
                    y_max=0.4,
                ),
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["object"],
    )

    pack = asyncio.run(
        DatabaseEvidenceProvider(
            SimpleNamespace(),
            class_name_by_id={"0": "person"},
        ).build_evidence_pack(plan)
    )

    object_ids = [item.evidence_id for item in pack.object_evidence]
    assert object_ids == ["object-frame-1-0-1", "object-frame-1-0-2"]
    assert len(pack.evidence_ids()) == 2


def test_database_provider_caches_class_name_session_lookup(monkeypatch) -> None:
    main_thread_id = threading.get_ident()

    class FakeSession:
        def __init__(self, calls):
            self.calls = calls

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def get(self, model, class_id):
            self.calls.append((model, class_id, threading.get_ident()))
            return SimpleNamespace(class_name="person")

    class FakeDbManager:
        def __init__(self):
            self.calls = []

        def session_factory(self):
            return FakeSession(self.calls)

    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            objects=[
                ObjectDetectionResult(
                    frame_id="frame-1",
                    class_id="0",
                    confidence=0.9,
                    x_min=0.1,
                    x_max=0.2,
                    y_min=0.1,
                    y_max=0.2,
                    detection_id=7,
                ),
                ObjectDetectionResult(
                    frame_id="frame-2",
                    class_id="0",
                    confidence=0.8,
                    x_min=0.3,
                    x_max=0.4,
                    y_min=0.3,
                    y_max=0.4,
                    detection_id=8,
                ),
            ],
            tracks=[
                ObjectTrackResult(
                    track_id=4,
                    shot_id="shot-1",
                    class_id="0",
                    start_ms=1200,
                    end_ms=1800,
                    observation_count=3,
                    model_name="detector",
                    model_version="1",
                    tracker_name="tracker",
                    tracker_version="1",
                    sampling_fps=1.0,
                    mapping_version="1",
                    avg_confidence=0.8,
                )
            ],
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="KIS",
        target_result_id="frame-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["object"],
    )
    db_mng = FakeDbManager()

    pack = asyncio.run(DatabaseEvidenceProvider(db_mng).build_evidence_pack(plan))

    assert [item.class_name for item in pack.object_evidence] == ["person", "person"]
    assert pack.track_evidence[0].class_name == "person"
    assert len(db_mng.calls) == 1
    assert db_mng.calls[0][2] != main_thread_id


def test_database_provider_requests_only_required_bounded_modalities(monkeypatch) -> None:
    calls = []
    main_thread_id = threading.get_ident()

    def fake_get_evidence_bundle(video_id, start_ms, end_ms, db_mng, **kwargs):
        calls.append({"thread_id": threading.get_ident(), **kwargs})
        return EvidenceBundle(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    monkeypatch.setattr(provider_module, "get_evidence_bundle", fake_get_evidence_bundle)
    plan = VerificationPlan(
        verification_id="ver-1",
        query_id="query-1",
        task="VQA",
        target_result_id="candidate-1",
        target_video_id="video-42",
        target_start_ms=1000,
        target_end_ms=2000,
        required_evidence_types=["asr", "caption"],
    )
    config = VerificationConfig(
        evidence=EvidenceConfig(max_text_items=4, max_frames=0, max_objects=0)
    )

    asyncio.run(
        DatabaseEvidenceProvider(SimpleNamespace(), config=config).build_evidence_pack(plan)
    )

    assert calls[0]["modalities"] == {"asr", "caption"}
    assert calls[0]["limits"] == {"asr": 4, "caption": 4}
    assert calls[0]["priority_evidence_ids"] == set()
    assert calls[0]["thread_id"] != main_thread_id
