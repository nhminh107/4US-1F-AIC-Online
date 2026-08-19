from BackEnd.app.services.evidence_service import get_evidence_bundle


class RecordingDatabaseManager:
    def __init__(self) -> None:
        self.calls = []

    def get_evidence_by_video_id_and_time(
        self,
        video_id,
        start_ms,
        end_ms,
        *,
        modalities=None,
        limits=None,
        priority_evidence_ids=None,
    ):
        self.calls.append(
            {
                "video_id": video_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "modalities": modalities,
                "limits": limits,
                "priority_evidence_ids": priority_evidence_ids,
            }
        )
        return [[], [], [], [], [], [], [], []]


def test_evidence_service_forwards_optional_query_controls() -> None:
    db_mng = RecordingDatabaseManager()

    bundle = get_evidence_bundle(
        "video-1",
        1000,
        2000,
        db_mng,
        modalities={"asr", "caption"},
        limits={"asr": 5, "caption": 3},
        priority_evidence_ids={"asr-9", "caption-7"},
    )

    assert bundle.video_id == "video-1"
    assert db_mng.calls == [
        {
            "video_id": "video-1",
            "start_ms": 1000,
            "end_ms": 2000,
            "modalities": {"asr", "caption"},
            "limits": {"asr": 5, "caption": 3},
            "priority_evidence_ids": {"asr-9", "caption-7"},
        }
    ]
