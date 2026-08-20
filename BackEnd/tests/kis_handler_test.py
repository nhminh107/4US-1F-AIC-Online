"""Script kiem tra thu cong cho KIS Handler (Module 6A).

Giong cach lam cua aggregator_test.py va fusion_test_01.py: KHONG dung
pytest/unittest, chi la 1 script chay bang:

    python -m BackEnd.tests.kis_handler_test

KISHandler can PostgreManager that (co ket noi Postgres) de goi cac ham
get_evidence_bundle / get_list_frame_in_shot / get_frame_record_by_video_id.
De test chay duoc OFFLINE (khong can DB that), file nay tu dung mot
"FakePostgreManager" - class gia lap dung cac phuong thuc cong khai ma
KISHandler thuc su goi toi (duck typing, khong ke thua PostgreManager that
vi __init__ that doi hoi DATABASE_URL/engine that).

Rieng get_temporal_neighbors() (dung cho Level 2 - mo rong sang shot lan can)
lai truy van truc tiep bang SQLAlchemy Session that ben trong
evidence_service.py nen KHONG the fake bang duck-typing PostgreManager. O
kich ban can toi Level 2, script se monkeypatch thang ten
`get_temporal_neighbors` da duoc import vao module kis_handler, roi phuc hoi
lai ngay sau do.
"""

from __future__ import annotations

import BackEnd.app.KIS.kis_handler as kis_handler_module
from BackEnd.app.KIS.kis_handler import KISHandler
from BackEnd.app.contracts.models import (
    CandidateEvidence,
    CanonicalEntityRef,
    ConstraintResult,
    RankedCandidateRegion,
    StructuredQuery,
    TemporalNeighbors,
)
from BackEnd.app.contracts.pipeline import ClipWindowMetadata, FrameMetadata, ShotMetadata


# ==============================================================================
# Fake PostgreManager - du lieu gia lap trong bo nho, khong can Postgres that
# ==============================================================================
def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and a_end >= b_start


class FakePostgreManager:
    """Mo phong tap con phuong thuc cua PostgreManager ma KISHandler can."""

    def __init__(self) -> None:
        # video_id -> {"shots": [...], "clips": [...], "frames": [...]}
        self._by_video: dict[str, dict[str, list]] = {}
        # shot_id -> danh sach frame thuoc shot do (dung cho Level 1/2)
        self._frames_by_shot: dict[str, list[FrameMetadata]] = {}
        # video_id -> toan bo frame cua video (dung cho Level 3)
        self._frames_by_video: dict[str, list[FrameMetadata]] = {}

    def seed_video(
        self,
        video_id: str,
        *,
        shots: list[ShotMetadata] = (),
        clips: list[ClipWindowMetadata] = (),
        frames_in_window: list[FrameMetadata] = (),
    ) -> None:
        """Khai bao du lieu se duoc get_evidence_by_video_id_and_time tra ve."""

        self._by_video[video_id] = {
            "shots": list(shots),
            "clips": list(clips),
            "frames": list(frames_in_window),
        }

    def seed_shot_frames(self, shot_id: str, frames: list[FrameMetadata]) -> None:
        self._frames_by_shot[shot_id] = frames

    def seed_video_frames(self, video_id: str, frames: list[FrameMetadata]) -> None:
        self._frames_by_video[video_id] = frames

    # -- Cac phuong thuc that su duoc kis_handler.py / evidence_service.py goi --

    def get_evidence_by_video_id_and_time(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
        **_kwargs,
    ) -> list[list]:
        data = self._by_video.get(video_id, {"shots": [], "clips": [], "frames": []})
        shots = [s for s in data["shots"] if _overlaps(s.start_ms, s.end_ms, start_ms, end_ms)]
        clips = [c for c in data["clips"] if _overlaps(c.start_ms, c.end_ms, start_ms, end_ms)]
        frames = [f for f in data["frames"] if start_ms <= f.timestamp_ms <= end_ms]
        # Thu tu phai khop dung voi PostgreManager that: shots, clips, frames,
        # ocr, asr, captions, objects, tracks.
        return [shots, clips, frames, [], [], [], [], []]

    def get_list_frame_in_shot(self, shot_id: str) -> list[FrameMetadata]:
        return list(self._frames_by_shot.get(shot_id, []))

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        if video_id not in self._frames_by_video:
            raise ValueError(f"Video '{video_id}' does not exist.")
        return list(self._frames_by_video[video_id])


# ==============================================================================
# Helper dung chung
# ==============================================================================
def _dummy_constraint_result() -> ConstraintResult:
    return ConstraintResult(hard_constraints_passed=True, negative_constraints_passed=True)


def _make_frame(frame_id: str, video_id: str, shot_id: str | None, timestamp_ms: int) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        timestamp_ms=timestamp_ms,
        fps=25.0,
        frame_idx=timestamp_ms // 40,  # gia dinh 25fps -> 40ms/frame, chi de co gia tri hop le
        source="official",
    )


_QUERY = StructuredQuery(query_id="Q_TEST", task="KIS")


# ==============================================================================
# Kich ban A - Level 0: bundle da co san frame khop truc tiep trong region
# ==============================================================================
def scenario_a_level0_direct_hit(db: FakePostgreManager) -> None:
    frame_a1 = _make_frame("fA1", "vidA", "shotA1", timestamp_ms=2000)
    db.seed_video(
        "vidA",
        shots=[ShotMetadata(shot_id="shotA1", video_id="vidA", shot_index=0, start_ms=0, end_ms=10000)],
        frames_in_window=[frame_a1],
    )

    region = RankedCandidateRegion(
        candidate_id="c_a1",
        video_id="vidA",
        start_ms=1000,
        end_ms=3000,
        fusion_score=5.0,
        constraint_result=_dummy_constraint_result(),
        evidence=[
            CandidateEvidence(
                source="faiss_frame_embedding",
                entity_type="frame",
                entity_id="fA1",
                start_ms=2000,
                end_ms=2000,
                rank=1,
                raw_score=0.9,
            )
        ],
    )

    results = KISHandler(db_mng=db).execute(_QUERY, [region])

    assert len(results) == 1, "Scenario A: phai resolve duoc dung 1 KISResult"
    result = results[0]
    assert result.representative_frame_id == "fA1", "Scenario A: phai chon dung frame co san trong region"
    assert result.start_ms == result.end_ms == 2000, "Scenario A: KISResult phai la 1 diem thoi gian (keyframe)"
    assert result.score == 1.0, "Scenario A: chi co 1 candidate -> score chuan hoa phai la 1.0"
    print("[OK] Scenario A (Level 0 - direct hit):", result)


# ==============================================================================
# Kich ban B - Level 1: khong co frame evidence trong region, phai mo rong
# sang toan bo frame cua shot da biet (get_list_frame_in_shot)
# ==============================================================================
def scenario_b_level1_expand_via_shot(db: FakePostgreManager) -> None:
    # Frame that su nam NGOAI khoang [start_ms, end_ms] cua region (15000 ms),
    # nhung thuoc shotB1 - shot nay co overlap voi region [5000, 9000].
    frame_b1 = _make_frame("fB1", "vidB", "shotB1", timestamp_ms=15000)
    db.seed_video(
        "vidB",
        shots=[ShotMetadata(shot_id="shotB1", video_id="vidB", shot_index=0, start_ms=0, end_ms=20000)],
        clips=[ClipWindowMetadata(clip_id="clipB1", shot_id="shotB1", start_ms=5000, end_ms=9000)],
        frames_in_window=[],  # KHONG co frame nao trong dung khoang region
    )
    db.seed_shot_frames("shotB1", [frame_b1])

    region = RankedCandidateRegion(
        candidate_id="c_b1",
        video_id="vidB",
        start_ms=5000,
        end_ms=9000,
        fusion_score=3.0,
        constraint_result=_dummy_constraint_result(),
        evidence=[
            CandidateEvidence(
                source="clip_embedding",
                entity_type="clip",
                entity_id="clipB1",
                start_ms=6000,
                end_ms=6000,
                rank=1,
                raw_score=0.8,
            )
        ],
    )

    results = KISHandler(db_mng=db).execute(_QUERY, [region])

    assert len(results) == 1, "Scenario B: phai resolve duoc dung 1 KISResult"
    assert results[0].representative_frame_id == "fB1", "Scenario B: phai tim thay frame qua mo rong Level 1"
    print("[OK] Scenario B (Level 1 - expand via known shot):", results[0])


# ==============================================================================
# Kich ban C - Level 2: shot da biet nhung chua co frame nao trong CSDL,
# phai mo rong tiep sang shot lan can qua get_temporal_neighbors (monkeypatch)
# ==============================================================================
def scenario_c_level2_expand_via_neighbor_shot(db: FakePostgreManager) -> None:
    db.seed_video(
        "vidC",
        shots=[ShotMetadata(shot_id="shotC1", video_id="vidC", shot_index=0, start_ms=0, end_ms=5000)],
        frames_in_window=[],
    )
    db.seed_shot_frames("shotC1", [])  # shot ton tai nhung chua co frame nao
    frame_c2 = _make_frame("fC2", "vidC", "shotC2", timestamp_ms=7000)
    db.seed_shot_frames("shotC2", [frame_c2])

    def fake_get_temporal_neighbors(entity_type, entity_id, db_mng, neighbor_count=1):
        assert entity_type == "shot" and entity_id == "shotC1"
        return TemporalNeighbors(
            previous=[],
            next=[
                CanonicalEntityRef(
                    video_id="vidC", shot_id="shotC2", start_ms=5000, end_ms=10000
                )
            ],
        )

    original_fn = kis_handler_module.get_temporal_neighbors
    kis_handler_module.get_temporal_neighbors = fake_get_temporal_neighbors
    try:
        region = RankedCandidateRegion(
            candidate_id="c_c1",
            video_id="vidC",
            start_ms=1000,
            end_ms=4000,
            fusion_score=2.0,
            constraint_result=_dummy_constraint_result(),
            evidence=[
                CandidateEvidence(
                    source="shot_detector",
                    entity_type="shot",
                    entity_id="shotC1",
                    start_ms=2000,
                    end_ms=2000,
                    rank=1,
                    raw_score=0.7,
                )
            ],
        )

        results = KISHandler(db_mng=db).execute(_QUERY, [region])
    finally:
        # Luon phuc hoi ham that de khong anh huong cac kich ban chay sau.
        kis_handler_module.get_temporal_neighbors = original_fn

    assert len(results) == 1, "Scenario C: phai resolve duoc dung 1 KISResult"
    assert results[0].representative_frame_id == "fC2", "Scenario C: phai tim thay frame qua shot lan can (Level 2)"
    print("[OK] Scenario C (Level 2 - expand via neighbor shot):", results[0])


# ==============================================================================
# Kich ban D - Level 3: khong biet shot/clip nao ca, phuong an cuoi la lay
# toan bo frame cua video (get_frame_record_by_video_id)
# ==============================================================================
def scenario_d_level3_fallback_whole_video(db: FakePostgreManager) -> None:
    frame_d1 = _make_frame("fD1", "vidD", None, timestamp_ms=50000)
    db.seed_video("vidD", frames_in_window=[])  # khong co shot/clip/frame nao khop
    db.seed_video_frames("vidD", [frame_d1])

    region = RankedCandidateRegion(
        candidate_id="c_d1",
        video_id="vidD",
        start_ms=1000,
        end_ms=2000,
        fusion_score=1.0,
        constraint_result=_dummy_constraint_result(),
        evidence=[
            CandidateEvidence(
                source="whisper_asr",
                entity_type="asr",
                entity_id="asrD1",
                start_ms=1500,
                end_ms=1500,
                rank=1,
                raw_score=0.6,
            )
        ],
    )

    results = KISHandler(db_mng=db).execute(_QUERY, [region])

    assert len(results) == 1, "Scenario D: phai resolve duoc dung 1 KISResult"
    assert results[0].representative_frame_id == "fD1", "Scenario D: phai tim thay frame qua Level 3 (toan video)"
    print("[OK] Scenario D (Level 3 - whole-video fallback):", results[0])


# ==============================================================================
# Kich ban E - Khong the resolve: khong co du lieu o bat ky level nao
# -> candidate phai bi bo qua (khong crash ca batch)
# ==============================================================================
def scenario_e_unresolvable_candidate_is_skipped(db: FakePostgreManager) -> None:
    # Co ban ghi video nhung khong seed frame/shot/clip nao ca.
    db.seed_video("vidE", frames_in_window=[])

    region = RankedCandidateRegion(
        candidate_id="c_e1",
        video_id="vidE",
        start_ms=1000,
        end_ms=2000,
        fusion_score=1.0,
        constraint_result=_dummy_constraint_result(),
        evidence=[
            CandidateEvidence(
                source="ocr_search",
                entity_type="ocr",
                entity_id="ocrE1",
                start_ms=1500,
                end_ms=1500,
                rank=1,
                raw_score=0.5,
            )
        ],
    )

    results = KISHandler(db_mng=db).execute(_QUERY, [region])

    assert results == [], "Scenario E: candidate khong the resolve phai bi bo qua, khong duoc trong ket qua"
    print("[OK] Scenario E (unresolvable candidate is skipped): results =", results)


# ==============================================================================
# Kich ban F - Top-N slicing + chuan hoa score tren nhieu candidate cung luc
# ==============================================================================
def scenario_f_top_n_and_score_normalization(db: FakePostgreManager) -> None:
    db.seed_video(
        "vidF",
        shots=[ShotMetadata(shot_id="shotF1", video_id="vidF", shot_index=0, start_ms=0, end_ms=10000)],
        frames_in_window=[
            _make_frame("fF1", "vidF", "shotF1", timestamp_ms=1000),
            _make_frame("fF2", "vidF", "shotF1", timestamp_ms=2000),
            _make_frame("fF3", "vidF", "shotF1", timestamp_ms=3000),
            _make_frame("fF4", "vidF", "shotF1", timestamp_ms=4000),
        ],
    )

    def make_region(candidate_id: str, frame_id: str, timestamp_ms: int, fusion_score: float):
        return RankedCandidateRegion(
            candidate_id=candidate_id,
            video_id="vidF",
            start_ms=timestamp_ms - 200,
            end_ms=timestamp_ms + 200,
            fusion_score=fusion_score,
            constraint_result=_dummy_constraint_result(),
            evidence=[
                CandidateEvidence(
                    source="faiss_frame_embedding",
                    entity_type="frame",
                    entity_id=frame_id,
                    start_ms=timestamp_ms,
                    end_ms=timestamp_ms,
                    rank=1,
                    raw_score=0.9,
                )
            ],
        )

    regions = [
        make_region("c_f1", "fF1", 1000, fusion_score=10.0),  # cao nhat -> score 1.0
        make_region("c_f2", "fF2", 2000, fusion_score=6.0),  # giua -> score 0.5
        make_region("c_f3", "fF3", 3000, fusion_score=2.0),  # thap nhat -> score 0.0
        make_region("c_f4", "fF4", 4000, fusion_score=1.0),  # se bi cat boi top_n=3
    ]

    results = KISHandler(db_mng=db).execute(_QUERY, regions, top_n=3)

    assert len(results) == 3, "Scenario F: top_n=3 -> chi duoc xu ly 3 candidate dau"
    assert [r.representative_frame_id for r in results] == ["fF1", "fF2", "fF3"], (
        "Scenario F: phai giu dung thu tu ranking ban dau"
    )
    assert results[0].score == 1.0 and results[2].score == 0.0, (
        "Scenario F: min-max normalize phai anh xa fusion_score cao nhat/thap nhat ve 1.0/0.0"
    )
    assert abs(results[1].score - 0.5) < 1e-9, "Scenario F: candidate giua phai co score = 0.5"
    print("[OK] Scenario F (Top-N slicing + score normalization):", results)


# ==============================================================================
# Entry point
# ==============================================================================
def main() -> None:
    db = FakePostgreManager()

    scenario_a_level0_direct_hit(db)
    scenario_b_level1_expand_via_shot(db)
    scenario_c_level2_expand_via_neighbor_shot(db)
    scenario_d_level3_fallback_whole_video(db)
    scenario_e_unresolvable_candidate_is_skipped(db)
    scenario_f_top_n_and_score_normalization(db)

    print("\nTat ca kich ban KIS Handler deu PASS.")


if __name__ == "__main__":
    main()
