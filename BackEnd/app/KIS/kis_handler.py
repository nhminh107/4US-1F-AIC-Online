"""Module 6A - KIS Handler.

Vai trò (theo ProposalOnlinePipeline.md, muc 10):
    Hybrid Fusion (Module 5) da chon ra duoc "vung thoi gian" (CandidateRegion)
    tot nhat cho tung candidate. KIS Handler tra loi cau hoi: "Trong vung do,
    khoanh khac/frame nao nen duoc dua cho nguoi dung xem hoac submit?".

Input:
    - StructuredQuery: ngu canh cua query goc (hien tai KIS Handler chua doc
      truc tiep field nao ben trong, giu lai de tuong thich contract va de
      mo rong constraint-aware selection sau nay - xem KIS_HANDLER_PIPELINE.md).
    - list[RankedCandidateRegion]: da duoc Module 5 sap xep giam dan theo
      fusion_score.

Output:
    - list[KISResult]: toi da TOP_N_KIS phan tu, giu nguyen thu tu ranking.

KHONG lam trong module nay (dung ranh gioi trach nhiem):
    - Khong hybrid fusion lan 2 (khong tinh lai fusion_score).
    - Khong search lai toan corpus (khong goi FAISS/Elasticsearch).
    - Chi duoc dung Evidence Utility (get_evidence_bundle, get_temporal_neighbors)
      de lay them ngu canh CUC BO quanh candidate da co san.
"""

from __future__ import annotations

from BackEnd.CONFIG import (
    ASR_WEIGHT,
    CLIP_EMBEDDING_WEIGHT,
    DETECT_WEIGHT,
    FRAME_EMBEDDING_WEIGHT,
    KIS_EDGE_RATIO,
    KIS_NEIGHBOR_SHOT_COUNT,
    OCR_WEIGHT,
    RRF_K,
    SHOT_EMBEDDING_WEIGHT,
    TOP_N_KIS,
    TRACK_WEIGHT,
)
from BackEnd.app.contracts.models import (
    EvidenceBundle,
    KISResult,
    RankedCandidateRegion,
    StructuredQuery,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.services.evidence_service import get_evidence_bundle, get_temporal_neighbors

# Trong so tinh dung de tinh "anchor" (moc thoi gian trong tam cua 1 region),
# tai su dung dung tap hang so ma FusionRanking da dung de 2 module nhat quan
# voi nhau. CHI co 7 key (giong han che da biet cua FusionRanking): entity_type
# "video" va "caption" khong co trong bang nay -> se duoc coi trong so = 0 khi
# tinh anchor (xem __compute_anchor_ms), khong bi KeyError.
_ANCHOR_WEIGHT_BY_ENTITY_TYPE: dict[str, float] = {
    "frame": FRAME_EMBEDDING_WEIGHT,
    "ocr": OCR_WEIGHT,
    "shot": SHOT_EMBEDDING_WEIGHT,
    "clip": CLIP_EMBEDDING_WEIGHT,
    "asr": ASR_WEIGHT,
    "object_track": TRACK_WEIGHT,
    "object_detection": DETECT_WEIGHT,
}


class KISHandler:
    """Chon khoanh khac/frame dai dien cho tung RankedCandidateRegion."""

    def __init__(self, db_mng: PostgreManager) -> None:
        # KIS Handler luon can truy van DB that (get_evidence_bundle,
        # get_list_frame_in_shot, ...) nen db_mng la bat buoc - khac voi
        # Aggregator (Module 4) la thuan tuy in-memory nen db_mng optional.
        self.db_mng = db_mng

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(
        self,
        structured_query: StructuredQuery,
        ranked_regions: list[RankedCandidateRegion],
        top_n: int | None = None,
    ) -> list[KISResult]:
        """Chon Top-N candidate tot nhat va build KISResult cho tung candidate.

        structured_query hien khong duoc doc truc tiep trong logic (xem docstring
        dau file) - tham so nay duoc giu lai vi 2 ly do: (1) dung dung chu ky ham
        voi Input/Output cua module trong proposal, (2) la cho de sau nay cam
        them logic constraint-aware (vd uu tien candidate khop negative_constraints)
        ma khong phai doi signature cong khai cua KISHandler.execute().
        """

        if not ranked_regions:
            return []

        n = top_n if top_n is not None else TOP_N_KIS
        top_regions = ranked_regions[:n]

        # Chuan hoa fusion_score (khong bi chan trong [0,1]) ve thang [0,1] de
        # khop rang buoc cua KISResult.score. Min-max tren chinh Top-N dang xu ly
        # (khong can tap validation/calibration) va la phep bien doi don dieu tang
        # nen KHONG lam thay doi thu tu ranking da co.
        normalized_scores = self.__normalize_scores(
            [region.fusion_score for region in top_regions]
        )

        results: list[KISResult] = []
        for region, score in zip(top_regions, normalized_scores):
            frame = self.__resolve_representative_frame(region)
            if frame is None:
                # Offline pipeline chua sinh frame nao cho video/shot lien quan
                # (hoac du lieu chua duoc index). Bo qua candidate nay thay vi
                # lam sap ca batch - dung nguyen tac "khong ep doan khi thieu
                # evidence" (tuong tu status=uncertain cua VQA Handler).
                continue

            results.append(
                KISResult(
                    video_id=region.video_id,
                    # Quy uoc cua TimeRangeModel (DATA_CONTRACTS.md): start_ms
                    # == end_ms bieu dien 1 thoi diem (keyframe). KISResult la
                    # ket qua da "chot" xuong 1 khoanh khac chinh xac nen dung
                    # dung quy uoc nay thay vi giu nguyen ca khoang [start,end]
                    # rong cua CandidateRegion goc.
                    start_ms=frame.timestamp_ms,
                    end_ms=frame.timestamp_ms,
                    representative_frame_id=frame.frame_id,
                    score=score,
                    evidence_ids=self.__collect_evidence_ids(region, frame),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Buoc 1: chuan hoa fusion_score -> score trong [0,1]
    # ------------------------------------------------------------------
    @staticmethod
    def __normalize_scores(fusion_scores: list[float]) -> list[float]:
        if not fusion_scores:
            return []

        lowest = min(fusion_scores)
        highest = max(fusion_scores)

        if highest == lowest:
            # Tat ca candidate dang xet co diem bang nhau (vd chi co 1 candidate)
            # -> khong co gi de so sanh, gan diem toi da cho tat ca.
            return [1.0 for _ in fusion_scores]

        return [(value - lowest) / (highest - lowest) for value in fusion_scores]

    # ------------------------------------------------------------------
    # Buoc 2: Precise Moment Selection - chon frame dai dien cho 1 region
    # ------------------------------------------------------------------
    def __resolve_representative_frame(
        self, region: RankedCandidateRegion
    ) -> FrameMetadata | None:
        anchor_ms = self.__compute_anchor_ms(region)

        # Level 0: lay evidence da resolve san trong dung khoang [start_ms, end_ms]
        # cua region qua Evidence Utility co san (khong phai retrieval corpus-wide).
        bundle = get_evidence_bundle(
            region.video_id, region.start_ms, region.end_ms, self.db_mng
        )
        candidate_frames: list[FrameMetadata] = list(bundle.frames)

        need_expand = not candidate_frames or self.__is_near_edge(
            anchor_ms, region.start_ms, region.end_ms
        )

        if need_expand:
            expanded = self.__expand_frames_via_shots(region, bundle)
            if expanded:
                # Gop danh sach, loai trung theo frame_id (uu tien giu ban ghi
                # da co truoc de khong doi thu tu uu tien khi tie-break).
                seen_frame_ids = {frame.frame_id for frame in candidate_frames}
                candidate_frames += [
                    frame for frame in expanded if frame.frame_id not in seen_frame_ids
                ]

        if not candidate_frames:
            return None

        return self.__nearest_frame(candidate_frames, anchor_ms)

    def __compute_anchor_ms(self, region: RankedCandidateRegion) -> int:
        """Uoc luong moc thoi gian "trong tam" cua region tu cac evidence.

        Dung cong thuc weighted-average voi trong so giong het cach
        FusionRanking cham diem tung evidence (weight_tinh / (RRF_K + rank))
        de anchor phan anh dung evidence nao "manh" nhat trong region, thay vi
        chi lay trung diem hinh hoc [start_ms, end_ms].
        """

        weighted_sum = 0.0
        weight_total = 0.0

        for evidence in region.evidence:
            weight = _ANCHOR_WEIGHT_BY_ENTITY_TYPE.get(evidence.entity_type, 0.0)
            weight /= RRF_K + evidence.rank
            weighted_sum += weight * evidence.start_ms
            weight_total += weight

        if weight_total <= 0:
            # Khong co evidence nao co trong so duong (vd toan bo evidence la
            # "caption"/"video" - 2 entity_type ngoai pham vi weight_mapping).
            # Fallback an toan: lay trung diem region.
            return (region.start_ms + region.end_ms) // 2

        return round(weighted_sum / weight_total)

    @staticmethod
    def __is_near_edge(anchor_ms: int, start_ms: int, end_ms: int) -> bool:
        duration = end_ms - start_ms
        if duration <= 0:
            # Region rong (diem thoi gian) -> luon coi la "sat bien" de kich
            # hoat mo rong tim kiem, tranh chi dua vao dung 1 moc thoi gian.
            return True

        threshold = duration * KIS_EDGE_RATIO
        return (anchor_ms - start_ms) <= threshold or (end_ms - anchor_ms) <= threshold

    def __expand_frames_via_shots(
        self, region: RankedCandidateRegion, bundle: EvidenceBundle
    ) -> list[FrameMetadata]:
        """Neighbor Expansion: mo rong tim frame ra ngoai [start_ms, end_ms].

        Level 1 - lay toan bo frame cua (cac) shot da biet chua trong bundle
        (shot thuong rong hon candidate region rat nhieu nen thuong da du).
        Level 2 - neu shot da biet nhung chua co frame nao trong CSDL, mo rong
        tiep sang shot lien ke (truoc/sau) qua get_temporal_neighbors.
        Level 3 - phuong an cuoi cung: lay toan bo frame cua video (chi khi
        Level 1 va Level 2 deu khong ra ket qua).
        """

        shot_ids = {shot.shot_id for shot in bundle.shots}
        shot_ids |= {clip.shot_id for clip in bundle.clips}

        frames: list[FrameMetadata] = []
        for shot_id in shot_ids:
            frames += self.db_mng.get_list_frame_in_shot(shot_id)

        if not frames and shot_ids:
            for shot_id in shot_ids:
                try:
                    neighbors = get_temporal_neighbors(
                        "shot", shot_id, self.db_mng, neighbor_count=KIS_NEIGHBOR_SHOT_COUNT
                    )
                except ValueError:
                    continue

                for ref in (*neighbors.previous, *neighbors.next):
                    if ref.shot_id:
                        frames += self.db_mng.get_list_frame_in_shot(ref.shot_id)

        if not frames:
            try:
                frames = self.db_mng.get_frame_record_by_video_id(region.video_id)
            except ValueError:
                frames = []

        return frames

    @staticmethod
    def __nearest_frame(frames: list[FrameMetadata], anchor_ms: int) -> FrameMetadata:
        def sort_key(frame: FrameMetadata) -> tuple[int, int, int]:
            return (
                abs(frame.timestamp_ms - anchor_ms),
                # Uu tien frame "official" (do Offline Pipeline chon san, chat
                # luong on dinh hon) khi 2 frame cach anchor bang nhau.
                0 if frame.source == "official" else 1,
                frame.frame_idx,
            )

        return min(frames, key=sort_key)

    # ------------------------------------------------------------------
    # Buoc 3: Prepare KIS Result - gom evidence_ids
    # ------------------------------------------------------------------
    @staticmethod
    def __collect_evidence_ids(
        region: RankedCandidateRegion, frame: FrameMetadata
    ) -> list[str]:
        evidence_ids = [evidence.entity_id for evidence in region.evidence]
        if frame.frame_id not in evidence_ids:
            evidence_ids.append(frame.frame_id)
        return evidence_ids
