from __future__ import annotations

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import (
    CandidateReview,
    CoverageCell,
    MomentBand,
    QueryAtom,
)
from BackEnd.app.retrieval_v2.frame_selector import (
    FrameCandidate,
    QueryAwareOfficialFrameSelector,
)


def _atom(atom_id: str, weight: float = 1.0, atom_type: str = "CONTEXT") -> QueryAtom:
    return QueryAtom(
        atom_id=atom_id,
        text=atom_id,
        modality="visual",
        atom_type=atom_type,
        discriminative_weight=weight,
    )


def _hit(
    *,
    entity_id: str,
    atom_id: str,
    timestamp_ms: int,
    rank: int,
    raw_score: float,
) -> SearchHit:
    return SearchHit(
        video_id="V1",
        start_ms=timestamp_ms,
        end_ms=timestamp_ms,
        source="clip",
        entity_type="frame",
        entity_id=entity_id,
        rank=rank,
        raw_score=raw_score,
        atom_id=atom_id,
    )


def _band(
    band_id: str,
    *,
    video_id: str = "V1",
    start_ms: int = 0,
    end_ms: int = 4_000,
    peak_ms: int = 2_000,
    hits: list[SearchHit] | None = None,
    status: str = "UNKNOWN",
    score: float = 1.0,
) -> MomentBand:
    return MomentBand(
        band_id=band_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=end_ms,
        peak_ms=peak_ms,
        coverage={
            "a1": CoverageCell(
                atom_id="a1",
                retrieval_status="RETRIEVED",
                status=status,
                score=0.8,
            )
        },
        hits=hits or [],
        score=score,
    )


class Provider:
    def __init__(self, frames_by_video: dict[str, list[FrameCandidate]]) -> None:
        self.frames_by_video = frames_by_video
        self.calls: list[tuple[str, int, int]] = []

    def get_official_frames(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> list[FrameCandidate]:
        self.calls.append((video_id, start_ms, end_ms))
        return [
            frame
            for frame in self.frames_by_video.get(video_id, [])
            if start_ms <= frame.timestamp_ms <= end_ms
        ]


def _frame(
    frame_idx: int,
    timestamp_ms: int,
    *,
    video_id: str = "V1",
    official: bool = True,
    exists: bool = True,
) -> FrameCandidate:
    return FrameCandidate(
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        img_url=f"https://frames/{video_id}/{frame_idx}.jpg",
        is_official=official,
        exists=exists,
    )


def test_atom_evidence_and_verification_outweigh_small_temporal_prior():
    band = _band(
        "b1",
        hits=[
            _hit(entity_id="h-near", atom_id="a1", timestamp_ms=3_000, rank=1, raw_score=0.95),
            _hit(entity_id="h-center", atom_id="a1", timestamp_ms=2_000, rank=20, raw_score=0.20),
        ],
    )
    provider = Provider({"V1": [_frame(20, 2_000), _frame(30, 3_000)]})
    review = CandidateReview(
        band_id="b1",
        verdict="match",
        confidence=0.9,
        atom_status={"a1": "PASS"},
        video_id="V1",
    )

    selected = QueryAwareOfficialFrameSelector().select(
        bands=[band],
        atoms=[_atom("a1", 1.8)],
        provider=provider,
        reviews=[review],
        moment_confidence={"b1": 0.95},
        limit=2,
    )

    assert [item.frame.frame_idx for item in selected] == [30, 20]
    assert selected[0].score_components["atom_evidence"] > selected[0].score_components["temporal_anchor"]
    assert selected[0].score_components["verification"] > 0


def test_confident_moment_uses_center_out_dense_sweep_when_evidence_is_tied():
    frames = [_frame(index, timestamp) for index, timestamp in enumerate(range(0, 5_000, 1_000))]
    selected = QueryAwareOfficialFrameSelector().select(
        bands=[_band("b1")],
        atoms=[_atom("a1")],
        provider=Provider({"V1": frames}),
        moment_confidence={"b1": 0.9},
        limit=5,
    )

    assert [item.frame.timestamp_ms for item in selected] == [2_000, 1_000, 3_000, 0, 4_000]


def test_uncertain_moments_rescue_diverse_videos_before_dense_fill():
    bands = [
        _band("b1", video_id="V1", score=1.0),
        _band("b2", video_id="V2", score=0.9),
    ]
    provider = Provider(
        {
            "V1": [_frame(1, 2_000), _frame(2, 2_100)],
            "V2": [_frame(3, 2_000, video_id="V2"), _frame(4, 2_100, video_id="V2")],
        }
    )

    selected = QueryAwareOfficialFrameSelector().select(
        bands=bands,
        atoms=[_atom("a1")],
        provider=provider,
        moment_confidence={"b1": 0.3, "b2": 0.2},
        limit=4,
    )

    assert {selected[0].frame.video_id, selected[1].frame.video_id} == {"V1", "V2"}
    assert len({item.source_band_id for item in selected[:2]}) == 2


def test_filters_non_official_missing_and_duplicates_and_caps_at_100():
    frames = [_frame(index, index * 10) for index in range(120)]
    frames.extend(
        [
            _frame(10, 100),
            _frame(121, 1_210, official=False),
            _frame(122, 1_220, exists=False),
        ]
    )
    band = _band("b1", end_ms=2_000, peak_ms=600)

    selected = QueryAwareOfficialFrameSelector().select(
        bands=[band],
        atoms=[_atom("a1")],
        provider=Provider({"V1": frames}),
        moment_confidence={"b1": 0.9},
        limit=500,
    )

    pairs = [(item.frame.video_id, item.frame.frame_idx) for item in selected]
    assert len(selected) == 100
    assert len(pairs) == len(set(pairs))
    assert all(item.frame.is_official and item.frame.exists for item in selected)


def test_empty_or_missing_provider_results_are_ignored():
    provider = Provider({"V1": []})

    selected = QueryAwareOfficialFrameSelector().select(
        bands=[_band("b1")],
        atoms=[_atom("a1")],
        provider=provider,
    )

    assert selected == []
    assert provider.calls == [("V1", 0, 4_000)]


def test_repeated_hits_from_one_embedding_family_do_not_inflate_frame_score():
    single = _band(
        "single",
        hits=[_hit(entity_id="h1", atom_id="a1", timestamp_ms=2_000, rank=1, raw_score=0.9)],
    )
    repeated = _band(
        "repeated",
        hits=[
            _hit(entity_id=f"h{index}", atom_id="a1", timestamp_ms=2_000, rank=1, raw_score=0.9)
            for index in range(1, 6)
        ],
    )
    for band in (single, repeated):
        band.hits[:] = [
            hit.model_copy(update={"retriever_family": "legacy_clip_b32"})
            for hit in band.hits
        ]
    provider = Provider({"V1": [_frame(20, 2_000)]})
    selector = QueryAwareOfficialFrameSelector()

    first = selector.select(bands=[single], atoms=[_atom("a1")], provider=provider)[0]
    second = selector.select(bands=[repeated], atoms=[_atom("a1")], provider=provider)[0]

    assert first.score_components["atom_evidence"] == second.score_components["atom_evidence"]


def test_action_evidence_can_move_the_selected_frame_away_from_scene_midpoint():
    band = _band(
        "action-band",
        hits=[
            _hit(entity_id="entity", atom_id="entity", timestamp_ms=2_000, rank=1, raw_score=0.9),
            _hit(entity_id="action", atom_id="action", timestamp_ms=3_000, rank=1, raw_score=0.8),
        ],
    )
    selected = QueryAwareOfficialFrameSelector().select(
        bands=[band],
        atoms=[_atom("entity", 1.0, "ENTITY"), _atom("action", 1.0, "ACTION")],
        provider=Provider({"V1": [_frame(20, 2_000), _frame(30, 3_000)]}),
        moment_confidence={"action-band": 0.9},
        limit=2,
    )

    assert selected[0].frame.frame_idx == 30
