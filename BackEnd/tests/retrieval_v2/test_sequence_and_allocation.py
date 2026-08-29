from BackEnd.app.retrieval_v2.allocation import allocate_submission_bands
from BackEnd.app.retrieval_v2.contracts import CoverageCell, MomentBand, QueryAtom, VideoHypothesis
from BackEnd.app.retrieval_v2.ranking import rerank_bands
from BackEnd.app.retrieval_v2.sequence import collapse_kis_sequences


def _band(video_id: str, event_id: str, start_ms: int, score: float) -> MomentBand:
    atom_id = event_id.replace("E", "A")
    return MomentBand(
        band_id=f"{video_id}-{event_id}-{start_ms}",
        video_id=video_id,
        event_id=event_id,
        start_ms=start_ms,
        end_ms=start_ms + 1_000,
        peak_ms=start_ms + 500,
        coverage={
            atom_id: CoverageCell(
                atom_id=atom_id,
                status="PASS",
                score=score,
                evidence_ids=[f"hit-{video_id}-{event_id}"],
            )
        },
        score=score,
    )


def test_kis_sequence_rejects_video_with_wrong_event_order():
    bands = [
        _band("V1", "E1", 1_000, 0.8),
        _band("V1", "E2", 5_000, 0.8),
        _band("V1", "E3", 9_000, 0.8),
        _band("V2", "E1", 8_000, 0.95),
        _band("V2", "E2", 4_000, 0.95),
        _band("V2", "E3", 12_000, 0.95),
    ]

    sequences = collapse_kis_sequences(bands, ["E1", "E2", "E3"], limit=10)

    assert sequences
    assert sequences[0].video_id == "V1"
    assert (sequences[0].start_ms, sequences[0].end_ms) == (1_000, 10_000)


def test_kis_sequence_prefers_compact_event_chain_at_equal_evidence_score():
    bands = [
        _band("V-compact", "E1", 1_000, 0.9),
        _band("V-compact", "E2", 10_000, 0.9),
        _band("V-distant", "E1", 1_000, 0.9),
        _band("V-distant", "E2", 301_000, 0.9),
    ]

    sequences = collapse_kis_sequences(bands, ["E1", "E2"], limit=10)

    assert sequences[0].video_id == "V-compact"
    assert sequences[0].score_breakdown["sequence_compactness"] > sequences[1].score_breakdown["sequence_compactness"]

    atoms = [
        QueryAtom(atom_id="A1", event_id="E1", text="first event", modality="visual", discriminative_weight=1.0),
        QueryAtom(atom_id="A2", event_id="E2", text="second event", modality="visual", discriminative_weight=1.0),
    ]
    reranked = rerank_bands(sequences, atoms, limit=10)
    assert reranked[0].video_id == "V-compact"


def test_allocator_concentrates_when_video_confidence_is_high():
    bands = [
        _band("V1", "E1", 1_000 + index * 2_000, 0.9 - index * 0.01)
        for index in range(5)
    ] + [_band("V2", "E1", 2_000, 0.7)]
    coverage = {"A1": CoverageCell(atom_id="A1", status="PASS", score=0.9)}
    hypotheses = [
        VideoHypothesis(
            video_id="V1",
            video_confidence=0.92,
            moment_confidence=0.82,
            coverage=coverage,
        ),
        VideoHypothesis(
            video_id="V2",
            video_confidence=0.4,
            moment_confidence=0.4,
            coverage=coverage,
        ),
    ]

    allocated = allocate_submission_bands(bands, hypotheses, limit=4)

    assert [band.video_id for band in allocated] == ["V1", "V1", "V1", "V1"]


def test_allocator_diversifies_when_video_confidence_is_low():
    bands = [
        _band("V1", "E1", 1_000, 0.9),
        _band("V1", "E1", 4_000, 0.85),
        _band("V2", "E1", 2_000, 0.8),
        _band("V3", "E1", 3_000, 0.75),
    ]
    coverage = {"A1": CoverageCell(atom_id="A1", status="PASS", score=0.8)}
    hypotheses = [
        VideoHypothesis(
            video_id=video_id,
            video_confidence=confidence,
            moment_confidence=0.5,
            coverage=coverage,
        )
        for video_id, confidence in (("V1", 0.6), ("V2", 0.55), ("V3", 0.5))
    ]

    allocated = allocate_submission_bands(bands, hypotheses, limit=3)

    assert len({band.video_id for band in allocated}) == 3
