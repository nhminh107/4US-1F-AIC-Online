from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import QueryAtom
from BackEnd.app.retrieval_v2.reservoir import select_candidate_reservoir


def _atom(atom_id: str, *, event_id: str, weight: float) -> QueryAtom:
    return QueryAtom(
        atom_id=atom_id,
        event_id=event_id,
        text=atom_id,
        modality="visual",
        discriminative_weight=weight,
        prompt_variants=[],
    )


def _hit(
    entity_id: str,
    *,
    video_id: str,
    start_ms: int,
    atom_id: str,
    event_id: str,
    family: str,
    rank: int,
) -> SearchHit:
    return SearchHit(
        source=family,
        entity_type="frame",
        entity_id=entity_id,
        video_id=video_id,
        start_ms=start_ms,
        end_ms=start_ms,
        atom_id=atom_id,
        event_id=event_id,
        retriever_family=family,
        rank=rank,
        raw_score=0.9,
    )


def test_reservoir_protects_each_atom_event_and_retriever_lane():
    atoms = [
        _atom("A-common", event_id="E1", weight=0.4),
        _atom("A-rare", event_id="E2", weight=2.0),
    ]
    hits = [
        _hit(
            f"common-{index}",
            video_id=f"V{index}",
            start_ms=index * 4_000,
            atom_id="A-common",
            event_id="E1",
            family="frame",
            rank=index + 1,
        )
        for index in range(20)
    ]
    hits.extend(
        [
            _hit(
                "rare-clip",
                video_id="V-correct",
                start_ms=80_000,
                atom_id="A-rare",
                event_id="E2",
                family="clip",
                rank=90,
            ),
            _hit(
                "rare-shot",
                video_id="V-correct",
                start_ms=80_000,
                atom_id="A-rare",
                event_id="E2",
                family="shot",
                rank=95,
            ),
        ]
    )

    selected = select_candidate_reservoir(hits, atoms, limit=4)

    assert len({(hit.video_id, hit.start_ms // 2_000) for hit in selected}) == 4
    assert any(hit.atom_id == "A-rare" for hit in selected)
    assert {hit.retriever_family for hit in selected} >= {"clip", "shot"}


def test_reservoir_prefers_multi_atom_consensus_over_one_generic_neighbor():
    atoms = [
        _atom("A1", event_id="E1", weight=1.0),
        _atom("A2", event_id="E1", weight=1.5),
    ]
    hits = [
        _hit(
            "generic",
            video_id="V-generic",
            start_ms=0,
            atom_id="A1",
            event_id="E1",
            family="frame",
            rank=1,
        ),
        _hit(
            "consensus-a1",
            video_id="V-consensus",
            start_ms=10_000,
            atom_id="A1",
            event_id="E1",
            family="frame",
            rank=5,
        ),
        _hit(
            "consensus-a2",
            video_id="V-consensus",
            start_ms=10_000,
            atom_id="A2",
            event_id="E1",
            family="clip",
            rank=5,
        ),
    ]

    selected = select_candidate_reservoir(hits, atoms, limit=1)

    assert {hit.video_id for hit in selected} == {"V-consensus"}

