from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.constraints import HardConstraintEngine
from BackEnd.app.retrieval_v2.moment_bands import build_moment_bands
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan
from BackEnd.app.contracts.models import StructuredQuery


def _hit(
    entity_id: str,
    *,
    atom_id: str,
    start_ms: int,
    end_ms: int,
    rank: int = 1,
    entity_type: str = "frame",
) -> SearchHit:
    return SearchHit(
        source=f"{entity_type}_embedding",
        entity_type=entity_type,
        entity_id=entity_id,
        video_id="V1",
        start_ms=start_ms,
        end_ms=end_ms,
        rank=rank,
        raw_score=0.8,
        atom_id=atom_id,
        prompt_role="global",
        retriever_family="legacy_clip_b32",
    )


def test_search_hit_keeps_atom_and_prompt_provenance():
    hit = _hit("F1", atom_id="A1", start_ms=1_000, end_ms=1_000)

    assert hit.atom_id == "A1"
    assert hit.prompt_role == "global"
    assert hit.retriever_family == "legacy_clip_b32"


def test_moment_band_preserves_extent_and_marks_missing_atom_unknown():
    hits = [
        _hit("F1", atom_id="A1", start_ms=1_000, end_ms=1_000),
        _hit(
            "C1",
            atom_id="A2",
            start_ms=1_500,
            end_ms=4_000,
            entity_type="clip",
        ),
    ]

    bands = build_moment_bands(
        hits,
        required_atom_ids=["A1", "A2", "A3"],
        merge_gap_ms=1_000,
    )

    assert len(bands) == 1
    assert (bands[0].start_ms, bands[0].end_ms) == (1_000, 4_000)
    assert bands[0].coverage["A1"].retrieval_status == "RETRIEVED"
    assert bands[0].coverage["A2"].retrieval_status == "RETRIEVED"
    assert bands[0].coverage["A1"].status == "UNKNOWN"
    assert bands[0].coverage["A2"].status == "UNKNOWN"
    assert bands[0].coverage["A3"].retrieval_status == "MISSING"
    assert bands[0].coverage["A3"].status == "UNKNOWN"


def test_hard_constraint_missing_evidence_is_unknown_not_fail():
    engine = HardConstraintEngine()

    decision = engine.evaluate_required_atom(
        atom_id="A1",
        evidence_ids=[],
        contradiction_evidence_ids=[],
        scope="MOMENT_BAND",
    )

    assert decision.status == "UNKNOWN"
    assert decision.reason_code == "MISSING_REQUIRED_ATOM"


def test_hard_constraint_positive_contradiction_fails_only_requested_scope():
    engine = HardConstraintEngine()

    decision = engine.evaluate_required_atom(
        atom_id="A1",
        evidence_ids=[],
        contradiction_evidence_ids=["review-7"],
        scope="MOMENT_BAND",
    )

    assert decision.status == "FAIL"
    assert decision.scope == "MOMENT_BAND"
    assert decision.evidence_ids == ["review-7"]


def test_visual_atom_rejects_asr_retriever_before_io():
    query = StructuredQuery(
        query_id="q-firewall",
        task="KIS",
        visual_queries=["a spacecraft visible over a city"],
    )
    atom = build_retrieval_plan(query).atoms[0]

    decision = HardConstraintEngine().validate_retriever(atom, "asr_search")

    assert decision.status == "FAIL"
    assert decision.scope == "PLAN"
    assert decision.reason_code == "FORBIDDEN_MODALITY"


def test_same_embedding_family_is_capped_while_independent_family_adds_support():
    same_family = build_moment_bands(
        [
            _hit("F1", atom_id="A1", start_ms=1_000, end_ms=1_000),
            _hit("F2", atom_id="A1", start_ms=1_000, end_ms=1_000),
            _hit("F3", atom_id="A1", start_ms=1_000, end_ms=1_000),
        ],
        required_atom_ids=["A1"],
    )[0]
    independent_hit = _hit("O1", atom_id="A1", start_ms=1_000, end_ms=1_000)
    independent_hit = independent_hit.model_copy(
        update={"retriever_family": "object_detector_v3"}
    )
    independent = build_moment_bands(
        [same_family.hits[0], independent_hit],
        required_atom_ids=["A1"],
    )[0]

    assert same_family.coverage["A1"].score == 1.0 / 61.0
    assert independent.coverage["A1"].score > same_family.coverage["A1"].score


def test_retrieved_but_unverified_required_atom_keeps_candidate_gate_unknown():
    band = build_moment_bands(
        [_hit("F1", atom_id="A1", start_ms=1_000, end_ms=1_000)],
        required_atom_ids=["A1"],
    )[0]
    atom = build_retrieval_plan(
        StructuredQuery(query_id="q-gate-unknown", task="KIS", visual_queries=["a car"])
    ).atoms[0]
    atom = atom.model_copy(update={"atom_id": "A1"})

    decisions = HardConstraintEngine().evaluate_band(band, [atom])

    assert decisions[0].status == "UNKNOWN"
    assert decisions[0].reason_code == "REQUIRED_ATOM_RETRIEVED_UNVERIFIED"
