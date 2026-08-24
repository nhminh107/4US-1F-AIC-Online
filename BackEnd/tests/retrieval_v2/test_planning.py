from BackEnd.app.contracts.models import Event, StructuredQuery
from BackEnd.app.retrieval_v2.planning import (
    build_retrieval_plan,
    estimate_discriminative_weight,
)


def test_visual_atom_gets_role_based_prompt_ensemble():
    query = StructuredQuery(
        query_id="q-prompts",
        task="KIS",
        visual_queries=[
            "lion dance performers jumping between poles and biting a pumpkin with a yellow flower"
        ],
    )

    plan = build_retrieval_plan(query)

    assert 1 <= len(plan.atoms) <= 6
    assert all(2 <= len(atom.prompt_variants) <= 4 for atom in plan.atoms)
    assert any(
        "rare_detail" in {prompt.role for prompt in atom.prompt_variants}
        for atom in plan.atoms
    )
    assert all(prompt.text.strip() for atom in plan.atoms for prompt in atom.prompt_variants)
    assert all(0.0 < atom.discriminative_weight <= 2.0 for atom in plan.atoms)


def test_rare_detail_scores_above_common_scene_description():
    common = estimate_discriminative_weight("people walking outside in the rain")
    rare = estimate_discriminative_weight(
        "lion dancer biting a pumpkin with a yellow flower while standing on poles"
    )

    assert rare > common


def test_kis_with_ordered_events_uses_sequence_profile():
    query = StructuredQuery(
        query_id="q-sequence",
        task="KIS",
        events=[
            Event(event_id="E1", description="a map appears"),
            Event(event_id="E2", description="an aerial view of a dam"),
            Event(event_id="E3", description="a close view of the dam in rain"),
        ],
    )

    plan = build_retrieval_plan(query)

    assert plan.execution_profile == "KIS_SEQUENCE"
    assert list(dict.fromkeys(atom.event_id for atom in plan.atoms)) == ["E1", "E2", "E3"]


def test_temporal_clauses_create_synthetic_kis_events_without_llm_events():
    query = StructuredQuery(
        query_id="q-synthetic-sequence",
        task="KIS",
        visual_queries=[
            "a map appears, then an aerial view of a dam, finally a close view in rain"
        ],
    )

    plan = build_retrieval_plan(query)

    assert plan.execution_profile == "KIS_SEQUENCE"
    assert list(dict.fromkeys(atom.event_id for atom in plan.atoms)) == ["E1", "E2", "E3"]


def test_visual_only_query_does_not_create_ocr_or_asr_atoms():
    query = StructuredQuery(
        query_id="q-visual-only",
        task="KIS",
        visual_queries=["a spacecraft flying above a city"],
    )

    plan = build_retrieval_plan(query)

    assert {atom.modality for atom in plan.atoms} == {"visual"}


def test_vqa_question_is_used_as_locator_when_no_other_atom_exists():
    query = StructuredQuery(
        query_id="q-vqa-fallback",
        task="VQA",
        question="What color is the shirt worn by the person near the bus?",
    )

    plan = build_retrieval_plan(query)

    assert len(plan.atoms) == 1
    assert plan.atoms[0].modality == "visual"
    assert "shirt" in plan.atoms[0].text
    assert plan.atoms[0].scope == "ANSWER_EVIDENCE"
    assert plan.answer_spec is not None
    assert plan.answer_spec.answer_type == "COLOR"


def test_trake_preserves_top_level_visual_query_as_video_anchor():
    query = StructuredQuery(
        query_id="q-trake-anchor",
        task="TRAKE",
        visual_queries=["a white lion dance head with a red nose beside a white flag"],
        events=[
            Event(event_id="E1", description="two golden dragons spin"),
            Event(event_id="E2", description="a lion lands on poles after spinning"),
            Event(event_id="E3", description="a mallet touches a bronze gong"),
        ],
    )

    plan = build_retrieval_plan(query)

    anchors = [atom for atom in plan.atoms if atom.scope == "VIDEO_ANCHOR"]
    assert anchors
    assert all(atom.event_id is None for atom in anchors)
    assert {atom.event_id for atom in plan.atoms if atom.scope == "EVENT"} == {
        "E1",
        "E2",
        "E3",
    }


def test_many_intent_visual_queries_share_one_bounded_anchor_budget():
    query = StructuredQuery(
        query_id="bounded-live-rain",
        task="KIS",
        visual_queries=[
            "three people walking down a slope in the rain",
            "two people holding umbrellas",
            "a raincoat with a bear image on the back",
            "many people walking towards a house",
            "a dirt road beside a pond",
        ],
        events=[
            Event(event_id="E1", description="three people descend a rainy slope with two umbrellas"),
            Event(event_id="E2", description="many people walk to a house on a dirt road beside a pond"),
        ],
    )

    plan = build_retrieval_plan(query)
    anchor_atoms = [atom for atom in plan.atoms if atom.scope == "VIDEO_ANCHOR"]
    event_atoms = [atom for atom in plan.atoms if atom.scope == "EVENT"]

    assert 1 <= len(anchor_atoms) <= 4
    assert len(event_atoms) <= 12
    assert len(plan.atoms) <= 16


def test_earthquake_vqa_has_mixed_number_answer_spec():
    query = StructuredQuery(
        query_id="q-map-count",
        task="VQA",
        question=(
            "Không tính bảng chú giải, có bao nhiêu vị trí ghi nhận động đất "
            "cấp độ 4?"
        ),
        visual_queries=["earthquake distribution map with a color legend"],
        ocr_constraints=["4"],
    )

    plan = build_retrieval_plan(query)

    assert plan.answer_spec is not None
    assert plan.answer_spec.answer_type == "NUMBER"
    assert plan.answer_spec.answer_source == "MIXED"
    assert all(atom.scope == "ANSWER_EVIDENCE" for atom in plan.atoms)
