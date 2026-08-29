from BackEnd.app.contracts.models import Event, StructuredQuery
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan


def test_lion_dance_plan_exposes_typed_count_relation_and_attribute_atoms():
    query = StructuredQuery(
        query_id="semantic-lion-dance",
        task="KIS",
        events=[
            Event(
                event_id="E1",
                description=(
                    "A lion dancer stands upright and spins on top of a pole, "
                    "then jumps across two adjacent poles, dives headfirst to bite "
                    "a pumpkin decorated with a yellow flower, and finally "
                    "continues jumping to the next poles."
                ),
            )
        ],
    )

    atoms = build_retrieval_plan(query).atoms

    assert len(atoms) <= 6
    assert any(atom.atom_type == "ENTITY" and "lion dancer" in atom.text.lower() for atom in atoms)
    assert any(atom.atom_type == "ACTION" and "bite" in atom.text.lower() for atom in atoms)
    assert any(atom.atom_type == "COUNT" and atom.count == 2 and "pole" in atom.text.lower() for atom in atoms)
    assert any(atom.atom_type == "RELATION" and "top of" in (atom.relation or "") for atom in atoms)
    assert any(
        atom.atom_type == "ATTRIBUTE"
        and atom.attributes.get("color") == "yellow"
        and "flower" in atom.text.lower()
        for atom in atoms
    )
    assert any(atom.group_id for atom in atoms)


def test_visible_sign_context_remains_visual_without_explicit_ocr_constraint():
    query = StructuredQuery(
        query_id="semantic-london-zoo",
        task="KIS",
        visual_queries=[
            "lions resting on wooden platforms in front of a London Zoo conservation sign"
        ],
    )

    plan = build_retrieval_plan(query)

    assert plan.atoms
    assert all(atom.modality == "visual" for atom in plan.atoms)
    assert all("ocr_search" not in atom.allowed_retrievers for atom in plan.atoms)


def test_ocr_and_asr_preserve_source_language_and_are_routed_only_when_explicit():
    query = StructuredQuery(
        query_id="semantic-source-text",
        task="KIS",
        visual_queries=["a spacecraft physically visible above a city"],
        ocr_constraints=["TÀU VŨ TRỤ"],
        asr_constraints=["đập thủy lợi"],
    )

    atoms = build_retrieval_plan(query).atoms

    assert any(atom.modality == "ocr" and atom.text == "TÀU VŨ TRỤ" for atom in atoms)
    assert any(atom.modality == "asr" and atom.text == "đập thủy lợi" for atom in atoms)
    assert all(
        atom.allowed_retrievers == ["ocr_search"]
        for atom in atoms
        if atom.modality == "ocr"
    )
    assert all(
        atom.allowed_retrievers == ["asr_search"]
        for atom in atoms
        if atom.modality == "asr"
    )


def test_object_count_is_typed_and_forwarded_as_a_hard_minimum():
    query = StructuredQuery(
        query_id="semantic-object-count",
        task="KIS",
        object_constraints=["three people"],
    )
    plan = build_retrieval_plan(query)
    atom = plan.atoms[0]

    assert atom.modality == "object"
    assert atom.atom_type == "COUNT"
    assert atom.count == 3
    assert atom.object == "people"
