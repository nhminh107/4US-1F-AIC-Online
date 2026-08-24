from __future__ import annotations

import pytest

from BackEnd.app.retrieval_v2.contracts import MomentBand, QueryAtom
from BackEnd.app.retrieval_v2.query_compiler import compile_visual_clause


def test_count_and_entity_phrases_do_not_include_trailing_copula():
    atoms = compile_visual_clause(
        "three people are walking down a rainy slope with umbrellas"
    )

    assert any(atom.atom_type == "ENTITY" and atom.text == "three people" for atom in atoms)
    assert any(
        atom.atom_type == "COUNT" and atom.object == "people" and atom.count == 3
        for atom in atoms
    )
    assert all(not atom.text.endswith(" are") for atom in atoms)


def test_contracts_reject_invalid_semantic_and_temporal_states():
    with pytest.raises(ValueError, match="MUST_NOT"):
        QueryAtom(
            atom_id="A1",
            text="no red car",
            modality="visual",
            required=True,
            operator="MUST_NOT",
            role="REQUIRED",
            discriminative_weight=1.0,
        )

    with pytest.raises(ValueError, match="end_ms"):
        MomentBand(
            band_id="bad",
            video_id="V1",
            start_ms=2_000,
            end_ms=1_000,
            peak_ms=1_500,
        )

    with pytest.raises(ValueError, match="peak_ms"):
        MomentBand(
            band_id="bad-peak",
            video_id="V1",
            start_ms=1_000,
            end_ms=2_000,
            peak_ms=3_000,
        )
