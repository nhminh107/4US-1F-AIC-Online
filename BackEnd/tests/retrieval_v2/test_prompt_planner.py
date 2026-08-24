from __future__ import annotations

import json
import math
import re

import pytest

from BackEnd.app.retrieval_v2.corpus_stats import (
    CorpusDocument,
    CorpusStats,
    SelectivityMetrics,
)
from BackEnd.app.retrieval_v2.prompt_planner import (
    CorpusAwarePromptPlanner,
    SemanticAtom,
)
from scripts.build_retrieval_corpus_stats import main as build_stats_main


def _stats() -> CorpusStats:
    return CorpusStats.from_documents(
        [
            CorpusDocument(
                document_id="d1",
                video_id="v1",
                text="lion dance performers balance on tall poles",
            ),
            CorpusDocument(
                document_id="d2",
                video_id="v2",
                text="lion dance performers jump between poles",
            ),
            CorpusDocument(
                document_id="d3",
                video_id="v3",
                text="yellow flower attached to an orange pumpkin",
            ),
            CorpusDocument(
                document_id="d4",
                video_id="v4",
                text="people walk outside near a building",
            ),
        ]
    )


def test_corpus_stats_round_trip_versioned_json_and_idf(tmp_path):
    stats = _stats()

    assert stats.schema_version == "retrieval-corpus-stats-v1"
    assert stats.document_count == 4
    assert stats.video_count == 4
    assert stats.document_frequency["lion"] == 2
    assert stats.idf("pumpkin") > stats.idf("lion")
    assert stats.idf("unseen-token") == pytest.approx(math.log(5.0) + 1.0)

    path = tmp_path / "corpus_stats.json"
    stats.save(path)
    loaded = CorpusStats.load(path)

    assert loaded == stats
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "retrieval-corpus-stats-v1"

    payload["schema_version"] = "future-v99"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        CorpusStats.load(path)


def test_online_selectivity_reports_entropy_and_unique_video_spread():
    concentrated = CorpusStats.measure_selectivity(
        video_ids=["v1", "v1", "v1", "v1"],
        scores=[1.0, 0.9, 0.8, 0.7],
    )
    broad = CorpusStats.measure_selectivity(
        video_ids=["v1", "v2", "v3", "v4"],
        scores=[1.0, 1.0, 1.0, 1.0],
    )

    assert concentrated == SelectivityMetrics(
        hit_count=4,
        unique_video_count=1,
        normalized_entropy=0.0,
        unique_video_spread=0.25,
    )
    assert broad.normalized_entropy == pytest.approx(1.0)
    assert broad.unique_video_spread == pytest.approx(1.0)
    assert concentrated.selectivity_score > broad.selectivity_score


def test_prompt_roles_are_semantically_distinct_and_corpus_aware():
    planner = CorpusAwarePromptPlanner(_stats())
    atom = SemanticAtom(
        text_en=(
            "A lion dance costume leaps between tall poles and bites an orange "
            "pumpkin decorated with a yellow flower"
        ),
        subject="two performers inside a lion dance costume",
        action="leaps between adjacent poles and bites the pumpkin",
        context="on top of tall performance poles",
        rare_details=(
            "orange pumpkin with a yellow flower",
            "lion dance costume",
        ),
        positive_discriminators=(
            "the lion head reaches downward toward the decorated pumpkin",
        ),
    )

    variants = planner.plan(atom)
    by_role = {variant.role: variant for variant in variants}

    assert set(by_role) == {"global", "rare_detail", "action", "context", "contrast"}
    assert len({variant.text.casefold() for variant in variants}) == 5
    assert "two performers" in by_role["global"].text.casefold()
    assert "yellow flower" in by_role["rare_detail"].text.casefold()
    assert "leaps" in by_role["action"].text.casefold()
    assert "tall performance poles" in by_role["context"].text.casefold()
    assert "reaches downward" in by_role["contrast"].text.casefold()
    assert by_role["rare_detail"].weight > by_role["context"].weight

    forbidden_negation = re.compile(r"\b(?:not|no|without)\b", flags=re.IGNORECASE)
    assert forbidden_negation.search(by_role["contrast"].text) is None
    assert all(variant.text.isascii() for variant in variants)
    assert all(not variant.text.casefold().endswith(atom.text_en.casefold()) for variant in variants)


def test_prompt_renderer_does_not_emit_empty_or_repeated_action_phrases():
    planner = CorpusAwarePromptPlanner()
    entity_variants = planner.plan(SemanticAtom(
        text_en="three people",
        subject="three people",
    ))
    action_variants = planner.plan(SemanticAtom(
        text_en="three people walking down a slope",
        subject="three people",
        action="three people walking down a slope",
    ))

    entity_global = next(item.text for item in entity_variants if item.role == "global")
    action_global = next(item.text for item in action_variants if item.role == "global")
    assert "participates in ," not in entity_global
    assert "three people three people" not in action_global.casefold()


def test_online_selectivity_can_reweight_prompt_roles():
    planner = CorpusAwarePromptPlanner(_stats())
    atom = SemanticAtom(
        text_en="A chef places coriander on a bowl of chicken noodle soup",
        subject="a chef",
        action="places coriander on a bowl of noodle soup",
        context="at a food preparation counter",
        rare_details=("a dipping bowl containing two chili pieces",),
    )
    selective = SelectivityMetrics(
        hit_count=20,
        unique_video_count=2,
        normalized_entropy=0.15,
        unique_video_spread=0.1,
    )
    broad = SelectivityMetrics(
        hit_count=20,
        unique_video_count=18,
        normalized_entropy=0.95,
        unique_video_spread=0.9,
    )

    variants = planner.plan(
        atom,
        online_selectivity={"rare_detail": selective, "context": broad},
    )
    by_role = {variant.role: variant for variant in variants}

    assert by_role["rare_detail"].weight > by_role["context"].weight


def test_prompt_planner_has_deterministic_english_fallback():
    planner = CorpusAwarePromptPlanner()
    atom = SemanticAtom(text_en="A red bus crosses a stone bridge", language="en")

    first = planner.plan(atom)
    second = planner.plan(atom)

    assert first == second
    assert len(first) == 5
    assert len({variant.text for variant in first}) == 5
    assert all(variant.text.isascii() for variant in first)
    contrast = next(variant.text for variant in first if variant.role == "contrast")
    assert re.search(r"\b(?:not|no|without)\b", contrast, re.IGNORECASE) is None

    with pytest.raises(ValueError, match="English"):
        SemanticAtom(text_en="Một chiếc xe buýt màu đỏ", language="vi")


def test_contrast_drops_negated_input_clauses_instead_of_inverting_them():
    planner = CorpusAwarePromptPlanner()
    atom = SemanticAtom(
        text_en="A cyclist without a helmet waits near a bus",
        subject="a cyclist without a helmet",
        rare_details=("no visible race number",),
        positive_discriminators=("not a motorcycle",),
    )

    contrast = next(
        variant.text for variant in planner.plan(atom) if variant.role == "contrast"
    )

    assert re.search(r"\b(?:not|no|without)\b", contrast, re.IGNORECASE) is None
    assert "helmet" not in contrast.casefold()
    assert "motorcycle" not in contrast.casefold()


def test_build_corpus_stats_script_reads_jsonl(tmp_path):
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "stats.json"
    rows = [
        {"id": "f1", "video_id": "v1", "text": "red bus on a bridge"},
        {"id": "f2", "video_id": "v2", "text": "blue bus near a station"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert build_stats_main([str(source), str(output)]) == 0
    stats = CorpusStats.load(output)

    assert stats.document_count == 2
    assert stats.video_count == 2
    assert stats.document_frequency["bus"] == 2
