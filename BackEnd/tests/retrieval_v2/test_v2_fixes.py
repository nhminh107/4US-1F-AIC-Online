"""Tests for Phase A-F fixes: scoring, contracts, constraints, gateway, submission."""
from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path

import pytest

from BackEnd.app.contracts.models import SearchHit, StructuredQuery, Event
from BackEnd.app.retrieval_v2.constraints import HardConstraintEngine
from BackEnd.app.retrieval_v2.contracts import (
    AtomLink,
    CoverageCell,
    MomentBand,
    QueryAtom,
    ScoringProfile,
    VideoHypothesis,
)
from BackEnd.app.retrieval_v2.moment_bands import build_moment_bands
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan
from BackEnd.app.retrieval_v2.ranking import (
    build_video_hypotheses,
    diagnose_hypotheses,
    rerank_bands,
)
from BackEnd.app.retrieval_v2.submission import SubmissionBuilder, SubmissionRow
from BackEnd.app.retrieval_v2.round_manifest import RoundManifest, DEFAULT_MANIFEST
from BackEnd.app.retrieval_v2.answer_spec import AnswerSpec, AnswerGenerator, GroundedAnswer


# ---------- helpers ----------


def test_diagnosis_checks_multiple_plausible_videos_and_absent_coverage_cells():
    atoms = [
        QueryAtom(atom_id="A1", text="rain", modality="visual", discriminative_weight=1.0),
        QueryAtom(atom_id="A2", text="house by pond", modality="visual", discriminative_weight=1.8),
    ]
    hypotheses = [
        VideoHypothesis(
            video_id="V1",
            video_confidence=0.8,
            moment_confidence=0.4,
            coverage={"A1": CoverageCell(atom_id="A1", retrieval_status="RETRIEVED", score=0.2)},
            band_ids=["B1"],
            lane_sources=["moment"],
        ),
        VideoHypothesis(
            video_id="V2",
            video_confidence=0.7,
            moment_confidence=0.3,
            coverage={"A2": CoverageCell(atom_id="A2", retrieval_status="RETRIEVED", score=0.2)},
            band_ids=["B2"],
            lane_sources=["moment"],
        ),
    ]

    diagnoses = diagnose_hypotheses(hypotheses, atoms)

    assert any(item.video_id == "V1" and item.atom_id == "A2" for item in diagnoses)
    assert any(item.video_id == "V2" and item.atom_id == "A1" for item in diagnoses)
    assert {item.video_id for item in diagnoses if item.reason == "LOW_MOMENT_CONFIDENCE"} == {"V1", "V2"}

def _hit(
    entity_id: str,
    *,
    atom_id: str,
    start_ms: int,
    end_ms: int,
    rank: int = 1,
    entity_type: str = "frame",
    retriever_family: str = "legacy_clip_b32",
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
        retriever_family=retriever_family,
    )


def _atom(atom_id: str, *, role: str = "REQUIRED", modality: str = "visual") -> QueryAtom:
    return QueryAtom(
        atom_id=atom_id,
        text="test",
        modality=modality,
        required=role == "REQUIRED",
        role=role,
        discriminative_weight=1.0,
    )


def _band_with_coverage(
    band_id: str,
    video_id: str,
    atom_scores: dict[str, tuple[str, float]],
    score: float = 0.5,
) -> MomentBand:
    coverage = {
        aid: CoverageCell(atom_id=aid, status=status, score=sc)
        for aid, (status, sc) in atom_scores.items()
    }
    return MomentBand(
        band_id=band_id,
        video_id=video_id,
        start_ms=1000,
        end_ms=5000,
        peak_ms=3000,
        coverage=coverage,
        score=score,
    )


# ==================== Phase A: Scoring ====================

class TestGeometricMeanScoring:
    """W1: rerank_bands should use geometric mean for required atoms."""

    def test_full_coverage_beats_partial_coverage(self):
        atoms = [_atom("A1"), _atom("A2")]
        # Band 1: both atoms covered
        full = _band_with_coverage("b1", "V1", {
            "A1": ("PASS", 0.5),
            "A2": ("PASS", 0.5),
        })
        # Band 2: one atom very strong but other missing
        partial = _band_with_coverage("b2", "V2", {
            "A1": ("PASS", 0.9),
            "A2": ("UNKNOWN", 0.0),
        })
        result = rerank_bands([full, partial], atoms, limit=10)
        # Full coverage should rank higher than partial with one strong atom
        assert result[0].band_id == "b1"

    def test_missing_penalty_applied(self):
        atoms = [_atom("A1"), _atom("A2")]
        band = _band_with_coverage("b1", "V1", {
            "A1": ("PASS", 0.5),
            "A2": ("UNKNOWN", 0.0),
        })
        result = rerank_bands([band], atoms, limit=10)
        # Score should be reduced by missing penalty
        profile = ScoringProfile()
        assert result[0].score < profile.W_required * 0.5

    def test_negative_penalty_applied(self):
        atoms = [_atom("A1")]
        band = _band_with_coverage("b1", "V1", {
            "A1": ("FAIL", 0.0),
        })
        result = rerank_bands([band], atoms, limit=10)
        assert result[0].score == 0.0  # Clamped to 0

    def test_custom_scoring_profile(self):
        profile = ScoringProfile(W_required=1.0, W_supporting=0.0, missing_penalty_per_atom=0.0)
        atoms = [_atom("A1")]
        band = _band_with_coverage("b1", "V1", {"A1": ("PASS", 0.5)})
        result = rerank_bands([band], atoms, limit=10, profile=profile)
        # With custom profile, score should be deterministic
        assert result[0].score > 0.0


class TestFamilyAwareEvidence:
    """W4: frame/clip/shot from same family should not triple-count."""

    def test_coverage_cell_carries_retriever_families(self):
        hits = [
            _hit("F1", atom_id="A1", start_ms=1000, end_ms=1000, retriever_family="legacy_clip_b32"),
            _hit("C1", atom_id="A1", start_ms=1000, end_ms=3000,
                 entity_type="clip", retriever_family="legacy_clip_b32"),
        ]
        bands = build_moment_bands(hits, required_atom_ids=["A1"])
        cell = bands[0].coverage["A1"]
        # Both from same family — should have 1 distinct family
        assert len(set(cell.retriever_families)) == 1

    def test_multi_family_gets_consensus_bonus(self):
        hits = [
            _hit("F1", atom_id="A1", start_ms=1000, end_ms=1000, retriever_family="legacy_clip_b32"),
            _hit("O1", atom_id="A1", start_ms=1000, end_ms=1000,
                 entity_type="frame", retriever_family="object_detector_v3"),
        ]
        bands = build_moment_bands(hits, required_atom_ids=["A1"])
        cell = bands[0].coverage["A1"]
        assert len(set(cell.retriever_families)) == 2


# ==================== Phase A: Shot outlier ====================

class TestShotOutlierClamp:
    """Long shots remain broad until child evidence can refine them."""

    def test_long_shot_is_not_blindly_trimmed_around_its_midpoint(self):
        # Shot of 1171 seconds = 1,171,000 ms
        hits = [
            _hit("S1", atom_id="A1", start_ms=0, end_ms=1_171_000,
                 entity_type="shot", retriever_family="legacy_clip_b32"),
        ]
        bands = build_moment_bands(hits, required_atom_ids=["A1"])
        assert len(bands) == 1
        duration = bands[0].end_ms - bands[0].start_ms
        assert duration == 1_171_000

    def test_long_shot_is_replaced_by_overlapping_child_clip(self):
        hits = [
            _hit("S1", atom_id="A1", start_ms=0, end_ms=1_171_000,
                 entity_type="shot", retriever_family="legacy_clip_b32"),
            _hit("C1", atom_id="A1", start_ms=50_000, end_ms=55_000,
                 entity_type="clip", retriever_family="legacy_clip_b32"),
        ]
        bands = build_moment_bands(hits, required_atom_ids=["A1"])
        assert len(bands) == 1
        assert (bands[0].start_ms, bands[0].end_ms) == (50_000, 55_000)


# ==================== Phase B: Contracts ====================

class TestEnrichedQueryAtom:
    """W5: QueryAtom should carry operator, role, granularity, allowed/forbidden retrievers."""

    def test_visual_atom_has_enriched_fields(self):
        query = StructuredQuery(
            query_id="q-enriched",
            task="KIS",
            visual_queries=["a spacecraft flying above a city"],
        )
        plan = build_retrieval_plan(query)
        atom = plan.atoms[0]
        assert atom.operator == "MUST"
        assert atom.role == "REQUIRED"
        assert atom.granularity == "MOMENT"
        assert "frame_search" in atom.allowed_retrievers
        assert "ocr_search" in atom.forbidden_retrievers

    def test_ocr_atom_has_correct_routing(self):
        query = StructuredQuery(
            query_id="q-ocr",
            task="KIS",
            visual_queries=["a scene"],
            ocr_constraints=["London Zoo"],
        )
        plan = build_retrieval_plan(query)
        ocr_atom = next(a for a in plan.atoms if a.modality == "ocr")
        assert ocr_atom.allowed_retrievers == ["ocr_search"]
        assert "frame_search" in ocr_atom.forbidden_retrievers

    def test_ocr_atom_linked_to_visual_atom(self):
        """W6: OCR atoms should be linked to visual atoms via same_moment."""
        query = StructuredQuery(
            query_id="q-linked",
            task="KIS",
            visual_queries=["lions in an enclosure"],
            ocr_constraints=["London Zoo"],
        )
        plan = build_retrieval_plan(query)
        ocr_atom = next(a for a in plan.atoms if a.modality == "ocr")
        assert any(link.relation == "same_moment" for link in ocr_atom.links)

    def test_scoring_profile_roundtrips(self):
        profile = ScoringProfile(W_required=0.8, missing_penalty_per_atom=0.3)
        assert profile.W_required == 0.8
        assert profile.missing_penalty_per_atom == 0.3
        assert profile.profile_id == "default_v1"

    def test_atom_link_model(self):
        link = AtomLink(target_atom_id="A1", relation="same_moment")
        assert link.target_atom_id == "A1"
        assert link.relation == "same_moment"


# ==================== Phase C: Constraints ====================

class TestConstraintEngine4Gate:
    """W8: HardConstraintEngine should have 4 gates."""

    def test_plan_gate_respects_atom_forbidden_list(self):
        atom = QueryAtom(
            atom_id="A1",
            text="test",
            modality="visual",
            discriminative_weight=1.0,
            forbidden_retrievers=["shot_search"],
        )
        engine = HardConstraintEngine()
        decision = engine.validate_retriever(atom, "shot_search")
        assert decision.status == "FAIL"

    def test_plan_gate_respects_atom_allowed_list(self):
        atom = QueryAtom(
            atom_id="A1",
            text="test",
            modality="visual",
            discriminative_weight=1.0,
            allowed_retrievers=["frame_search"],
        )
        engine = HardConstraintEngine()
        # clip_search not in allowed list
        decision = engine.validate_retriever(atom, "clip_search")
        assert decision.status == "FAIL"
        # frame_search in allowed list
        decision = engine.validate_retriever(atom, "frame_search")
        assert decision.status == "PASS"

    def test_task_gate_temporal_order_violation(self):
        engine = HardConstraintEngine()
        bands = [
            MomentBand(band_id="b1", video_id="V1", event_id="E1",
                       start_ms=10000, end_ms=12000, peak_ms=11000, score=0.5),
            MomentBand(band_id="b2", video_id="V1", event_id="E2",
                       start_ms=5000, end_ms=7000, peak_ms=6000, score=0.5),
        ]
        decision = engine.validate_temporal_order(bands, ["E1", "E2"])
        assert decision.status == "FAIL"
        assert decision.reason_code == "TEMPORAL_ORDER_VIOLATION"

    def test_task_gate_temporal_order_valid(self):
        engine = HardConstraintEngine()
        bands = [
            MomentBand(band_id="b1", video_id="V1", event_id="E1",
                       start_ms=5000, end_ms=7000, peak_ms=6000, score=0.5),
            MomentBand(band_id="b2", video_id="V1", event_id="E2",
                       start_ms=10000, end_ms=12000, peak_ms=11000, score=0.5),
        ]
        decision = engine.validate_temporal_order(bands, ["E1", "E2"])
        assert decision.status == "PASS"

    def test_task_gate_trake_arity_satisfied(self):
        engine = HardConstraintEngine()
        bands = [
            MomentBand(band_id="b1", video_id="V1", event_id="E1",
                       start_ms=1000, end_ms=2000, peak_ms=1500, score=0.5),
            MomentBand(band_id="b2", video_id="V1", event_id="E2",
                       start_ms=3000, end_ms=4000, peak_ms=3500, score=0.5),
        ]
        decision = engine.validate_trake_arity(bands, 2, "V1")
        assert decision.status == "PASS"

    def test_task_gate_trake_arity_missing(self):
        engine = HardConstraintEngine()
        bands = [
            MomentBand(band_id="b1", video_id="V1", event_id="E1",
                       start_ms=1000, end_ms=2000, peak_ms=1500, score=0.5),
        ]
        decision = engine.validate_trake_arity(bands, 3, "V1")
        assert decision.status == "UNKNOWN"
        assert decision.reason_code == "TRAKE_EVENT_ARITY_MISMATCH"

    def test_submission_gate_duplicate_row(self):
        engine = HardConstraintEngine()
        seen = {("V1", 100)}
        decision = engine.validate_submission_row(
            video_id="V1", frame_idx=100,
            is_official_frame=True, seen_pairs=seen,
        )
        assert decision.status == "FAIL"
        assert decision.reason_code == "DUPLICATE_SUBMISSION_ROW"

    def test_submission_gate_non_official_frame(self):
        engine = HardConstraintEngine()
        decision = engine.validate_submission_row(
            video_id="V1", frame_idx=100,
            is_official_frame=False,
        )
        assert decision.status == "FAIL"
        assert decision.reason_code == "INVALID_OFFICIAL_FRAME"


# ==================== Phase E: QA Answer ====================

class TestAnswerSpec:
    """W11: QA answer pipeline."""

    def test_answer_spec_model(self):
        spec = AnswerSpec(answer_type="NUMBER", answer_source="VISUAL")
        assert spec.max_length == 100
        assert spec.normalization == "short_vietnamese"

    def test_answer_generator_ocr(self):
        gen = AnswerGenerator()
        spec = AnswerSpec(answer_source="OCR")
        result = gen.generate("What text?", spec, ocr_texts=["London Zoo", "Exit"])
        assert result is not None
        assert result.answer_source == "OCR"
        # Shortest first
        assert result.answer_text == "Exit"

    def test_answer_generator_no_evidence(self):
        gen = AnswerGenerator()
        spec = AnswerSpec(answer_source="VISUAL")
        result = gen.generate("What color?", spec)
        assert result is None  # MVP: visual needs VLM


# ==================== Phase F: Submission ====================

class TestSubmissionBuilder:
    """W13: Submission builder."""

    def test_kis_dedup_and_limit(self):
        builder = SubmissionBuilder(max_rows=3)
        rows = builder.build_kis([
            ("V1", 100), ("V1", 100), ("V2", 200), ("V3", 300), ("V4", 400),
        ])
        assert len(rows) == 3
        # Deduped
        assert rows[0] == SubmissionRow(video_id="V1", frame_idx=100)

    def test_vqa_rows(self):
        builder = SubmissionBuilder()
        rows = builder.build_vqa([("V1", 100, "red")])
        assert rows[0].answer == "red"

    def test_trake_validates_same_video_and_order(self):
        builder = SubmissionBuilder()
        rows = builder.build_trake([
            ("V1", 100), ("V1", 200), ("V1", 150),  # bad: 150 < 200
            ("V2", 10), ("V2", 20), ("V2", 30),      # good
        ], n_events=3)
        # Only V2 sequence is valid
        assert all(r.video_id == "V2" for r in rows)
        assert len(rows) == 3

    def test_trake_limit_is_total_csv_rows_not_sequence_count(self):
        builder = SubmissionBuilder(max_rows=5)
        rows = builder.build_trake(
            [
                ("V1", 1), ("V1", 2), ("V1", 3),
                ("V2", 1), ("V2", 2), ("V2", 3),
            ],
            n_events=3,
        )

        assert len(rows) == 3

    def test_trake_rejects_non_positive_event_count(self):
        with pytest.raises(ValueError, match="n_events"):
            SubmissionBuilder().build_trake([("V1", 1)], n_events=0)

    def test_csv_output_no_header(self):
        builder = SubmissionBuilder()
        rows = [SubmissionRow(video_id="V1", frame_idx=100)]
        csv = builder.write_csv(rows, "KIS")
        assert not csv.startswith("video_id")
        assert "V1,100" in csv

    def test_zip_structure(self):
        builder = SubmissionBuilder()
        csv_content = "V1,100\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = builder.write_zip(csv_content, "q1", Path(tmp) / "sub.zip")
            assert path.exists()
            with zipfile.ZipFile(path) as zf:
                assert "submission/q1.csv" in zf.namelist()

    def test_validate_duplicate(self):
        builder = SubmissionBuilder()
        rows = [
            SubmissionRow(video_id="V1", frame_idx=100),
            SubmissionRow(video_id="V1", frame_idx=100),
        ]
        decisions = builder.validate(rows, "KIS")
        assert any(d.reason_code == "DUPLICATE_SUBMISSION_ROW" for d in decisions)


class TestRoundManifest:
    """W14: RoundManifest."""

    def test_default_manifest(self):
        assert DEFAULT_MANIFEST.max_submissions == 4
        assert DEFAULT_MANIFEST.max_rows_per_submission == 100
        assert "KIS" in DEFAULT_MANIFEST.tasks

    def test_custom_manifest(self):
        manifest = RoundManifest(
            round_id="round-1",
            max_submissions=3,
            tasks=["KIS", "VQA"],
        )
        assert manifest.max_submissions == 3
        assert "TRAKE" not in manifest.tasks
