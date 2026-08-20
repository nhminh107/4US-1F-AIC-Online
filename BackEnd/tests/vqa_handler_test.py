from BackEnd.app.contracts.models import (
    CandidateEvidence,
    ConstraintResult,
    EvidenceBundle,
    RankedCandidateRegion,
    StructuredQuery,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.vqa import VQAHandler


def _candidate(
    candidate_id: str,
    *,
    score: float,
    hard_constraints_passed: bool = True,
    negative_constraints_passed: bool = True,
) -> RankedCandidateRegion:
    return RankedCandidateRegion(
        candidate_id=candidate_id,
        video_id="L01_V001",
        start_ms=1000,
        end_ms=2000,
        fusion_score=score,
        constraint_result=ConstraintResult(
            hard_constraints_passed=hard_constraints_passed,
            negative_constraints_passed=negative_constraints_passed,
        ),
        evidence=[
            CandidateEvidence(
                source="clip",
                entity_type="frame",
                entity_id=f"evidence-{candidate_id}",
                start_ms=1000,
                end_ms=2000,
                rank=1,
                raw_score=score,
            )
        ],
    )


def _bundle(video_id: str, start_ms: int, end_ms: int) -> EvidenceBundle:
    return EvidenceBundle(
        video_id=video_id,
        start_ms=start_ms,
        end_ms=end_ms,
        frames=[
            FrameMetadata(
                frame_id="F-near",
                video_id=video_id,
                shot_id="S001",
                timestamp_ms=1490,
                fps=30.0,
                frame_idx=45,
            ),
            FrameMetadata(
                frame_id="F-far",
                video_id=video_id,
                shot_id="S001",
                timestamp_ms=1900,
                fps=30.0,
                frame_idx=57,
            ),
        ],
    )


def test_vqa_handler_returns_kis_style_ranked_candidates_for_human_review():
    query = StructuredQuery(query_id="Q1", task="VQA", question="What is visible?")
    results = VQAHandler(max_candidates=2).handle(
        query,
        [_candidate("low", score=0.5), _candidate("high", score=0.9)],
        _bundle,
    )

    assert [result.score for result in results] == [0.9, 0.5]
    assert all(result.representative_frame_id == "F-near" for result in results)
    assert results[0].evidence_ids == ["F-near", "evidence-high"]


def test_vqa_handler_skips_failed_constraints_and_candidates_without_frames():
    query = StructuredQuery(query_id="Q1", task="VQA", question="What is visible?")

    def empty_bundle(video_id: str, start_ms: int, end_ms: int) -> EvidenceBundle:
        return EvidenceBundle(video_id=video_id, start_ms=start_ms, end_ms=end_ms)

    results = VQAHandler().handle(
        query,
        [
            _candidate("failed", score=0.99, hard_constraints_passed=False),
            _candidate("no-frame", score=0.9),
        ],
        empty_bundle,
    )

    assert results == []


def test_vqa_handler_requires_a_vqa_question():
    handler = VQAHandler()
    query = StructuredQuery(query_id="Q1", task="KIS")

    try:
        handler.handle(query, [], _bundle)
    except ValueError as error:
        assert "task='VQA'" in str(error)
    else:
        raise AssertionError("Expected a VQA task validation error")
