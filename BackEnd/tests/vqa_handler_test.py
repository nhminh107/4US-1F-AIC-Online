from pathlib import Path

from BackEnd.app.contracts.models import (
    ConstraintResult,
    EvidenceBundle,
    RankedCandidateRegion,
    StructuredQuery,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.services.vqa_handler import VQAHandler, VQAModelAnswer


class FakeVLM:
    def answer(self, *, question, prompt, image_paths):
        assert "visible evidence" in prompt
        assert image_paths
        return VQAModelAnswer("A motorcycle is visible.", 0.9)


def test_vqa_handler_returns_shared_vqa_result(tmp_path):
    image = tmp_path / "frame.jpg"
    image.touch()
    query = StructuredQuery(query_id="Q1", task="VQA", question="What is visible?")
    candidate = RankedCandidateRegion(
        candidate_id="CR1",
        video_id="L01_V001",
        start_ms=1000,
        end_ms=2000,
        fusion_score=0.8,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )
    bundle = EvidenceBundle(
        video_id="L01_V001",
        start_ms=1000,
        end_ms=2000,
        frames=[
            FrameMetadata(
                frame_id="F001",
                video_id="L01_V001",
                shot_id="S001",
                timestamp_ms=1200,
                fps=30.0,
                frame_idx=36,
                frame_path=Path(image),
            )
        ],
    )

    result = VQAHandler(FakeVLM()).handle(
        query,
        [candidate],
        lambda _video_id, _start_ms, _end_ms: bundle,
    )
    assert result.status == "answered"
    assert result.confidence == 0.9
    assert result.evidence_ids == ["F001"]


def test_vqa_handler_is_uncertain_without_images():
    query = StructuredQuery(query_id="Q1", task="VQA", question="What is visible?")
    candidate = RankedCandidateRegion(
        candidate_id="CR1",
        video_id="L01_V001",
        start_ms=0,
        end_ms=1000,
        fusion_score=0.8,
        constraint_result=ConstraintResult(
            hard_constraints_passed=True,
            negative_constraints_passed=True,
        ),
    )
    bundle = EvidenceBundle(video_id="L01_V001", start_ms=0, end_ms=1000)
    result = VQAHandler(FakeVLM()).handle(
        query,
        [candidate],
        lambda _video_id, _start_ms, _end_ms: bundle,
    )
    assert result.status == "uncertain"
    assert result.confidence == 0.0
