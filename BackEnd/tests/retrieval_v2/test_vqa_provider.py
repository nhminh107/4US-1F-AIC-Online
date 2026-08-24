from pathlib import Path
from types import SimpleNamespace

from BackEnd.app.vqa.handler import VQAModelAnswer
from BackEnd.app.vqa.provider import GroundedVQAProvider, build_vqa_instruction


class Client:
    def __init__(self) -> None:
        self.prompt = ""

    def answer(self, *, question, prompt, image_paths):
        self.prompt = prompt
        assert question
        assert image_paths
        return VQAModelAnswer(answer="7", confidence=0.9)


def test_grounded_vqa_provider_uses_only_allowlisted_official_frames(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")

    class DB:
        def get_frame_record_by_frame_id(self, frame_id):
            return SimpleNamespace(frame_path=str(image) if frame_id == "F1" else None)

    frames = [
        SimpleNamespace(display_frame_id="F1"),
        SimpleNamespace(display_frame_id="F2"),
    ]
    client = Client()
    provider = GroundedVQAProvider(DB(), client)

    import asyncio
    claims = asyncio.run(provider.answer_claims(
        "How many magnitude 4 markers are on the map, excluding the legend?",
        frames,
        {"F1"},
    ))

    assert len(claims) == 1
    assert claims[0].evidence_id == "F1"
    assert claims[0].answer == "7"
    assert "Exclude every sample symbol" in client.prompt


def test_map_legend_instruction_is_not_added_to_unrelated_questions():
    assert "legend box" not in build_vqa_instruction("What color is the car?")
