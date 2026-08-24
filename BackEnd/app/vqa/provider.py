from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Sequence

from BackEnd.app.retrieval_v2.task_heads import AnswerClaim
from BackEnd.app.vqa.fpt_client import FPTVLMClient
from BackEnd.app.vqa.handler import VQAModelClient


def build_vqa_instruction(question: str) -> str:
    normalized = question.casefold()
    instructions = [
        "Answer only from the supplied official evidence frames.",
        "Return the shortest direct answer and do not infer hidden content.",
    ]
    if any(token in normalized for token in ("legend", "key", "chu giai", "map")):
        instructions.extend([
            "Use the legend only to identify the requested symbol class.",
            "Exclude every sample symbol inside the legend box from the count.",
            "Count matching markers only in the map body and recheck the total once.",
        ])
    return " ".join(instructions)


class GroundedVQAProvider:
    def __init__(self, db_mng: Any, client: VQAModelClient) -> None:
        self.db_mng = db_mng
        self.client = client

    async def answer_claims(
        self,
        question: str,
        frames: Sequence[Any],
        allowed_evidence_ids: set[str],
    ) -> Sequence[AnswerClaim]:
        evidence_ids: list[str] = []
        image_paths: list[Path] = []
        for frame in frames:
            evidence_id = frame.display_frame_id
            if evidence_id not in allowed_evidence_ids:
                continue
            try:
                metadata = self.db_mng.get_frame_record_by_frame_id(evidence_id)
            except ValueError:
                continue
            if metadata.frame_path is None:
                continue
            path = Path(metadata.frame_path)
            if not path.is_file():
                continue
            evidence_ids.append(evidence_id)
            image_paths.append(path)
            if len(image_paths) == 5:
                break
        if not image_paths:
            return ()

        answer = await asyncio.to_thread(
            self.client.answer,
            question=question,
            prompt=build_vqa_instruction(question),
            image_paths=image_paths,
        )
        return (
            AnswerClaim(
                evidence_id=evidence_ids[0],
                answer=answer.answer,
                confidence=answer.confidence,
            ),
        )


def build_default_vqa_provider(db_mng: Any) -> GroundedVQAProvider | None:
    if not os.getenv("VQA_API_KEY", "").strip() or not os.getenv("VQA_MODEL", "").strip():
        return None
    return GroundedVQAProvider(db_mng, FPTVLMClient.from_env())


__all__ = ["GroundedVQAProvider", "build_default_vqa_provider", "build_vqa_instruction"]
