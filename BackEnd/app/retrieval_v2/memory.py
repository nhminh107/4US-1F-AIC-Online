from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel


class GoldRecord(ContractModel):
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    task: Literal["KIS", "VQA", "TRAKE"]
    accepted_video_ids: list[str] = Field(default_factory=list)
    accepted_intervals_ms: dict[str, list[list[int]]] = Field(default_factory=dict)
    rejected_regions: list[str] = Field(default_factory=list)
    effective_prompts: list[str] = Field(default_factory=list)
    review_verdicts: dict[str, str] = Field(default_factory=dict)


class JsonlGoldMemory:
    """Append-only evaluation memory; never used as a blind answer lookup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: GoldRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")

    def list_records(self) -> list[GoldRecord]:
        if not self.path.exists():
            return []
        with self._lock, self.path.open("r", encoding="utf-8") as stream:
            return [
                GoldRecord.model_validate_json(line)
                for line in stream
                if line.strip()
            ]


__all__ = ["GoldRecord", "JsonlGoldMemory"]
