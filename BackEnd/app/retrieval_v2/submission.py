"""Top-100 submission builder for BTC format (KIS, VQA, TRAKE).

Pipeline.md §7.4, §8.3, §9.4: builds CSV rows, validates format,
deduplicates (video_id, frame_idx), and writes ZIP with submission/ directory.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel
from BackEnd.app.retrieval_v2.constraints import HardConstraintEngine
from BackEnd.app.retrieval_v2.contracts import ConstraintDecision, MomentBand


class SubmissionRow(ContractModel):
    """One row in a BTC submission CSV."""

    video_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    answer: str | None = None  # QA only


class SubmissionBuilder:
    """Build, validate and write top-100 CSV/ZIP submissions."""

    def __init__(
        self,
        *,
        constraint_engine: HardConstraintEngine | None = None,
        official_frame_checker: Callable[[str, int], bool] | None = None,
        max_rows: int = 100,
    ) -> None:
        self.constraint_engine = constraint_engine or HardConstraintEngine()
        self.official_frame_checker = official_frame_checker
        self.max_rows = max_rows

    def build_kis(
        self,
        rows: list[tuple[str, int]],
    ) -> list[SubmissionRow]:
        """Dedup and limit KIS submission rows."""
        return self._dedup_and_limit([
            SubmissionRow(video_id=vid, frame_idx=fidx)
            for vid, fidx in rows
        ])

    def build_vqa(
        self,
        rows: list[tuple[str, int, str]],
    ) -> list[SubmissionRow]:
        """Dedup and limit VQA submission rows."""
        return self._dedup_and_limit([
            SubmissionRow(video_id=vid, frame_idx=fidx, answer=ans)
            for vid, fidx, ans in rows
        ])

    def build_trake(
        self,
        rows: list[tuple[str, int]],
        n_events: int,
    ) -> list[SubmissionRow]:
        """Validate TRAKE rows: same video per sequence, N frames, increasing time."""
        if n_events <= 0:
            raise ValueError("n_events must be positive")
        if not rows:
            return []
        # TRAKE: each sequence has n_events rows, must be same video and increasing frame_idx
        result: list[SubmissionRow] = []
        for i in range(0, len(rows), n_events):
            seq = rows[i : i + n_events]
            if len(seq) != n_events:
                break
            seq_rows = [SubmissionRow(video_id=vid, frame_idx=fidx) for vid, fidx in seq]
            # All same video
            if len({r.video_id for r in seq_rows}) != 1:
                continue
            # Increasing frame_idx
            if not all(seq_rows[j].frame_idx < seq_rows[j + 1].frame_idx for j in range(len(seq_rows) - 1)):
                continue
            if len(result) + n_events > self.max_rows:
                break
            result.extend(seq_rows)
        return result[: self.max_rows]

    def validate(
        self,
        rows: list[SubmissionRow],
        task: Literal["KIS", "VQA", "TRAKE"],
    ) -> list[ConstraintDecision]:
        """Validate submission format: dedup, official frame, column count."""
        decisions: list[ConstraintDecision] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            pair = (row.video_id, row.frame_idx)
            if pair in seen:
                decisions.append(ConstraintDecision(
                    constraint_id=f"submission:{row.video_id}:{row.frame_idx}",
                    status="FAIL",
                    scope="SUBMISSION_ROW",
                    reason_code="DUPLICATE_SUBMISSION_ROW",
                ))
            else:
                seen.add(pair)
                decisions.append(ConstraintDecision(
                    constraint_id=f"submission:{row.video_id}:{row.frame_idx}",
                    status="PASS",
                    scope="SUBMISSION_ROW",
                    reason_code="VALID_SUBMISSION_ROW",
                ))
            if self.official_frame_checker is None:
                decisions.append(ConstraintDecision(
                    constraint_id=f"official:{row.video_id}:{row.frame_idx}",
                    status="UNKNOWN",
                    scope="SUBMISSION_ROW",
                    reason_code="OFFICIAL_FRAME_NOT_CHECKED",
                ))
            elif not self.official_frame_checker(row.video_id, row.frame_idx):
                decisions.append(ConstraintDecision(
                    constraint_id=f"official:{row.video_id}:{row.frame_idx}",
                    status="FAIL",
                    scope="SUBMISSION_ROW",
                    reason_code="NON_OFFICIAL_FRAME",
                ))
            else:
                decisions.append(ConstraintDecision(
                    constraint_id=f"official:{row.video_id}:{row.frame_idx}",
                    status="PASS",
                    scope="SUBMISSION_ROW",
                    reason_code="OFFICIAL_FRAME_CONFIRMED",
                ))
            if task == "VQA" and not row.answer:
                decisions.append(ConstraintDecision(
                    constraint_id=f"submission_answer:{row.video_id}:{row.frame_idx}",
                    status="FAIL",
                    scope="SUBMISSION_ROW",
                    reason_code="MISSING_VQA_ANSWER",
                ))
        return decisions

    def write_csv(
        self,
        rows: list[SubmissionRow],
        task: Literal["KIS", "VQA", "TRAKE"],
    ) -> str:
        """Render CSV string (no header, UTF-8)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in rows:
            if task == "VQA":
                writer.writerow([row.video_id, row.frame_idx, row.answer or ""])
            else:
                writer.writerow([row.video_id, row.frame_idx])
        return buf.getvalue()

    def write_zip(
        self,
        csv_content: str,
        query_id: str,
        output_path: str | Path,
    ) -> Path:
        """Write ZIP with submission/ directory structure as required by BTC."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"submission/{query_id}.csv", csv_content)
        return path

    def _dedup_and_limit(self, rows: list[SubmissionRow]) -> list[SubmissionRow]:
        seen: set[tuple[str, int]] = set()
        deduped: list[SubmissionRow] = []
        for row in rows:
            pair = (row.video_id, row.frame_idx)
            if pair not in seen:
                seen.add(pair)
                deduped.append(row)
        return deduped[: self.max_rows]


__all__ = ["SubmissionBuilder", "SubmissionRow"]
