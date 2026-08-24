"""Test suite for competition exporter and docker guard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from BackEnd.app.api.models import (
    QueryResponse,
    TopKParameters,
    TrakeEventResult,
    TrakeSequenceResult,
    VerificationSummary,
)
from BackEnd.app.contracts.models import StructuredQuery
from scripts.competition.exporter import ResultExporter


def test_result_exporter_kis() -> None:
    with TemporaryDirectory() as tmpdir:
        exporter = ResultExporter(output_base_dir=tmpdir)
        mock_response = {
            "query_id": "test_kis_01",
            "task": "KIS",
            "results": [
                {
                    "video_id": "L01_V001",
                    "frame_idx": 1250,
                    "score": 0.95,
                    "start_ms": 50000,
                    "display_frame_id": "L01_V001_125",
                    "img_url": "data/keyframes/L01_V001/125.jpg",
                },
                {
                    "video_id": "L01_V002",
                    "frame_idx": 3400,
                    "score": 0.88,
                    "start_ms": 136000,
                    "display_frame_id": "L01_V002_340",
                    "img_url": "data/keyframes/L01_V002/340.jpg",
                },
            ],
            "verification": {
                "status": "accepted",
                "confidence": 0.95,
            },
        }

        exported = exporter.export(
            query_id="test_kis_01",
            task="KIS",
            prompt="Phi hành gia mặc áo đen",
            api_response=mock_response,
            latency_ms=120.5,
        )

        assert exported["csv_file"].exists()
        assert exported["html_file"].exists()
        assert exported["txt_file"].exists()
        assert exported["response_file"].exists()
        assert exported["audit_file"].exists()

        # Check CSV content
        with open(exported["csv_file"], newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2
            assert reader[0] == ["L01_V001", "1250"]
            assert reader[1] == ["L01_V002", "3400"]

        # Check HTML content
        html_text = exported["html_file"].read_text(encoding="utf-8")
        assert "test_kis_01" in html_text
        assert "Phi hành gia mặc áo đen" in html_text
        assert "L01_V001" in html_text
        assert "https://pub-c8f3587e831a418ebf0d427203860188.r2.dev/data/keyframes/L01_V001/125.jpg" in html_text

        # Check TXT links content
        txt_text = exported["txt_file"].read_text(encoding="utf-8")
        assert "https://pub-c8f3587e831a418ebf0d427203860188.r2.dev/data/keyframes/L01_V001/125.jpg" in txt_text

        response = json.loads(exported["response_file"].read_text(encoding="utf-8"))
        assert response["task"] == "KIS"
        audit = json.loads(exported["audit_file"].read_text(encoding="utf-8"))
        assert audit["query_id"] == "test_kis_01"
        assert audit["result_count"] == 2


def test_result_exporter_vqa() -> None:
    with TemporaryDirectory() as tmpdir:
        exporter = ResultExporter(output_base_dir=tmpdir)
        mock_response = {
            "query_id": "test_vqa_01",
            "task": "VQA",
            "answer": "Xã Cam Phước Tây",
            "results": [
                {
                    "video_id": "L03_V012",
                    "frame_idx": 8420,
                    "score": 0.91,
                    "start_ms": 336800,
                    "answer": "Xã Cam Phước Tây",
                    "display_frame_id": "L03_V012_842",
                }
            ],
            "verification": {
                "status": "accepted",
                "confidence": 0.91,
            },
        }

        exported = exporter.export(
            query_id="test_vqa_01",
            task="VQA",
            prompt="Hỏi xã này có tên là gì?",
            api_response=mock_response,
            latency_ms=85.0,
        )

        assert exported["csv_file"].exists()
        with open(exported["csv_file"], newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 1
            assert reader[0] == ["L03_V012", "8420", "Xã Cam Phước Tây"]


def test_result_exporter_trake() -> None:
    with TemporaryDirectory() as tmpdir:
        exporter = ResultExporter(output_base_dir=tmpdir)
        mock_response = QueryResponse(
            query_id="test_trake_01",
            task="TRAKE",
            structured_query=StructuredQuery(query_id="test_trake_01", task="TRAKE"),
            top_k=TopKParameters(),
            execution_path="retrieval_v2",
            search_hit_count=2,
            candidate_count=1,
            results=[
                TrakeSequenceResult(
                    video_id="L15_V008",
                    sequence_score=0.92,
                    events=[
                        TrakeEventResult(
                            event_id="E1",
                            candidate_id="c1",
                            start_ms=124000,
                            end_ms=125000,
                            display_frame_id="f1",
                            frame_idx=3100,
                            img_url="data/keyframes/L15_V008/3100.jpg",
                        ),
                        TrakeEventResult(
                            event_id="E2",
                            candidate_id="c2",
                            start_ms=134000,
                            end_ms=135000,
                            display_frame_id="f2",
                            frame_idx=3350,
                            img_url="data/keyframes/L15_V008/3350.jpg",
                        ),
                    ],
                )
            ],
            verification=VerificationSummary(enabled=False, applied=False),
        ).model_dump(mode="json")

        exported = exporter.export(
            query_id="test_trake_01",
            task="TRAKE",
            prompt="E1 ... E2 ...",
            api_response=mock_response,
            latency_ms=150.0,
        )

        assert exported["csv_file"].exists()
        with open(exported["csv_file"], newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2
            assert reader[0] == ["L15_V008", "3100"]
            assert reader[1] == ["L15_V008", "3350"]
