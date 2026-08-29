from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_retrieval_v2 as benchmark


FIXTURE_DIR = Path(__file__).with_name("fixtures") / "v1"


def test_versioned_gold_bundle_loads_in_stable_query_order() -> None:
    bundle = benchmark.load_fixture_bundle(FIXTURE_DIR)

    assert bundle["fixture_version"] == "kis-gold-v1"
    assert [query["query_id"] for query in bundle["queries"]] == [
        "query-p1-2-kis",
        "query-p1-4-kis",
        "query-p1-18-kis",
        "query-p1-19-kis",
        "query-p1-20-kis",
        "query-p1-24-kis",
    ]
    assert all(query["accepted"]["frames"] for query in bundle["queries"])

    categories = {
        negative["category"]
        for query in bundle["queries"]
        for negative in query["hard_negatives"]
    }
    assert categories == {
        "correct_video_wrong_moment",
        "ocr_asr_noun_leakage",
        "same_noun_wrong_action",
        "wrong_count_relation",
        "wrong_event_order",
    }


def test_replay_report_emits_deterministic_metrics_for_every_stage() -> None:
    first = benchmark.run_replay(FIXTURE_DIR)
    second = benchmark.run_replay(FIXTURE_DIR)

    assert first == second
    assert first["schema_version"] == "retrieval-v2-benchmark-report/v1"
    assert first["fixture_version"] == "kis-gold-v1"
    assert [stage["stage"] for stage in first["stages"]] == [
        "global",
        "reviewed",
    ]

    global_metrics, reviewed_metrics = first["stages"]
    assert global_metrics["query_count"] == 6
    assert global_metrics["video_recall"] == pytest.approx(0.5)
    assert global_metrics["moment_recall"] == pytest.approx(1 / 3)
    assert global_metrics["official_frame_hit_rate"] == pytest.approx(1 / 3)
    assert global_metrics["hard_negative_exposure_count"] > 0

    assert reviewed_metrics["video_recall"] == 1.0
    assert reviewed_metrics["moment_recall"] == 1.0
    assert reviewed_metrics["official_frame_hit_rate"] == 1.0
    assert reviewed_metrics["hard_negative_rejection_rate"] == 1.0

    smoke = first["controller_smoke"]
    assert "not a live retrieval benchmark" in smoke["purpose"]
    assert [stage["stage"] for stage in smoke["stages"]] == [
        "controller_global",
        "controller_reviewed",
    ]
    assert smoke["stages"][0]["video_recall"] == global_metrics["video_recall"]
    assert smoke["stages"][1]["moment_recall"] == reviewed_metrics["moment_recall"]


def test_fixture_validation_rejects_a_hard_negative_inside_gold(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "v1"
    fixture_dir.mkdir()
    (fixture_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "retrieval-v2-fixture-manifest/v1",
                "fixture_version": "bad-v1",
                "gold_file": "gold_kis.json",
                "replay_file": "replay_baseline.json",
            }
        ),
        encoding="utf-8",
    )
    (fixture_dir / "gold_kis.json").write_text(
        json.dumps(
            {
                "schema_version": "retrieval-v2-kis-gold/v1",
                "queries": [
                    {
                        "query_id": "q1",
                        "task": "KIS",
                        "description_vi": "fixture",
                        "accepted": {
                            "video_id": "L01_V001",
                            "intervals_ms": [[1000, 2000]],
                            "frames": [10],
                        },
                        "hard_negatives": [
                            {
                                "id": "bad-negative",
                                "category": "correct_video_wrong_moment",
                                "video_id": "L01_V001",
                                "interval_ms": [1500, 1600],
                                "frame_idx": 10,
                                "reason": "Overlaps accepted evidence.",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (fixture_dir / "replay_baseline.json").write_text(
        json.dumps(
            {
                "schema_version": "retrieval-v2-replay/v1",
                "stages": [
                    {
                        "name": "global",
                        "queries": {"q1": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(benchmark.FixtureValidationError, match="overlaps accepted"):
        benchmark.load_fixture_bundle(fixture_dir)


def test_cli_prints_one_machine_readable_json_document(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = benchmark.main(["--fixtures", str(FIXTURE_DIR)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "retrieval-v2-benchmark-report/v1"
    assert report["summary"]["query_count"] == 6
