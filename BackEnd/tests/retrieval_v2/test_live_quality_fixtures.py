from __future__ import annotations

import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "live_quality_v1"


def test_live_quality_manifest_references_valid_cases() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "retrieval-live-quality-v1"
    assert len(manifest["cases"]) == 7
    for filename in manifest["cases"]:
        payload = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
        assert payload["task"] in {"KIS", "VQA", "TRAKE"}
        assert payload["prompt"].strip()
        assert payload["gold"]["status"] in {"reviewed", "needs_review"}


def test_gold_fixtures_are_not_imported_by_runtime() -> None:
    runtime = Path(__file__).parents[2] / "app"
    offenders = [
        path
        for path in runtime.rglob("*.py")
        if "live_quality_v1" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
