"""Deterministic, database-free replay benchmark for Retrieval V2 fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_FIXTURE_DIR = ROOT / "BackEnd" / "tests" / "retrieval_v2" / "fixtures" / "v1"

MANIFEST_SCHEMA = "retrieval-v2-fixture-manifest/v1"
GOLD_SCHEMA = "retrieval-v2-kis-gold/v1"
REPLAY_SCHEMA = "retrieval-v2-replay/v1"
REPORT_SCHEMA = "retrieval-v2-benchmark-report/v1"
VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")
HARD_NEGATIVE_CATEGORIES = {
    "correct_video_wrong_moment",
    "ocr_asr_noun_leakage",
    "same_noun_wrong_action",
    "wrong_count_relation",
    "wrong_event_order",
}


class FixtureValidationError(ValueError):
    """Raised when a benchmark fixture is malformed or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FixtureValidationError(f"Fixture file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FixtureValidationError(f"Fixture root must be an object: {path}")
    return payload


def _require_schema(payload: dict[str, Any], expected: str, path: Path) -> None:
    if payload.get("schema_version") != expected:
        raise FixtureValidationError(
            f"Unexpected schema_version in {path}: expected {expected!r}"
        )


def _validate_video_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not VIDEO_ID_PATTERN.fullmatch(value):
        raise FixtureValidationError(f"{context}.video_id is invalid: {value!r}")
    return value


def _validate_interval(value: Any, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise FixtureValidationError(f"{context} must be [start_ms, end_ms] with end > start")
    return value[0], value[1]


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _natural_key(value: str) -> tuple[tuple[int, str | int], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", value)
    )


def _load_manifest(fixture_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = fixture_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    _require_schema(manifest, MANIFEST_SCHEMA, manifest_path)
    fixture_version = manifest.get("fixture_version")
    if not isinstance(fixture_version, str) or not fixture_version:
        raise FixtureValidationError("manifest.fixture_version must be a non-empty string")

    paths: list[Path] = []
    fixture_root = fixture_dir.resolve()
    for field in ("gold_file", "replay_file"):
        filename = manifest.get(field)
        if not isinstance(filename, str) or not filename:
            raise FixtureValidationError(f"manifest.{field} must be a non-empty string")
        path = (fixture_dir / filename).resolve()
        if path.parent != fixture_root:
            raise FixtureValidationError(f"manifest.{field} must name a file inside the fixture directory")
        paths.append(path)
    return manifest, paths[0], paths[1]


def _validate_gold(gold: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    _require_schema(gold, GOLD_SCHEMA, path)
    queries = gold.get("queries")
    if not isinstance(queries, list) or not queries:
        raise FixtureValidationError("gold.queries must be a non-empty list")

    seen_query_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for query in queries:
        if not isinstance(query, dict):
            raise FixtureValidationError("Every gold query must be an object")
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise FixtureValidationError("query_id must be a non-empty string")
        if query_id in seen_query_ids:
            raise FixtureValidationError(f"Duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        if query.get("task") != "KIS":
            raise FixtureValidationError(f"{query_id}.task must be KIS")
        if not isinstance(query.get("description_vi"), str) or not query["description_vi"]:
            raise FixtureValidationError(f"{query_id}.description_vi must be non-empty")

        accepted = query.get("accepted")
        if not isinstance(accepted, dict):
            raise FixtureValidationError(f"{query_id}.accepted must be an object")
        accepted_video = _validate_video_id(accepted.get("video_id"), f"{query_id}.accepted")
        intervals_raw = accepted.get("intervals_ms")
        if not isinstance(intervals_raw, list) or not intervals_raw:
            raise FixtureValidationError(f"{query_id}.accepted.intervals_ms must be non-empty")
        intervals = [
            _validate_interval(interval, f"{query_id}.accepted.intervals_ms[{index}]")
            for index, interval in enumerate(intervals_raw)
        ]
        frames = accepted.get("frames")
        if (
            not isinstance(frames, list)
            or not frames
            or any(not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 for frame in frames)
            or frames != sorted(set(frames))
        ):
            raise FixtureValidationError(
                f"{query_id}.accepted.frames must be a sorted, unique, non-negative integer list"
            )

        negatives = query.get("hard_negatives")
        if not isinstance(negatives, list) or not negatives:
            raise FixtureValidationError(f"{query_id}.hard_negatives must be non-empty")
        seen_negative_ids: set[str] = set()
        for negative in negatives:
            if not isinstance(negative, dict):
                raise FixtureValidationError(f"{query_id} hard negative must be an object")
            negative_id = negative.get("id")
            if not isinstance(negative_id, str) or not negative_id:
                raise FixtureValidationError(f"{query_id} hard negative id must be non-empty")
            if negative_id in seen_negative_ids:
                raise FixtureValidationError(f"{query_id} has duplicate hard negative {negative_id}")
            seen_negative_ids.add(negative_id)
            category = negative.get("category")
            if category not in HARD_NEGATIVE_CATEGORIES:
                raise FixtureValidationError(f"{query_id}.{negative_id} has invalid category {category!r}")
            negative_video = _validate_video_id(negative.get("video_id"), f"{query_id}.{negative_id}")
            negative_interval = None
            if "interval_ms" in negative:
                negative_interval = _validate_interval(
                    negative["interval_ms"], f"{query_id}.{negative_id}.interval_ms"
                )
            negative_frame = negative.get("frame_idx")
            if negative_frame is not None and (
                not isinstance(negative_frame, int)
                or isinstance(negative_frame, bool)
                or negative_frame < 0
            ):
                raise FixtureValidationError(f"{query_id}.{negative_id}.frame_idx is invalid")
            if not isinstance(negative.get("reason"), str) or not negative["reason"]:
                raise FixtureValidationError(f"{query_id}.{negative_id}.reason must be non-empty")
            if negative_video == accepted_video:
                frame_conflict = negative_frame is not None and negative_frame in frames
                interval_conflict = negative_interval is not None and any(
                    _overlaps(negative_interval, interval) for interval in intervals
                )
                if frame_conflict or interval_conflict:
                    raise FixtureValidationError(
                        f"{query_id}.{negative_id} overlaps accepted evidence"
                    )
        validated.append(query)
    return sorted(validated, key=lambda item: _natural_key(item["query_id"]))


def _validate_candidate(candidate: Any, context: str) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise FixtureValidationError(f"{context} candidate must be an object")
    _validate_video_id(candidate.get("video_id"), context)
    if "interval_ms" in candidate:
        _validate_interval(candidate["interval_ms"], f"{context}.interval_ms")
    frame = candidate.get("frame_idx")
    if frame is not None and (
        not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
    ):
        raise FixtureValidationError(f"{context}.frame_idx is invalid")
    if "interval_ms" not in candidate and frame is None:
        raise FixtureValidationError(f"{context} needs interval_ms or frame_idx")
    return candidate


def _validate_replay(
    replay: dict[str, Any], path: Path, query_ids: set[str]
) -> list[dict[str, Any]]:
    _require_schema(replay, REPLAY_SCHEMA, path)
    stages = replay.get("stages")
    if not isinstance(stages, list) or not stages:
        raise FixtureValidationError("replay.stages must be a non-empty list")
    seen_stage_names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise FixtureValidationError("Every replay stage must be an object")
        name = stage.get("name")
        if not isinstance(name, str) or not name:
            raise FixtureValidationError("Replay stage name must be non-empty")
        if name in seen_stage_names:
            raise FixtureValidationError(f"Duplicate replay stage: {name}")
        seen_stage_names.add(name)
        query_candidates = stage.get("queries")
        if not isinstance(query_candidates, dict) or set(query_candidates) != query_ids:
            raise FixtureValidationError(f"Stage {name} must contain exactly the gold query ids")
        for query_id in sorted(query_candidates):
            candidates = query_candidates[query_id]
            if not isinstance(candidates, list):
                raise FixtureValidationError(f"{name}.{query_id} candidates must be a list")
            for rank, candidate in enumerate(candidates, start=1):
                _validate_candidate(candidate, f"{name}.{query_id}[{rank}]")
        validated.append(stage)
    return validated


def load_fixture_bundle(fixture_dir: Path | str = DEFAULT_FIXTURE_DIR) -> dict[str, Any]:
    """Load and validate one versioned gold/replay fixture bundle."""

    root = Path(fixture_dir)
    manifest, gold_path, replay_path = _load_manifest(root)
    gold = _read_json(gold_path)
    queries = _validate_gold(gold, gold_path)
    replay = _read_json(replay_path)
    stages = _validate_replay(replay, replay_path, {query["query_id"] for query in queries})
    return {
        "fixture_version": manifest["fixture_version"],
        "source_artifacts": list(manifest.get("source_artifacts", [])),
        "queries": queries,
        "stages": stages,
    }


def _candidate_interval(candidate: dict[str, Any]) -> tuple[int, int] | None:
    value = candidate.get("interval_ms")
    return (value[0], value[1]) if value is not None else None


def _candidate_hits_moment(candidate: dict[str, Any], query: dict[str, Any]) -> bool:
    accepted = query["accepted"]
    if candidate["video_id"] != accepted["video_id"]:
        return False
    if candidate.get("frame_idx") in accepted["frames"]:
        return True
    interval = _candidate_interval(candidate)
    return interval is not None and any(
        _overlaps(interval, (accepted_interval[0], accepted_interval[1]))
        for accepted_interval in accepted["intervals_ms"]
    )


def _candidate_hits_frame(candidate: dict[str, Any], query: dict[str, Any]) -> bool:
    accepted = query["accepted"]
    return (
        candidate["video_id"] == accepted["video_id"]
        and candidate.get("frame_idx") in accepted["frames"]
    )


def _matches_negative(candidate: dict[str, Any], negative: dict[str, Any]) -> bool:
    if candidate["video_id"] != negative["video_id"]:
        return False
    if negative.get("frame_idx") is not None:
        return candidate.get("frame_idx") == negative["frame_idx"]
    negative_interval = negative.get("interval_ms")
    candidate_interval = _candidate_interval(candidate)
    if negative_interval is not None and candidate_interval is not None:
        return _overlaps(candidate_interval, (negative_interval[0], negative_interval[1]))
    return True


def _first_rank(
    candidates: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> int | None:
    return next((rank for rank, candidate in enumerate(candidates, start=1) if predicate(candidate)), None)


def _evaluate_query_stage(
    query: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    accepted_video = query["accepted"]["video_id"]
    video_rank = _first_rank(candidates, lambda item: item["video_id"] == accepted_video)
    moment_rank = _first_rank(candidates, lambda item: _candidate_hits_moment(item, query))
    frame_rank = _first_rank(candidates, lambda item: _candidate_hits_frame(item, query))
    exposed: dict[str, str] = {}
    for candidate in candidates:
        for negative in query["hard_negatives"]:
            if _matches_negative(candidate, negative):
                exposed[negative["id"]] = negative["category"]
    return {
        "query_id": query["query_id"],
        "candidate_count": len(candidates),
        "unique_video_count": len({candidate["video_id"] for candidate in candidates}),
        "video_hit": video_rank is not None,
        "video_best_rank": video_rank,
        "moment_hit": moment_rank is not None,
        "moment_best_rank": moment_rank,
        "official_frame_hit": frame_rank is not None,
        "official_frame_best_rank": frame_rank,
        "hard_negative_total": len(query["hard_negatives"]),
        "hard_negative_exposed": sorted(exposed),
        "hard_negative_categories": dict(sorted(Counter(exposed.values()).items())),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _evaluate_stage(
    stage: dict[str, Any], queries: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query = [
        _evaluate_query_stage(query, stage["queries"][query["query_id"]])
        for query in queries
    ]
    query_count = len(per_query)
    negative_total = sum(item["hard_negative_total"] for item in per_query)
    negative_exposed = sum(len(item["hard_negative_exposed"]) for item in per_query)
    metrics = {
        "stage": stage["name"],
        "query_count": query_count,
        "candidate_count": sum(item["candidate_count"] for item in per_query),
        "unique_video_count_sum": sum(item["unique_video_count"] for item in per_query),
        "video_recall": _rate(sum(item["video_hit"] for item in per_query), query_count),
        "moment_recall": _rate(sum(item["moment_hit"] for item in per_query), query_count),
        "official_frame_hit_rate": _rate(
            sum(item["official_frame_hit"] for item in per_query), query_count
        ),
        "hard_negative_total": negative_total,
        "hard_negative_exposure_count": negative_exposed,
        "hard_negative_rejection_rate": _rate(negative_total - negative_exposed, negative_total),
    }
    return metrics, per_query


def run_replay(fixture_dir: Path | str = DEFAULT_FIXTURE_DIR) -> dict[str, Any]:
    """Evaluate frozen replay candidates against versioned gold judgments."""

    bundle = load_fixture_bundle(fixture_dir)
    stage_metrics: list[dict[str, Any]] = []
    query_metrics: list[dict[str, Any]] = []
    for stage in bundle["stages"]:
        metrics, per_query = _evaluate_stage(stage, bundle["queries"])
        stage_metrics.append(metrics)
        query_metrics.extend({"stage": stage["name"], **item} for item in per_query)
    final_stage = stage_metrics[-1]
    report = {
        "schema_version": REPORT_SCHEMA,
        "fixture_version": bundle["fixture_version"],
        "summary": {
            "task": "KIS",
            "query_count": len(bundle["queries"]),
            "stage_count": len(stage_metrics),
            "final_video_recall": final_stage["video_recall"],
            "final_moment_recall": final_stage["moment_recall"],
            "final_official_frame_hit_rate": final_stage["official_frame_hit_rate"],
            "final_hard_negative_rejection_rate": final_stage[
                "hard_negative_rejection_rate"
            ],
        },
        "stages": stage_metrics,
        "queries": query_metrics,
    }
    report["controller_smoke"] = _run_controller_smoke(bundle)
    return report


def _run_controller_smoke(bundle: dict[str, Any]) -> dict[str, Any]:
    """Replay frozen candidates through production planning/controller code.

    This validates orchestration and contracts only. The reviewed fixture is a
    frozen human-confirmed artifact, so these metrics are deliberately kept
    separate from promotion-quality live retrieval metrics.
    """

    from BackEnd.app.contracts.models import SearchHit, StructuredQuery
    from BackEnd.app.retrieval_v2.contracts import CandidateBudget
    from BackEnd.app.retrieval_v2.controller import SearchController

    async def run_stage(stage: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        query_outputs: dict[str, list[dict[str, Any]]] = {}
        for query in bundle["queries"]:
            frozen = stage["queries"][query["query_id"]]

            class FrozenGateway:
                async def search_many(self, calls):
                    groups = []
                    for call in calls:
                        allowed = set(call.video_ids)
                        candidates = [
                            item
                            for item in frozen
                            if not allowed or item["video_id"] in allowed
                        ]
                        groups.append([
                            SearchHit(
                                source="fixture_replay",
                                entity_type="shot",
                                entity_id=f"{call.call_id}:{index}",
                                video_id=item["video_id"],
                                start_ms=item.get("interval_ms", [item.get("frame_idx", 0), item.get("frame_idx", 0)])[0],
                                end_ms=item.get("interval_ms", [item.get("frame_idx", 0), item.get("frame_idx", 0)])[1],
                                rank=index,
                                raw_score=max(0.0, 1.0 - index * 0.05),
                            )
                            for index, item in enumerate(candidates[: call.top_k], start=1)
                        ])
                    return groups

            controller = SearchController(
                FrozenGateway(),
                budget=CandidateBudget(
                    raw_retrieval_target=20,
                    raw_retrieval_max=20,
                    unique_candidate_min=1,
                    unique_candidate_max=100,
                    moment_band_limit=100,
                    video_shortlist_limit=20,
                    local_retrieval_k=20,
                    retry_retrieval_k=10,
                    rerank_limit=100,
                    max_retry_rounds=0,
                ),
            )
            result = await controller.search(
                StructuredQuery(
                    query_id=query["query_id"],
                    task="KIS",
                    visual_queries=[query["description_vi"]],
                )
            )
            output: list[dict[str, Any]] = []
            for band in result.reranked_bands:
                item: dict[str, Any] = {
                    "video_id": band.video_id,
                    "interval_ms": [band.start_ms, band.end_ms],
                }
                matching = next(
                    (
                        candidate
                        for candidate in frozen
                        if candidate["video_id"] == band.video_id
                        and "frame_idx" in candidate
                        and _overlaps(
                            (band.start_ms, band.end_ms),
                            _candidate_interval(candidate) or (band.start_ms, band.end_ms),
                        )
                    ),
                    None,
                )
                if matching is not None:
                    item["frame_idx"] = matching["frame_idx"]
                output.append(item)
            query_outputs[query["query_id"]] = output
        return _evaluate_stage(
            {"name": f"controller_{stage['name']}", "queries": query_outputs},
            bundle["queries"],
        )

    stage_metrics: list[dict[str, Any]] = []
    query_metrics: list[dict[str, Any]] = []
    for stage in bundle["stages"]:
        metrics, per_query = asyncio.run(run_stage(stage))
        stage_metrics.append(metrics)
        query_metrics.extend({"stage": metrics["stage"], **item} for item in per_query)
    return {
        "purpose": "production-controller plumbing smoke test; not a live retrieval benchmark",
        "stages": stage_metrics,
        "queries": query_metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory containing manifest.json for one fixture version.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    return parser


def render_report(report: dict[str, Any], *, pretty: bool = False) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_replay(args.fixtures)
    rendered = render_report(report, pretty=args.pretty)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
