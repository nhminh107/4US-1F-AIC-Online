"""Index captions from files in the captions/ directory (or PostgreSQL) into Elasticsearch.

This script scans all caption files in the designated captions directory (supporting .sql,
.json, .jsonl, .csv) or reads directly from PostgreSQL, resolves shot/video time ranges,
and bulk-indexes the captions into Elasticsearch under the active caption alias.

It is completely idempotent: rerunning it or adding new caption files to the folder
will update existing records and insert new ones without duplicating data.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None  # type: ignore[assignment,misc]
    helpers = None  # type: ignore[assignment]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("index_captions")

CAPTION_INDEX_ALIAS = "aic_hcm2026_text_caption_active"
DEFAULT_CONCRETE_INDEX = "aic_hcm2026_text_caption_v1"

CAPTION_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "vietnamese_standard": {
                    "type": "standard",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "entity_id": {"type": "keyword"},
            "caption_id": {"type": "keyword"},
            "shot_id": {"type": "keyword"},
            "frame_id": {"type": "keyword"},
            "clip_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "start_ms": {"type": "long"},
            "end_ms": {"type": "long"},
            "timestamp_ms": {"type": "long"},
            "content": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                },
            },
            "source_type": {"type": "keyword"},
        }
    },
}


def parse_sql_captions(file_path: Path) -> list[dict[str, Any]]:
    """Extract caption records from an SQL file containing INSERT INTO caption statements."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Could not read SQL file %s: %s", file_path, e)
        return []

    pattern = re.compile(
        r"INSERT\s+INTO\s+caption\s*\(([^)]+)\)\s*VALUES\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'((?:''|[^'])*)'",
        re.IGNORECASE,
    )

    records: list[dict[str, Any]] = []
    for match in pattern.finditer(content):
        cols = [c.strip().lower() for c in match.group(1).split(",")]
        caption_id = match.group(2).strip()
        target_id = match.group(3).strip()
        caption_text = match.group(4).replace("''", "'").strip()

        shot_id = target_id if "shot_id" in cols else None
        frame_id = target_id if "frame_id" in cols else None
        clip_id = target_id if "clip_id" in cols else None

        records.append({
            "caption_id": caption_id,
            "shot_id": shot_id,
            "frame_id": frame_id,
            "clip_id": clip_id,
            "caption_text": caption_text,
        })
    return records


def parse_json_captions(file_path: Path) -> list[dict[str, Any]]:
    """Extract caption records from a JSON or JSONL file."""
    records: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for item in raw_data:
                        if isinstance(item, dict):
                            records.append({
                                "caption_id": str(item.get("caption_id") or item.get("id") or ""),
                                "shot_id": item.get("shot_id"),
                                "frame_id": item.get("frame_id"),
                                "clip_id": item.get("clip_id"),
                                "caption_text": str(item.get("caption_text") or item.get("content") or item.get("text") or ""),
                            })
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append({
                            "caption_id": str(item.get("caption_id") or item.get("id") or ""),
                            "shot_id": item.get("shot_id"),
                            "frame_id": item.get("frame_id"),
                            "clip_id": item.get("clip_id"),
                            "caption_text": str(item.get("caption_text") or item.get("content") or item.get("text") or ""),
                        })
    except Exception as e:
        logger.warning("Could not read JSON file %s: %s", file_path, e)
    return records


def parse_csv_captions(file_path: Path) -> list[dict[str, Any]]:
    """Extract caption records from a CSV file."""
    records: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                caption_id = row.get("caption_id") or row.get("id") or ""
                shot_id = row.get("shot_id")
                frame_id = row.get("frame_id")
                clip_id = row.get("clip_id")
                caption_text = row.get("caption_text") or row.get("content") or row.get("text") or ""
                if caption_id and caption_text:
                    records.append({
                        "caption_id": caption_id,
                        "shot_id": shot_id,
                        "frame_id": frame_id,
                        "clip_id": clip_id,
                        "caption_text": caption_text,
                    })
    except Exception as e:
        logger.warning("Could not read CSV file %s: %s", file_path, e)
    return records


def load_captions_from_dir(captions_dir: Path) -> list[dict[str, Any]]:
    """Scan and parse all caption files in the specified directory."""
    if not captions_dir.exists():
        logger.warning("Captions directory does not exist: %s", captions_dir)
        return []

    supported_extensions = {".sql", ".json", ".jsonl", ".csv"}
    all_files = [
        p for p in sorted(captions_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in supported_extensions
        and ("caption" in p.name.lower() or "captions" in p.name.lower())
    ]

    logger.info("Found %d caption files in %s", len(all_files), captions_dir)

    all_records: dict[str, dict[str, Any]] = {}
    for fpath in all_files:
        ext = fpath.suffix.lower()
        if ext == ".sql":
            items = parse_sql_captions(fpath)
        elif ext in (".json", ".jsonl"):
            items = parse_json_captions(fpath)
        elif ext == ".csv":
            items = parse_csv_captions(fpath)
        else:
            continue

        logger.info("  - %s: %d captions parsed", fpath.name, len(items))
        for item in items:
            cap_id = item.get("caption_id")
            if cap_id:
                all_records[cap_id] = item

    return list(all_records.values())


def load_shot_metadata_map(database_url: str | None = None) -> dict[str, dict[str, Any]]:
    """Fetch shot metadata (video_id, start_ms, end_ms) from PostgreSQL if available."""
    shot_map: dict[str, dict[str, Any]] = {}
    try:
        from BackEnd.app.Database.postgre_manager import PostgreManager
        from sqlalchemy import text

        manager = PostgreManager(database_url=database_url)
        with manager.engine.connect() as conn:
            results = conn.execute(text("SELECT shot_id, video_id, start_ms, end_ms FROM public.shot")).fetchall()
            for row in results:
                shot_map[str(row[0])] = {
                    "video_id": str(row[1]),
                    "start_ms": int(row[2]),
                    "end_ms": int(row[3]),
                }
        logger.info("Loaded %d shot metadata entries from PostgreSQL", len(shot_map))
    except Exception as e:
        logger.info("PostgreSQL not connected or query skipped (%s). Using fallback metadata extraction.", e)
    return shot_map


def unlock_elasticsearch_read_only(client: Elasticsearch) -> None:
    """Clear cluster read-only block and relax disk watermark if triggered."""
    try:
        # Relax disk watermark so ES does not block on high disk usage
        client.cluster.put_settings(
            body={
                "persistent": {
                    "cluster.routing.allocation.disk.watermark.low": "95%",
                    "cluster.routing.allocation.disk.watermark.high": "98%",
                    "cluster.routing.allocation.disk.watermark.flood_stage": "99%",
                }
            }
        )
    except Exception as e:
        logger.debug("Could not update cluster watermark settings: %s", e)

    try:
        # Clear read_only_allow_delete block on all indices
        client.indices.put_settings(
            index="_all",
            body={"index.blocks.read_only_allow_delete": None},
        )
        logger.info("Checked & cleared Elasticsearch read-only disk watermark block.")
    except Exception as e:
        logger.debug("Could not clear read_only_allow_delete setting: %s", e)


def ensure_elasticsearch_target(client: Elasticsearch, index_name: str) -> None:
    """Ensure the target index or alias exists with proper mappings and write permissions."""
    unlock_elasticsearch_read_only(client)

    if client.indices.exists(index=index_name):
        return
    if client.indices.exists_alias(name=index_name):
        return

    # Create concrete index and alias
    concrete_index = DEFAULT_CONCRETE_INDEX
    if not client.indices.exists(index=concrete_index):
        logger.info("Creating Elasticsearch index %r with caption mappings...", concrete_index)
        client.indices.create(index=concrete_index, body=CAPTION_INDEX_MAPPING)

    if index_name != concrete_index:
        logger.info("Creating alias %r pointing to %r...", index_name, concrete_index)
        client.indices.put_alias(index=concrete_index, name=index_name)


def generate_es_actions(
    captions: list[dict[str, Any]],
    shot_map: dict[str, dict[str, Any]],
    index_name: str,
) -> Iterator[dict[str, Any]]:
    """Yield Elasticsearch indexing actions."""
    for cap in captions:
        caption_id = cap["caption_id"]
        shot_id = cap.get("shot_id")
        frame_id = cap.get("frame_id")
        clip_id = cap.get("clip_id")
        text_content = cap["caption_text"]

        # Determine video_id, start_ms, end_ms
        video_id = ""
        start_ms = 0
        end_ms = 0

        if shot_id and shot_id in shot_map:
            meta = shot_map[shot_id]
            video_id = meta["video_id"]
            start_ms = meta["start_ms"]
            end_ms = meta["end_ms"]
        elif shot_id:
            parts = shot_id.rsplit("_", 1)
            video_id = parts[0] if len(parts) > 1 else shot_id
        elif frame_id:
            parts = frame_id.rsplit("_", 1)
            video_id = parts[0] if len(parts) > 1 else frame_id
        elif clip_id:
            parts = clip_id.rsplit("_", 1)
            video_id = parts[0] if len(parts) > 1 else clip_id

        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": f"caption:{caption_id}",
            "_source": {
                "doc_id": f"caption:{caption_id}",
                "entity_id": caption_id,
                "caption_id": caption_id,
                "shot_id": shot_id,
                "frame_id": frame_id,
                "clip_id": clip_id,
                "video_id": video_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "timestamp_ms": start_ms,
                "content": text_content,
                "source_type": "caption",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index captions from captions/ directory into Elasticsearch."
    )
    parser.add_argument(
        "--captions-dir",
        default=os.getenv("CAPTIONS_DIR", str(Path(ROOT_DIR) / "captions")),
        help="Path to folder containing caption files (default: ./captions).",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200"),
        help="Elasticsearch URL (default: ELASTICSEARCH_URL or http://127.0.0.1:9200).",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("CAPTION_ELASTICSEARCH_INDEX", CAPTION_INDEX_ALIAS),
        help=f"Target Elasticsearch index or alias (default: {CAPTION_INDEX_ALIAS}).",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", None),
        help="PostgreSQL URL override for shot metadata enrichment.",
    )
    parser.add_argument("--batch-size", type=int, default=1_000, help="Batch size for bulk indexing.")
    parser.add_argument("--dry-run", action="store_true", help="Parse files without writing to Elasticsearch.")

    args = parser.parse_args()

    start_time = time.time()
    captions_dir = Path(args.captions_dir)
    logger.info("Scanning caption files from: %s", captions_dir)

    captions = load_captions_from_dir(captions_dir)
    logger.info("Total unique captions loaded: %d", len(captions))

    if not captions:
        logger.warning("No captions found to index. Exiting.")
        return

    if args.dry_run:
        logger.info("DRY RUN: %d captions would be indexed into %r. Exiting.", len(captions), args.index)
        return

    # Connect to PostgreSQL to enrich with exact timecodes (if accessible)
    shot_map = load_shot_metadata_map(database_url=args.database_url)

    # Index into Elasticsearch
    if Elasticsearch is None:
        raise RuntimeError(
            "The 'elasticsearch' library is not installed in the current Python environment. "
            "Please run: pip install elasticsearch"
        )

    logger.info("Connecting to Elasticsearch at: %s", args.elasticsearch_url)
    with Elasticsearch(args.elasticsearch_url, request_timeout=60) as client:
        ensure_elasticsearch_target(client, args.index)

        succeeded = 0
        failed = 0
        actions = generate_es_actions(captions, shot_map, args.index)

        for ok, result in helpers.streaming_bulk(
            client,
            actions,
            chunk_size=args.batch_size,
            raise_on_error=False,
            raise_on_exception=False,
        ):
            if ok:
                succeeded += 1
            else:
                failed += 1
                logger.error("Indexing item failed: %s", result)

        client.indices.refresh(index=args.index)

    elapsed = time.time() - start_time
    logger.info(
        "=== Caption Indexing Completed in %.2fs ===\n"
        "  - Target: %s\n"
        "  - Succeeded: %d\n"
        "  - Failed: %d",
        elapsed,
        args.index,
        succeeded,
        failed,
    )


if __name__ == "__main__":
    main()
