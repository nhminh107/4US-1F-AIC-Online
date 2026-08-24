"""Verify restored PostgreSQL data needed by the AIC runtime package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.Database.postgre_manager import PostgreManager


COUNT_QUERIES: dict[str, str] = {
    "video": "SELECT count(*) FROM public.video",
    "shot": "SELECT count(*) FROM public.shot",
    "frame": "SELECT count(*) FROM public.frame",
    "ocr": "SELECT count(*) FROM public.ocr",
    "caption": "SELECT count(*) FROM public.caption",
    "classid": "SELECT count(*) FROM public.classid",
    "objectdetection": "SELECT count(*) FROM public.objectdetection",
    "objecttrack": "SELECT count(*) FROM public.objecttrack",
    "trackobservation": "SELECT count(*) FROM public.trackobservation",
    "clipwindow": "SELECT count(*) FROM public.clipwindow",
    "transcriptsegment": "SELECT count(*) FROM public.transcriptsegment",
}

ORPHAN_QUERIES: dict[str, str] = {
    "shot_video_orphans": """
        SELECT count(*)
        FROM public.shot s
        LEFT JOIN public.video v ON v.video_id = s.video_id
        WHERE v.video_id IS NULL
    """,
    "frame_video_orphans": """
        SELECT count(*)
        FROM public.frame f
        LEFT JOIN public.video v ON v.video_id = f.video_id
        WHERE v.video_id IS NULL
    """,
    "ocr_frame_orphans": """
        SELECT count(*)
        FROM public.ocr o
        LEFT JOIN public.frame f ON f.frame_id = o.frame_id
        WHERE f.frame_id IS NULL
    """,
    "objectdetection_frame_orphans": """
        SELECT count(*)
        FROM public.objectdetection od
        LEFT JOIN public.frame f ON f.frame_id = od.frame_id
        WHERE f.frame_id IS NULL
    """,
    "objectdetection_class_orphans": """
        SELECT count(*)
        FROM public.objectdetection od
        LEFT JOIN public.classid c ON c.class_id = od.class_id
        WHERE c.class_id IS NULL
    """,
    "objecttrack_shot_orphans": """
        SELECT count(*)
        FROM public.objecttrack ot
        LEFT JOIN public.shot s ON s.shot_id = ot.shot_id
        WHERE s.shot_id IS NULL
    """,
    "objecttrack_class_orphans": """
        SELECT count(*)
        FROM public.objecttrack ot
        LEFT JOIN public.classid c ON c.class_id = ot.class_id
        WHERE c.class_id IS NULL
    """,
    "trackobservation_track_orphans": """
        SELECT count(*)
        FROM public.trackobservation obs
        LEFT JOIN public.objecttrack ot ON ot.track_id = obs.track_id
        WHERE ot.track_id IS NULL
    """,
    "clipwindow_shot_orphans": """
        SELECT count(*)
        FROM public.clipwindow cw
        LEFT JOIN public.shot s ON s.shot_id = cw.shot_id
        WHERE s.shot_id IS NULL
    """,
    "transcript_video_orphans": """
        SELECT count(*)
        FROM public.transcriptsegment ts
        LEFT JOIN public.video v ON v.video_id = ts.video_id
        WHERE v.video_id IS NULL
    """,
    "caption_target_orphans": """
        SELECT count(*)
        FROM public.caption c
        LEFT JOIN public.frame f ON f.frame_id = c.frame_id
        LEFT JOIN public.shot s ON s.shot_id = c.shot_id
        LEFT JOIN public.clipwindow cw ON cw.clip_id = c.clip_id
        WHERE (c.frame_id IS NOT NULL AND f.frame_id IS NULL)
           OR (c.shot_id IS NOT NULL AND s.shot_id IS NULL)
           OR (c.clip_id IS NOT NULL AND cw.clip_id IS NULL)
    """,
}


def verify_postgres_data(database_url: str | None = None) -> dict[str, Any]:
    manager = PostgreManager(database_url=database_url)
    try:
        with manager.engine.connect() as connection:
            counts = {
                name: int(connection.execute(text(query)).scalar_one())
                for name, query in COUNT_QUERIES.items()
            }
            orphans = {
                name: int(connection.execute(text(query)).scalar_one())
                for name, query in ORPHAN_QUERIES.items()
            }
    finally:
        manager.engine.dispose()

    empty_required_tables = [
        name
        for name, count in counts.items()
        if count <= 0
    ]
    orphan_failures = {
        name: count
        for name, count in orphans.items()
        if count != 0
    }

    return {
        "ok": not empty_required_tables and not orphan_failures,
        "counts": counts,
        "orphans": orphans,
        "empty_required_tables": empty_required_tables,
        "orphan_failures": orphan_failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify PostgreSQL runtime data after SQL restore."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL URL override. Defaults to DATABASE_URL from .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = verify_postgres_data(database_url=args.database_url)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
