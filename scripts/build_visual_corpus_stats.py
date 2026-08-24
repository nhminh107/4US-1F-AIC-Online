"""Build Retrieval V2 IDF statistics from visual captions and video metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Database.sql_models import Caption, ClipWindow, Frame, Shot, Video
from BackEnd.app.retrieval_v2.corpus_stats import CorpusDocument, CorpusStats


def build(output: Path) -> dict[str, int | str]:
    db = PostgreManager()
    documents: list[CorpusDocument] = []
    with db.session_factory() as session:
        for video in session.scalars(select(Video).order_by(Video.video_id)):
            text = " ".join(
                str(value).strip()
                for value in (video.title, video.description, video.keywords)
                if value and str(value).strip()
            )
            if text:
                documents.append(CorpusDocument(
                    document_id=f"video:{video.video_id}",
                    video_id=video.video_id,
                    text=text,
                ))

        caption_queries = (
            select(Caption.caption_id, Caption.caption_text, Frame.video_id)
            .join(Frame, Caption.frame_id == Frame.frame_id)
            .where(Caption.frame_id.is_not(None)),
            select(Caption.caption_id, Caption.caption_text, Shot.video_id)
            .join(Shot, Caption.shot_id == Shot.shot_id)
            .where(Caption.shot_id.is_not(None)),
            select(Caption.caption_id, Caption.caption_text, Shot.video_id)
            .join(ClipWindow, Caption.clip_id == ClipWindow.clip_id)
            .join(Shot, ClipWindow.shot_id == Shot.shot_id)
            .where(Caption.clip_id.is_not(None)),
        )
        seen_caption_ids: set[int] = set()
        for statement in caption_queries:
            for caption_id, caption_text, video_id in session.execute(statement):
                if caption_id in seen_caption_ids or not caption_text.strip():
                    continue
                seen_caption_ids.add(caption_id)
                documents.append(CorpusDocument(
                    document_id=f"caption:{caption_id}",
                    video_id=video_id,
                    text=caption_text,
                ))

    stats = CorpusStats.from_documents(documents)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats.save(output)
    manifest = {
        "schema_version": stats.schema_version,
        "document_count": stats.document_count,
        "video_count": stats.video_count,
        "source": "video_metadata_and_visual_captions",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
