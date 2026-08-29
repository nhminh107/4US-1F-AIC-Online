"""Build a versioned diverse-representative video index from shot FAISS vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from BackEnd.app.retrieval.visual_retrieval import build_default_visual_retrieval_tools
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex


SCHEMA_VERSION = "video-level-index-v1"


def build(output: Path, *, representatives: int, batch_size: int) -> dict[str, object]:
    tools = build_default_visual_retrieval_tools()
    index = tools.registry.shot_index
    vectors_by_video: dict[str, list[np.ndarray]] = {}

    for offset in range(0, int(index.ntotal), batch_size):
        faiss_ids = list(range(offset, min(offset + batch_size, int(index.ntotal))))
        resolved = tools.db_mng.get_shot_hits_by_faiss_ids(
            faiss_ids,
            index_version=tools.config.index_version,
            model_name=tools.config.model_name,
            model_version=tools.config.model_version,
            pooling_method=tools.config.pooling_method,
        )
        video_by_id = {item.faiss_id: item.video_id for item in resolved}
        for faiss_id in faiss_ids:
            video_id = video_by_id.get(faiss_id)
            if video_id is None:
                continue
            vectors_by_video.setdefault(video_id, []).append(
                np.asarray(index.reconstruct(faiss_id), dtype=np.float32)
            )

    matrices = {video_id: np.stack(vectors) for video_id, vectors in vectors_by_video.items()}
    video_index = VideoLevelIndex.build(
        matrices,
        representatives_per_video=representatives,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    video_index.save(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "video_count": len(matrices),
        "representative_count": len(video_index.video_ids),
        "representatives_per_video": representatives,
        "source_entity": "shot",
        "source_ntotal": int(index.ntotal),
        "index_version": tools.config.index_version,
        "model_name": tools.config.model_name,
        "model_version": tools.config.model_version,
        "pooling_method": tools.config.pooling_method,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()
    if args.representatives < 1 or args.batch_size < 1:
        parser.error("representatives and batch-size must be positive")
    print(json.dumps(build(
        args.output,
        representatives=args.representatives,
        batch_size=args.batch_size,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
