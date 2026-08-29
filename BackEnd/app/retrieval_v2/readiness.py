from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from BackEnd.app.retrieval_v2.corpus_stats import CorpusStats
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex


@dataclass(frozen=True)
class RetrievalArtifactReadiness:
    corpus_stats_ready: bool
    video_index_ready: bool
    degraded_reasons: tuple[str, ...]


@lru_cache(maxsize=8)
def inspect_retrieval_artifacts(
    corpus_stats_path: str | None,
    video_index_path: str | None,
) -> RetrievalArtifactReadiness:
    reasons: list[str] = []
    corpus_ready = False
    video_ready = False

    if not corpus_stats_path:
        reasons.append("corpus_stats_path_not_configured")
    elif not Path(corpus_stats_path).is_file():
        reasons.append("corpus_stats_missing")
    else:
        try:
            CorpusStats.load(corpus_stats_path)
            corpus_ready = True
        except Exception:
            reasons.append("corpus_stats_invalid")

    if not video_index_path:
        reasons.append("video_index_path_not_configured")
    elif not Path(video_index_path).is_file():
        reasons.append("video_index_missing")
    else:
        try:
            VideoLevelIndex.load_versioned(video_index_path)
            video_ready = True
        except Exception:
            reasons.append("video_index_invalid")

    return RetrievalArtifactReadiness(
        corpus_stats_ready=corpus_ready,
        video_index_ready=video_ready,
        degraded_reasons=tuple(reasons),
    )


__all__ = ["RetrievalArtifactReadiness", "inspect_retrieval_artifacts"]
