from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class VideoLevelIndex:
    """Small exact index over diverse representative vectors per video."""

    def __init__(self, video_ids: list[str], vectors: np.ndarray) -> None:
        if len(video_ids) != len(vectors):
            raise ValueError("video_ids and vectors must have the same length")
        self.video_ids = list(video_ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    @classmethod
    def build(
        cls,
        vectors_by_video: dict[str, np.ndarray],
        *,
        representatives_per_video: int = 4,
    ) -> "VideoLevelIndex":
        if representatives_per_video < 1:
            raise ValueError("representatives_per_video must be positive")
        video_ids: list[str] = []
        representatives: list[np.ndarray] = []
        for video_id in sorted(vectors_by_video):
            vectors = np.asarray(vectors_by_video[video_id], dtype=np.float32)
            if vectors.ndim != 2 or not len(vectors):
                raise ValueError(f"Video {video_id!r} must contain a non-empty 2D matrix")
            normalized = np.stack([_normalize(vector) for vector in vectors])
            selected = cls._diverse_representatives(
                normalized,
                representatives_per_video,
            )
            video_ids.extend([video_id] * len(selected))
            representatives.extend(selected)
        return cls(video_ids, np.stack(representatives))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self.video_ids:
            return []
        scores = self.vectors @ _normalize(query_vector)
        best_by_video: dict[str, float] = {}
        for index, video_id in enumerate(self.video_ids):
            best_by_video[video_id] = max(
                best_by_video.get(video_id, -1.0),
                float(scores[index]),
            )
        return sorted(
            best_by_video.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

    @staticmethod
    def _diverse_representatives(
        vectors: np.ndarray,
        limit: int,
    ) -> list[np.ndarray]:
        if len(vectors) <= limit:
            return [vector for vector in vectors]
        centroid = _normalize(vectors.mean(axis=0))
        first = max(
            range(len(vectors)),
            key=lambda index: (1.0 - float(vectors[index] @ centroid), -index),
        )
        selected = [first]
        while len(selected) < limit:
            candidate = max(
                (index for index in range(len(vectors)) if index not in selected),
                key=lambda index: (
                    min(
                        1.0 - float(vectors[index] @ vectors[chosen])
                        for chosen in selected
                    ),
                    -index,
                ),
            )
            selected.append(candidate)
        return [vectors[index] for index in selected]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            video_ids=np.asarray(self.video_ids),
            vectors=self.vectors,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VideoLevelIndex":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["video_ids"].astype(str).tolist(), data["vectors"])

    @classmethod
    def load_versioned(cls, path: str | Path) -> "VideoLevelIndex":
        artifact = Path(path)
        manifest_path = artifact.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Missing video index manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "video-level-index-v1":
            raise ValueError("Unsupported video level index schema version")
        if manifest.get("source_entity") != "shot":
            raise ValueError("Video level index must be built from shot representatives")
        loaded = cls.load(artifact)
        if len(loaded.video_ids) != int(manifest.get("representative_count", -1)):
            raise ValueError("Video index and manifest representative counts differ")
        return loaded


__all__ = ["VideoLevelIndex"]
