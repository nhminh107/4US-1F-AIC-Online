"""Task 12: Candidate reranker adapter protocol and fallback implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import numpy as np

from BackEnd.app.retrieval_v2.contracts import MomentBand, QueryAtom
from BackEnd.app.retrieval_v2.frame_selector import OfficialFrameProvider


class CandidateReranker(Protocol):
    """Protocol for candidate rerankers (e.g. cross-encoders, visual-text fine-grained matchers)."""

    async def rerank(
        self,
        bands: list[MomentBand],
        atoms: list[QueryAtom],
        limit: int,
    ) -> list[MomentBand]:
        """Rerank moment bands using fine-grained model scoring."""
        ...


class DeterministicFallbackReranker:
    """Fallback reranker when no neural reranker model is loaded."""

    async def rerank(
        self,
        bands: list[MomentBand],
        atoms: list[QueryAtom],
        limit: int,
    ) -> list[MomentBand]:
        # Stable deterministic sort by band score and timestamp
        return sorted(bands, key=lambda b: (-b.score, b.video_id, b.start_ms))[:limit]


class ClipOfficialFrameReranker:
    """Bounded atom-wise reranking over real official frames in each band."""

    def __init__(self, provider: OfficialFrameProvider, embedder, *, frames_per_band: int = 3) -> None:
        self.provider = provider
        self.embedder = embedder
        self.frames_per_band = max(1, frames_per_band)

    async def rerank(
        self,
        bands: list[MomentBand],
        atoms: list[QueryAtom],
        limit: int,
    ) -> list[MomentBand]:
        return await asyncio.to_thread(self._rerank_sync, bands, atoms, limit)

    def _rerank_sync(
        self,
        bands: list[MomentBand],
        atoms: list[QueryAtom],
        limit: int,
    ) -> list[MomentBand]:
        visual_atoms = [
            atom for atom in atoms if atom.modality == "visual" and atom.operator != "MUST_NOT"
        ]
        if not visual_atoms:
            return bands[:limit]
        texts = [self._atom_prompt(atom) for atom in visual_atoms]
        text_vectors = self._normalize_rows(np.asarray(self.embedder.encode_texts(texts)))
        rescored: list[MomentBand] = []
        for band in bands:
            frames = list(self.provider.get_official_frames(
                band.video_id,
                band.start_ms,
                band.end_ms,
            ))
            sampled = self._sample_frames(frames, band.peak_ms)
            paths = [Path(frame.img_url) for frame in sampled if frame.img_url and Path(frame.img_url).is_file()]
            if not paths:
                rescored.append(band)
                continue
            image_vectors = self._normalize_rows(np.stack([
                np.asarray(self.embedder.encode_image(str(path))) for path in paths
            ]))
            similarities = image_vectors @ text_vectors.T
            atom_scores = (np.max(similarities, axis=0) + 1.0) / 2.0
            weights = np.asarray([atom.discriminative_weight for atom in visual_atoms], dtype=float)
            fine_score = float(np.average(atom_scores, weights=np.maximum(weights, 0.01)))
            fused_score = 0.65 * band.score + 0.35 * fine_score
            rescored.append(band.model_copy(update={
                "score": max(0.0, fused_score),
                "score_breakdown": {
                    **band.score_breakdown,
                    "fine_grained_visual": fine_score,
                },
            }))
        return sorted(rescored, key=lambda band: (-band.score, band.video_id, band.start_ms))[:limit]

    def _sample_frames(self, frames, peak_ms: int):
        ordered = sorted(frames, key=lambda frame: (frame.timestamp_ms, frame.frame_idx))
        if len(ordered) <= self.frames_per_band:
            return ordered
        candidates = [ordered[0], min(ordered, key=lambda frame: abs(frame.timestamp_ms - peak_ms)), ordered[-1]]
        unique = list(dict.fromkeys((frame.video_id, frame.frame_idx) for frame in candidates))
        by_key = {(frame.video_id, frame.frame_idx): frame for frame in candidates}
        return [by_key[key] for key in unique[: self.frames_per_band]]

    @staticmethod
    def _atom_prompt(atom: QueryAtom) -> str:
        role_order = ("action", "rare_detail", "contrast", "global", "context")
        by_role = {prompt.role: prompt.text for prompt in atom.prompt_variants}
        return next((by_role[role] for role in role_order if role in by_role), atom.text)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = vectors.reshape(1, -1) if vectors.ndim == 1 else vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


__all__ = [
    "CandidateReranker",
    "ClipOfficialFrameReranker",
    "DeterministicFallbackReranker",
]
