"""Query-aware selection of official submission frames from moment bands."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import (
    CandidateReview,
    MomentBand,
    QueryAtom,
)


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """An official frame returned by the corpus-specific provider."""

    video_id: str
    frame_idx: int
    timestamp_ms: int
    img_url: str | None
    frame_id: str | None = None
    is_official: bool = True
    exists: bool = True


@dataclass(frozen=True, slots=True)
class SelectedOfficialFrame:
    """A ranked official frame with inspectable score provenance."""

    frame: FrameCandidate
    score: float
    source_band_id: str
    score_components: Mapping[str, float] = field(default_factory=dict)


class OfficialFrameProvider(Protocol):
    """Resolves corpus-owned official frames within an inclusive time band."""

    def get_official_frames(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> Sequence[FrameCandidate]: ...


class QueryAwareOfficialFrameSelector:
    """Ranks official frames by local evidence while preserving uncertainty."""

    def __init__(
        self,
        *,
        evidence_window_ms: int = 2_000,
        confident_threshold: float = 0.75,
    ) -> None:
        if evidence_window_ms <= 0:
            raise ValueError("evidence_window_ms must be positive")
        if not 0.0 <= confident_threshold <= 1.0:
            raise ValueError("confident_threshold must be between 0 and 1")
        self.evidence_window_ms = evidence_window_ms
        self.confident_threshold = confident_threshold

    def select(
        self,
        *,
        bands: Sequence[MomentBand],
        atoms: Sequence[QueryAtom],
        provider: OfficialFrameProvider,
        reviews: Sequence[CandidateReview] = (),
        moment_confidence: Mapping[str, float] | None = None,
        limit: int = 100,
    ) -> list[SelectedOfficialFrame]:
        """Return at most 100 unique, existing official frames."""

        effective_limit = min(max(limit, 0), 100)
        if effective_limit == 0 or not bands:
            return []

        confidences = moment_confidence or {}
        type_multiplier = {
            "ACTION": 1.30,
            "ATTRIBUTE": 1.15,
            "COUNT": 1.15,
            "RELATION": 1.15,
        }
        atom_weights = {
            atom.atom_id: atom.discriminative_weight * type_multiplier.get(atom.atom_type, 1.0)
            for atom in atoms
            if atom.operator != "MUST_NOT"
        }
        reviews_by_band = {review.band_id: review for review in reviews}
        pools: list[list[SelectedOfficialFrame]] = []

        prefetch = getattr(provider, "prefetch", None)
        if callable(prefetch):
            prefetch(bands)

        for band in bands:
            candidates = provider.get_official_frames(
                band.video_id,
                band.start_ms,
                band.end_ms,
            )
            unique_frames = self._valid_unique_frames(candidates, band.video_id)
            review = reviews_by_band.get(band.band_id)
            ranked = [
                self._score_frame(frame, band, atom_weights, review)
                for frame in unique_frames
            ]
            ranked.sort(key=lambda item: self._ranking_key(item, band))
            if ranked:
                pools.append(ranked)

        if not pools:
            return []

        highest_confidence = max(
            (confidences.get(pool[0].source_band_id, 0.0) for pool in pools),
            default=0.0,
        )
        if highest_confidence >= self.confident_threshold:
            ordered = sorted(
                (item for pool in pools for item in pool),
                key=lambda item: (
                    -item.score,
                    item.frame.video_id,
                    item.frame.frame_idx,
                ),
            )
        else:
            ordered = self._diversity_first(pools)

        selected: list[SelectedOfficialFrame] = []
        seen: set[tuple[str, int]] = set()
        for item in ordered:
            key = (item.frame.video_id, item.frame.frame_idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) == effective_limit:
                break
        return selected

    @staticmethod
    def _valid_unique_frames(
        candidates: Sequence[FrameCandidate],
        expected_video_id: str,
    ) -> list[FrameCandidate]:
        valid: dict[tuple[str, int], FrameCandidate] = {}
        for frame in candidates:
            key = (frame.video_id, frame.frame_idx)
            if (
                frame.video_id == expected_video_id
                and frame.frame_idx >= 0
                and frame.timestamp_ms >= 0
                and frame.is_official
                and frame.exists
                and bool(frame.img_url)
            ):
                valid.setdefault(key, frame)
        return list(valid.values())

    def _score_frame(
        self,
        frame: FrameCandidate,
        band: MomentBand,
        atom_weights: Mapping[str, float],
        review: CandidateReview | None,
    ) -> SelectedOfficialFrame:
        evidence_by_family: dict[tuple[str, str], float] = {}
        for hit in band.hits:
            if hit.atom_id not in atom_weights:
                continue
            key = (hit.atom_id or "", hit.retriever_family or hit.source)
            contribution = self._hit_evidence(hit, frame.timestamp_ms, atom_weights)
            evidence_by_family[key] = max(
                evidence_by_family.get(key, 0.0),
                contribution,
            )
        evidence = sum(evidence_by_family.values())
        verification = self._verification_score(band, atom_weights, review)
        anchor_distance = abs(frame.timestamp_ms - band.peak_ms)
        anchor = 0.05 * math.exp(-anchor_distance / self.evidence_window_ms)
        band_prior = 0.10 * band.score
        components = {
            "atom_evidence": evidence,
            "verification": verification,
            "temporal_anchor": anchor,
            "band_prior": band_prior,
        }
        return SelectedOfficialFrame(
            frame=frame,
            score=sum(components.values()),
            source_band_id=band.band_id,
            score_components=components,
        )

    def _hit_evidence(
        self,
        hit: SearchHit,
        timestamp_ms: int,
        atom_weights: Mapping[str, float],
    ) -> float:
        distance = self._distance_to_hit(hit, timestamp_ms)
        proximity = math.exp(-distance / self.evidence_window_ms)
        raw_score = max(0.0, min(float(hit.raw_score), 1.0))
        rank_score = 1.0 / math.sqrt(hit.rank)
        atom_weight = atom_weights.get(hit.atom_id or "", 0.0)
        return atom_weight * proximity * (0.75 * raw_score + 0.25 * rank_score)

    @staticmethod
    def _distance_to_hit(hit: SearchHit, timestamp_ms: int) -> int:
        if hit.start_ms <= timestamp_ms <= hit.end_ms:
            return 0
        return min(abs(timestamp_ms - hit.start_ms), abs(timestamp_ms - hit.end_ms))

    @staticmethod
    def _verification_score(
        band: MomentBand,
        atom_weights: Mapping[str, float],
        review: CandidateReview | None,
    ) -> float:
        total = 0.0
        weight_sum = 0.0
        for atom_id, cell in band.coverage.items():
            weight = atom_weights.get(atom_id, 0.5)
            weight_sum += weight
            if cell.status == "PASS":
                total += weight
            elif cell.status == "FAIL":
                total -= weight

        if review is not None:
            for atom_id, status in review.atom_status.items():
                weight = atom_weights.get(atom_id, 0.5)
                weight_sum += weight
                if status == "PASS":
                    total += weight
                elif status == "FAIL":
                    total -= weight
            verdict_factor = {
                "match": 1.0,
                "partial": 0.35,
                "uncertain": 0.0,
                "mismatch": -1.0,
            }[review.verdict]
            total += verdict_factor * review.confidence
            weight_sum += 1.0

        if weight_sum == 0:
            return 0.0
        return 0.35 * total / weight_sum

    @staticmethod
    def _ranking_key(
        item: SelectedOfficialFrame,
        band: MomentBand,
    ) -> tuple[float, int, int, int]:
        distance = abs(item.frame.timestamp_ms - band.peak_ms)
        side = 0 if item.frame.timestamp_ms <= band.peak_ms else 1
        return (-item.score, distance, side, item.frame.frame_idx)

    @staticmethod
    def _diversity_first(
        pools: Sequence[Sequence[SelectedOfficialFrame]],
    ) -> list[SelectedOfficialFrame]:
        ranked_pools = sorted(pools, key=lambda pool: -pool[0].score)
        ordered: list[SelectedOfficialFrame] = []
        rescued_videos: set[str] = set()

        for pool in ranked_pools:
            video_id = pool[0].frame.video_id
            if video_id not in rescued_videos:
                ordered.append(pool[0])
                rescued_videos.add(video_id)

        depth = 0
        while True:
            emitted = False
            for pool in ranked_pools:
                if depth < len(pool):
                    item = pool[depth]
                    if item not in ordered:
                        ordered.append(item)
                    emitted = True
            if not emitted:
                break
            depth += 1
        return ordered


__all__ = [
    "FrameCandidate",
    "OfficialFrameProvider",
    "QueryAwareOfficialFrameSelector",
    "SelectedOfficialFrame",
]
