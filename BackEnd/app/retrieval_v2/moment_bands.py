from __future__ import annotations

import hashlib

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import CoverageCell, MomentBand


# Pipeline.md §6C: "Shot trên 30 giây phải chia"
_SHOT_MAX_EXTENT_MS = 30_000


def _band_id(video_id: str, event_id: str | None, start_ms: int, end_ms: int) -> str:
    raw = f"{video_id}|{event_id or '-'}|{start_ms // 250}|{end_ms // 250}"
    return f"mb_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _refine_long_hits(hits: list[SearchHit], max_extent_ms: int) -> list[SearchHit]:
    """Prefer child frame/clip evidence over an imprecise long shot.

    A shot midpoint is not semantic evidence. When no child hit exists the full
    shot remains a broad hypothesis so later local retrieval can refine it.
    """

    refined: list[SearchHit] = []
    for hit in hits:
        if hit.end_ms - hit.start_ms <= max_extent_ms:
            refined.append(hit)
            continue
        has_child = any(
            other is not hit
            and other.video_id == hit.video_id
            and other.event_id == hit.event_id
            and other.atom_id == hit.atom_id
            and other.retriever_family == hit.retriever_family
            and other.entity_type in {"frame", "clip"}
            and other.start_ms <= hit.end_ms
            and other.end_ms >= hit.start_ms
            for other in hits
        )
        if not has_child:
            refined.append(hit)
    return refined


def _to_band(
    hits: list[SearchHit],
    required_atom_ids: list[str],
    negative_atom_ids: list[str],
) -> MomentBand:
    start_ms = min(hit.start_ms for hit in hits)
    end_ms = max(hit.end_ms for hit in hits)
    best_hit = min(hits, key=lambda hit: (hit.rank, -hit.raw_score))
    def build_cell(atom_id: str) -> CoverageCell:
        atom_hits = [hit for hit in hits if hit.atom_id == atom_id]
        families = list(dict.fromkeys(
            hit.retriever_family for hit in atom_hits
            if hit.retriever_family is not None
        ))
        family_scores: dict[str, float] = {}
        for hit in atom_hits:
            family = hit.retriever_family or hit.source
            contribution = 1.0 / (60.0 + hit.rank)
            family_scores[family] = max(family_scores.get(family, 0.0), contribution)
        base_score = sum(family_scores.values())
        independent_bonus = base_score * 0.1 * max(0, len(family_scores) - 1)
        deterministic_pass = any(
            hit.retriever_family in {"object_search", "track_search"}
            and hit.raw_score >= 0.5
            for hit in atom_hits
        )
        return CoverageCell(
            atom_id=atom_id,
            retrieval_status="RETRIEVED" if atom_hits else "MISSING",
            status="PASS" if deterministic_pass else "UNKNOWN",
            score=base_score + independent_bonus,
            evidence_ids=list(dict.fromkeys(hit.entity_id for hit in atom_hits)),
            retriever_families=families,
            prompt_roles=list(dict.fromkeys(
                hit.prompt_role for hit in atom_hits if hit.prompt_role is not None
            )),
            family_scores=family_scores,
        )
    coverage = {atom_id: build_cell(atom_id) for atom_id in required_atom_ids}
    contradictions = {atom_id: build_cell(atom_id) for atom_id in negative_atom_ids}
    score = sum(cell.score for cell in coverage.values())
    return MomentBand(
        band_id=_band_id(hits[0].video_id, hits[0].event_id, start_ms, end_ms),
        video_id=hits[0].video_id,
        event_id=hits[0].event_id,
        start_ms=start_ms,
        end_ms=end_ms,
        peak_ms=(best_hit.start_ms + best_hit.end_ms) // 2,
        coverage=coverage,
        contradictions=contradictions,
        hits=hits,
        score=score,
        score_breakdown={
            "positive_retrieval_support": score,
            "negative_retrieval_support": sum(
                cell.score for cell in contradictions.values()
            ),
        },
    )


def build_moment_bands(
    hits: list[SearchHit],
    *,
    required_atom_ids: list[str],
    negative_atom_ids: list[str] | None = None,
    merge_gap_ms: int = 1_000,
    max_duration_ms: int = 15_000,
    shot_max_extent_ms: int = _SHOT_MAX_EXTENT_MS,
) -> list[MomentBand]:
    """Group hits while using child evidence to refine imprecise long shots."""

    clamped = _refine_long_hits(hits, shot_max_extent_ms)

    groups: dict[tuple[str, str | None], list[SearchHit]] = {}
    for hit in clamped:
        groups.setdefault((hit.video_id, hit.event_id), []).append(hit)

    bands: list[MomentBand] = []
    for group_hits in groups.values():
        ordered = sorted(group_hits, key=lambda hit: (hit.start_ms, hit.end_ms))
        cluster: list[SearchHit] = []
        cluster_end = 0
        cluster_start = 0
        for hit in ordered:
            would_exceed = (
                cluster
                and hit.start_ms > cluster_start
                and hit.end_ms - cluster_start > max_duration_ms
            )
            if cluster and (hit.start_ms > cluster_end + merge_gap_ms or would_exceed):
                bands.append(_to_band(cluster, required_atom_ids, negative_atom_ids or []))
                cluster = []
            if not cluster:
                cluster_start = hit.start_ms
                cluster_end = hit.end_ms
            cluster.append(hit)
            cluster_end = max(cluster_end, hit.end_ms)
        if cluster:
            bands.append(_to_band(cluster, required_atom_ids, negative_atom_ids or []))

    # OCR/ASR constraints are query-level in the public contract, but their
    # timestamps can still ground a particular KIS/TRAKE event. Join unscoped
    # evidence into overlapping event bands instead of leaving it unusable in
    # a separate event_id=None group.
    event_bands = [band for band in bands if band.event_id is not None]
    unscoped = [band for band in bands if band.event_id is None]
    if event_bands and unscoped:
        joined: list[MomentBand] = []
        consumed_unscoped: set[str] = set()
        for event_band in event_bands:
            nearby = [
                band
                for band in unscoped
                if band.video_id == event_band.video_id
                and band.start_ms <= event_band.end_ms + merge_gap_ms
                and band.end_ms >= event_band.start_ms - merge_gap_ms
            ]
            if nearby:
                consumed_unscoped.update(band.band_id for band in nearby)
                joined_hits = [
                    hit.model_copy(update={"event_id": event_band.event_id})
                    for band in [event_band, *nearby]
                    for hit in band.hits
                ]
                joined.append(
                    _to_band(
                        joined_hits,
                        required_atom_ids,
                        negative_atom_ids or [],
                    )
                )
            else:
                joined.append(event_band)
        bands = [
            *joined,
            *(band for band in unscoped if band.band_id not in consumed_unscoped),
        ]

    return sorted(bands, key=lambda band: (-band.score, band.video_id, band.start_ms))


__all__ = ["build_moment_bands"]
