from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.retrieval_v2.contracts import CoverageCell, MomentBand


def _collapse(sequence: list[MomentBand]) -> MomentBand:
    coverage: dict[str, CoverageCell] = {}
    contradictions: dict[str, CoverageCell] = {}
    hits = []
    for band in sequence:
        hits.extend(band.hits)
        for atom_id, cell in band.coverage.items():
            current = coverage.get(atom_id)
            if current is None or cell.score > current.score:
                coverage[atom_id] = cell
        for atom_id, cell in band.contradictions.items():
            current = contradictions.get(atom_id)
            if current is None or cell.score > current.score:
                contradictions[atom_id] = cell
    raw_id = "|".join(band.band_id for band in sequence)
    best = max(sequence, key=lambda band: band.score)
    duration_ms = max(0, sequence[-1].end_ms - sequence[0].start_ms)
    compactness = math.exp(-duration_ms / 120_000.0)
    sequence_score = sum(band.score for band in sequence) * (0.35 + 0.65 * compactness)
    return MomentBand(
        band_id=f"seq_{hashlib.sha1(raw_id.encode('utf-8')).hexdigest()[:12]}",
        video_id=sequence[0].video_id,
        start_ms=sequence[0].start_ms,
        end_ms=sequence[-1].end_ms,
        peak_ms=best.peak_ms,
        coverage=coverage,
        contradictions=contradictions,
        hits=hits,
        score=sequence_score,
        score_breakdown={
            "event_score_sum": sum(band.score for band in sequence),
            "sequence_compactness": compactness,
        },
    )


def collapse_kis_sequences(
    bands: list[MomentBand],
    event_ids: list[str],
    *,
    temporal_constraints: Sequence[TemporalConstraint] = (),
    limit: int = 100,
    beam_size: int = 50,
) -> list[MomentBand]:
    """Create same-video, increasing-time KIS sequence bands."""

    if len(event_ids) < 2:
        return bands[:limit]
    by_video: dict[str, dict[str, list[MomentBand]]] = {}
    for band in bands:
        if band.event_id in event_ids:
            by_video.setdefault(band.video_id, {}).setdefault(band.event_id, []).append(band)

    collapsed: list[MomentBand] = []
    for event_bands in by_video.values():
        if any(not event_bands.get(event_id) for event_id in event_ids):
            continue
        beam = [[band] for band in event_bands[event_ids[0]]]
        for event_id in event_ids[1:]:
            expanded = [
                [*sequence, candidate]
                for sequence in beam
                for candidate in event_bands[event_id]
                if _can_append(sequence, candidate, temporal_constraints)
            ]
            beam = sorted(
                expanded,
                key=lambda sequence: -sum(band.score for band in sequence),
            )[:beam_size]
            if not beam:
                break
        collapsed.extend(_collapse(sequence) for sequence in beam if len(sequence) == len(event_ids))

    return sorted(collapsed, key=lambda band: (-band.score, band.video_id, band.start_ms))[:limit]


def _can_append(
    sequence: list[MomentBand],
    candidate: MomentBand,
    constraints: Sequence[TemporalConstraint],
) -> bool:
    if candidate.start_ms < sequence[-1].end_ms:
        relevant_last = next(
            (
                constraint
                for constraint in constraints
                if constraint.before == sequence[-1].event_id
                and constraint.after == candidate.event_id
            ),
            None,
        )
        if relevant_last is None or not relevant_last.allow_overlap:
            return False

    by_event = {band.event_id: band for band in sequence if band.event_id is not None}
    by_event[candidate.event_id] = candidate
    for constraint in constraints:
        before = by_event.get(constraint.before)
        after = by_event.get(constraint.after)
        if before is None or after is None:
            continue
        gap_ms = after.start_ms - before.end_ms
        if gap_ms < 0 and not constraint.allow_overlap:
            return False
        if constraint.min_gap_ms is not None and gap_ms < constraint.min_gap_ms:
            return False
        if constraint.max_gap_ms is not None and gap_ms > constraint.max_gap_ms:
            return False
    return True


__all__ = ["collapse_kis_sequences"]
