from __future__ import annotations

from BackEnd.app.retrieval_v2.contracts import MomentBand, VideoHypothesis


def allocate_submission_bands(
    bands: list[MomentBand],
    hypotheses: list[VideoHypothesis],
    *,
    limit: int = 100,
) -> list[MomentBand]:
    """Allocate result slots according to measured video-level uncertainty."""

    ordered = sorted(bands, key=lambda band: (-band.score, band.video_id, band.start_ms))
    if not ordered or not hypotheses:
        return ordered[:limit]

    top = hypotheses[0]
    if top.video_confidence >= 0.8:
        focused = [band for band in ordered if band.video_id == top.video_id]
        rescue = [band for band in ordered if band.video_id != top.video_id]
        return (focused + rescue)[:limit]

    by_video: dict[str, list[MomentBand]] = {}
    for band in ordered:
        by_video.setdefault(band.video_id, []).append(band)
    video_order = [hypothesis.video_id for hypothesis in hypotheses if hypothesis.video_id in by_video]
    video_order.extend(video_id for video_id in by_video if video_id not in video_order)

    allocated: list[MomentBand] = []
    depth = 0
    while len(allocated) < limit:
        added = False
        for video_id in video_order:
            candidates = by_video[video_id]
            if depth < len(candidates):
                allocated.append(candidates[depth])
                added = True
                if len(allocated) == limit:
                    break
        if not added:
            break
        depth += 1
    return allocated


__all__ = ["allocate_submission_bands"]
