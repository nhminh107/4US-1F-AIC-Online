from __future__ import annotations

import math

from BackEnd.app.retrieval_v2.contracts import (
    CoverageCell,
    MomentBand,
    QueryAtom,
    RetryDiagnosis,
    ScoringProfile,
    VideoHypothesis,
)


_DEFAULT_PROFILE = ScoringProfile()


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean; returns 0.0 for empty input."""
    if not values:
        return 0.0
    product = math.prod(values)
    return product ** (1.0 / len(values))


def rerank_bands(
    bands: list[MomentBand],
    atoms: list[QueryAtom],
    limit: int,
    *,
    profile: ScoringProfile | None = None,
) -> list[MomentBand]:
    """Rank bands using geometric mean for required atoms + penalties.

    This implements pipeline.md §6F:
      S_required = geometric_mean(ε + S_atom for required atoms)
      S_support  = weighted_sum(S_atom for supporting atoms)
      S_candidate = W_req * S_req + W_sup * S_sup - missing_penalty - negative_penalty
    """
    sp = profile or _DEFAULT_PROFILE
    required_atoms = [
        a for a in atoms if a.role == "REQUIRED" and a.operator == "MUST"
    ]
    supporting_atoms = [
        a for a in atoms if a.role == "SUPPORTING" and a.operator != "MUST_NOT"
    ]
    atom_weights = {a.atom_id: a.discriminative_weight for a in atoms}

    rescored: list[MomentBand] = []
    for band in bands:
        # Required: geometric mean of (epsilon + weighted cell score)
        required_scores = [
            sp.epsilon + band.coverage.get(a.atom_id, CoverageCell(
                atom_id=a.atom_id, status="UNKNOWN", score=0.0,
            )).score * atom_weights.get(a.atom_id, 1.0)
            for a in required_atoms
        ]
        s_required = _geometric_mean(required_scores) if required_scores else 0.0

        # Supporting: weighted sum
        s_supporting = sum(
            band.coverage.get(a.atom_id, CoverageCell(
                atom_id=a.atom_id, status="UNKNOWN", score=0.0,
            )).score * atom_weights.get(a.atom_id, 1.0)
            for a in supporting_atoms
        )

        # Missing means retrieval did not support the atom. Retrieved but not
        # visually verified remains UNKNOWN without being treated as absent.
        missing_count = sum(
            1 for a in required_atoms
            if band.coverage.get(a.atom_id, CoverageCell(
                atom_id=a.atom_id, status="UNKNOWN", score=0.0,
            )).retrieval_status == "MISSING"
        )
        missing_pen = sp.missing_penalty_per_atom * missing_count

        # Negative penalty: required atoms that are FAIL
        negative_count = sum(
            1 for a in required_atoms
            if band.coverage.get(a.atom_id, CoverageCell(
                atom_id=a.atom_id, status="UNKNOWN", score=0.0,
            )).status == "FAIL"
        )
        negative_pen = sp.negative_penalty * negative_count

        base_score = (
            sp.W_required * s_required
            + sp.W_supporting * s_supporting
            - missing_pen
            - negative_pen
        )
        sequence_compactness = band.score_breakdown.get("sequence_compactness")
        temporal_score = 0.0
        if sequence_compactness is not None:
            temporal_score = max(0.0, min(1.0, sequence_compactness))
            base_score *= 0.35 + 0.65 * temporal_score
        score = max(0.0, base_score + sp.W_temporal * temporal_score)
        rescored.append(band.model_copy(update={"score": score}))

    return sorted(rescored, key=lambda b: (-b.score, b.video_id, b.start_ms))[:limit]


def build_video_hypotheses(
    bands: list[MomentBand],
    atoms: list[QueryAtom],
    limit: int,
    *,
    profile: ScoringProfile | None = None,
) -> list[VideoHypothesis]:
    sp = profile or _DEFAULT_PROFILE
    grouped: dict[str, list[MomentBand]] = {}
    for band in bands:
        grouped.setdefault(band.video_id, []).append(band)

    positive_atoms = [atom for atom in atoms if atom.operator != "MUST_NOT"]
    total_weight = sum(a.discriminative_weight for a in positive_atoms) or 1.0
    hypotheses: list[VideoHypothesis] = []
    for video_id, video_bands in grouped.items():
        coverage: dict[str, CoverageCell] = {}
        covered_weight = 0.0
        for atom in positive_atoms:
            cells = [
                band.coverage[atom.atom_id]
                for band in video_bands
                if atom.atom_id in band.coverage
            ]
            evidence_ids = list(dict.fromkeys(
                eid for cell in cells for eid in cell.evidence_ids
            ))
            # Family-aware: count distinct retriever families, not raw hits (W4)
            all_families = list(dict.fromkeys(
                fam for cell in cells for fam in cell.retriever_families
            ))
            best_score = max((cell.score for cell in cells), default=0.0)
            retrieval_status = "RETRIEVED" if evidence_ids else "MISSING"
            status = "UNKNOWN"
            # Small consensus bonus only when >1 distinct family contributes
            family_bonus = min(0.1, 0.05 * max(0, len(set(all_families)) - 1))
            coverage[atom.atom_id] = CoverageCell(
                atom_id=atom.atom_id,
                retrieval_status=retrieval_status,
                status=status,
                score=best_score + family_bonus,
                evidence_ids=evidence_ids,
                retriever_families=all_families,
            )
            if retrieval_status == "RETRIEVED":
                covered_weight += atom.discriminative_weight

        coverage_ratio = covered_weight / total_weight
        best_band = max(video_bands, key=lambda b: b.score)
        # max possible RRF score across all prompt variants of all atoms
        max_rrf = sum(
            sum(p.weight * a.discriminative_weight / 61.0 for p in a.prompt_variants) or (a.discriminative_weight / 61.0)
            for a in positive_atoms
        ) or 1.0
        raw_strength = max(0.0, best_band.score / max_rrf)
        strength = raw_strength / (1.0 + raw_strength)
        consistency = 1.0 / (1.0 + max(0, len(video_bands) - 1) * 0.02)
        anchor_atoms = [atom for atom in positive_atoms if atom.scope == "VIDEO_ANCHOR"]
        anchor_weight = sum(atom.discriminative_weight for atom in anchor_atoms)
        anchor_coverage = (
            sum(
                atom.discriminative_weight
                for atom in anchor_atoms
                if coverage[atom.atom_id].retrieval_status == "RETRIEVED"
            ) / anchor_weight
            if anchor_weight
            else coverage_ratio
        )
        video_confidence = min(
            1.0,
            0.55 * coverage_ratio
            + 0.20 * anchor_coverage
            + 0.15 * strength
            + 0.10 * consistency,
        )
        moment_atoms = [atom for atom in positive_atoms if atom.scope != "VIDEO_ANCHOR"]
        event_keys = list(dict.fromkeys(atom.event_id or "__moment__" for atom in moment_atoms))
        localized_weight = 0.0
        localized_support = 0.0
        for event_key in event_keys:
            event_atoms = [
                atom for atom in moment_atoms if (atom.event_id or "__moment__") == event_key
            ]
            event_weight = sum(atom.discriminative_weight for atom in event_atoms)
            localized_weight += event_weight
            best_event_coverage = max(
                (
                    sum(
                        atom.discriminative_weight
                        for atom in event_atoms
                        if atom.atom_id in band.coverage
                        and band.coverage[atom.atom_id].retrieval_status == "RETRIEVED"
                    ) / event_weight
                    for band in video_bands
                    if event_key == "__moment__" or band.event_id in {None, event_key}
                ),
                default=0.0,
            )
            localized_support += best_event_coverage * event_weight
        moment_coverage = localized_support / localized_weight if localized_weight else coverage_ratio
        moment_confidence = min(1.0, 0.80 * moment_coverage + 0.20 * strength)
        hypotheses.append(
            VideoHypothesis(
                video_id=video_id,
                video_confidence=round(video_confidence, 6),
                moment_confidence=round(moment_confidence, 6),
                coverage=coverage,
                band_ids=[band.band_id for band in video_bands],
                lane_sources=["moment"],
            )
        )

    return sorted(
        hypotheses,
        key=lambda item: (-item.video_confidence, -item.moment_confidence, item.video_id),
    )[:limit]


def diagnose_hypotheses(
    hypotheses: list[VideoHypothesis],
    atoms: list[QueryAtom],
    *,
    max_hypotheses: int = 5,
) -> list[RetryDiagnosis]:
    if not hypotheses:
        return [RetryDiagnosis(reason="LOW_VIDEO_CONFIDENCE", action="BROADEN_VIDEO_SEARCH")]
    positive_required = [
        atom for atom in atoms if atom.required and atom.operator == "MUST"
    ]
    plausible = [
        hypothesis
        for hypothesis in hypotheses[:max_hypotheses]
        if hypothesis.video_confidence >= 0.45
    ]
    if not plausible:
        return [RetryDiagnosis(reason="LOW_VIDEO_CONFIDENCE", action="BROADEN_VIDEO_SEARCH")]

    diagnoses: list[RetryDiagnosis] = []
    for hypothesis in plausible:
        for atom in positive_required:
            cell = hypothesis.coverage.get(atom.atom_id)
            if cell is None or cell.retrieval_status != "RETRIEVED":
                diagnoses.append(RetryDiagnosis(
                    reason="MISSING_REQUIRED_ATOM",
                    action="RETRY_WEAK_ATOM",
                    atom_id=atom.atom_id,
                    video_id=hypothesis.video_id,
                ))
        if hypothesis.moment_confidence < 0.55:
            diagnoses.append(RetryDiagnosis(
                reason="LOW_MOMENT_CONFIDENCE",
                action="EXPAND_LOCAL_SEARCH",
                video_id=hypothesis.video_id,
            ))
    return diagnoses


__all__ = ["build_video_hypotheses", "diagnose_hypotheses", "rerank_bands"]
