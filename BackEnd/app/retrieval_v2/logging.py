"""Task 13: Structured audit logger for V2 search controller."""

from __future__ import annotations

import logging
from typing import Any

from BackEnd.app.retrieval_v2.contracts import SearchControllerResult

logger = logging.getLogger("retrieval_v2.audit")


def build_search_audit_log(result: SearchControllerResult) -> dict[str, Any]:
    """Create a structured, sanitised audit dictionary for a search run."""
    return {
        "query_id": result.plan.query_id,
        "task": result.plan.task,
        "profile": result.plan.execution_profile,
        "atom_count": len(result.plan.atoms),
        "atom_ids": [atom.atom_id for atom in result.plan.atoms],
        "modalities": list(dict.fromkeys(atom.modality for atom in result.plan.atoms)),
        "raw_hits": result.session.raw_hit_count,
        "dedup_hits": result.session.deduplicated_hit_count,
        "bands_count": len(result.bands),
        "top_video_ids": [h.video_id for h in result.hypotheses[:5]],
        "diagnoses": [
            {
                "reason": item.reason,
                "action": item.action,
                "atom_id": item.atom_id,
                "video_id": item.video_id,
                "band_id": item.band_id,
            }
            for item in result.session.diagnoses
        ],
        "reviews": [
            {
                "band_id": item.band_id,
                "verdict": item.verdict,
                "confidence": item.confidence,
                "failure_reason": item.failure_reason,
            }
            for item in result.session.reviews
        ],
        "top_bands": [
            {
                "band_id": band.band_id,
                "video_id": band.video_id,
                "start_ms": band.start_ms,
                "end_ms": band.end_ms,
                "score": round(band.score, 6),
                "coverage": {
                    atom_id: {
                        "retrieval": cell.retrieval_status,
                        "semantic": cell.status,
                        "families": cell.retriever_families,
                    }
                    for atom_id, cell in band.coverage.items()
                },
                "gates": [
                    {
                        "constraint_id": decision.constraint_id,
                        "status": decision.status,
                        "reason": decision.reason_code,
                    }
                    for decision in band.constraint_decisions
                ],
            }
            for band in result.reranked_bands[:7]
        ],
        "rounds": [
            {
                "phase": round_.phase,
                "requested_k": round_.requested_k,
                "hits": round_.hit_count,
                "unique": round_.unique_candidate_count,
                "new_videos": round_.new_video_gain,
                "new_moments": round_.new_moment_gain,
            }
            for round_ in result.session.rounds
        ],
        "rounds_count": len(result.session.rounds),
        "stop_reason": result.session.stop_reason,
    }


def emit_audit_log(result: SearchControllerResult) -> None:
    """Log audit record via standard python logging."""
    payload = build_search_audit_log(result)
    logger.info("SEARCH_AUDIT: %s", payload)


__all__ = ["build_search_audit_log", "emit_audit_log"]
