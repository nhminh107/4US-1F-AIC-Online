"""Task 13: Retrieval metrics and diagnostics collector."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMetrics:
    query_id: str
    task: str
    total_latency_ms: float = 0.0
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    raw_hit_count: int = 0
    dedup_hit_count: int = 0
    moment_band_count: int = 0
    video_hypothesis_count: int = 0
    round_count: int = 0
    retry_reasons: list[str] = field(default_factory=list)
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "stage_latencies_ms": {k: round(v, 2) for k, v in self.stage_latencies_ms.items()},
            "raw_hit_count": self.raw_hit_count,
            "dedup_hit_count": self.dedup_hit_count,
            "moment_band_count": self.moment_band_count,
            "video_hypothesis_count": self.video_hypothesis_count,
            "round_count": self.round_count,
            "retry_reasons": self.retry_reasons,
            "cache_hit": self.cache_hit,
        }


class RetrievalMetricsCollector:
    """Collects and aggregates performance metrics across search sessions."""

    def __init__(self) -> None:
        self.sessions: list[SessionMetrics] = []

    def start_session(self, query_id: str, task: str) -> SessionMetrics:
        metrics = SessionMetrics(query_id=query_id, task=task)
        self.sessions.append(metrics)
        return metrics

    def summary(self) -> dict[str, Any]:
        if not self.sessions:
            return {"total_queries": 0}
        latencies = [s.total_latency_ms for s in self.sessions]
        return {
            "total_queries": len(self.sessions),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "total_raw_hits": sum(s.raw_hit_count for s in self.sessions),
            "total_bands": sum(s.moment_band_count for s in self.sessions),
        }


__all__ = ["RetrievalMetricsCollector", "SessionMetrics"]
