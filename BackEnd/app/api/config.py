"""Small, explicit configuration surface for the HTTP API."""

from __future__ import annotations

import os

from BackEnd.app.retrieval_v2.contracts import CandidateBudget


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw_value!r}.")


# Important switch: restart the API after changing this environment variable.
SELECTIVE_VERIFIER_ENABLED: bool = _read_bool(
    "SELECTIVE_VERIFIER_ENABLED",
    default=True,
)

# V2 is default active for online retrieval pipeline.
RETRIEVAL_V2_ENABLED: bool = _read_bool("RETRIEVAL_V2_ENABLED", default=False)
RETRIEVAL_V2_CORPUS_STATS_PATH = os.getenv("RETRIEVAL_V2_CORPUS_STATS_PATH")
RETRIEVAL_V2_VIDEO_INDEX_PATH = os.getenv("RETRIEVAL_V2_VIDEO_INDEX_PATH")
RETRIEVAL_V2_CANDIDATE_BUDGET = CandidateBudget(
    raw_retrieval_target=int(os.getenv("RETRIEVAL_V2_RAW_TARGET", "3000")),
    raw_retrieval_max=int(os.getenv("RETRIEVAL_V2_RAW_MAX", "5000")),
    unique_candidate_min=int(os.getenv("RETRIEVAL_V2_UNIQUE_MIN", "1600")),
    unique_candidate_max=int(os.getenv("RETRIEVAL_V2_UNIQUE_MAX", "2000")),
    moment_band_limit=int(os.getenv("RETRIEVAL_V2_BAND_LIMIT", "800")),
    video_shortlist_limit=int(os.getenv("RETRIEVAL_V2_VIDEO_LIMIT", "60")),
    local_retrieval_k=int(os.getenv("RETRIEVAL_V2_LOCAL_K", "1500")),
    retry_retrieval_k=int(os.getenv("RETRIEVAL_V2_RETRY_K", "800")),
    rerank_limit=int(os.getenv("RETRIEVAL_V2_RERANK_LIMIT", "100")),
    review_limit=int(os.getenv("RETRIEVAL_V2_REVIEW_LIMIT", "7")),
    max_retry_rounds=int(os.getenv("RETRIEVAL_V2_MAX_RETRIES", "2")),
)

# Number of human-review images returned for a VQA query.
VQA_MAX_CANDIDATES = int(os.getenv("VQA_MAX_CANDIDATES", "100"))

# Nearby hits within this distance are grouped into one CandidateRegion.
CANDIDATE_MERGE_GAP_MS = int(os.getenv("CANDIDATE_MERGE_GAP_MS", "1000"))

API_PREFIX = "/api/v1"


__all__ = [
    "API_PREFIX",
    "CANDIDATE_MERGE_GAP_MS",
    "RETRIEVAL_V2_ENABLED",
    "RETRIEVAL_V2_CANDIDATE_BUDGET",
    "RETRIEVAL_V2_CORPUS_STATS_PATH",
    "RETRIEVAL_V2_VIDEO_INDEX_PATH",
    "SELECTIVE_VERIFIER_ENABLED",
    "VQA_MAX_CANDIDATES",
]
