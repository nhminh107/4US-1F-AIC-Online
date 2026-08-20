"""Small, explicit configuration surface for the HTTP API."""

from __future__ import annotations

import os


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
    default=False,
)

# Number of human-review images returned for a VQA query.
VQA_MAX_CANDIDATES = int(os.getenv("VQA_MAX_CANDIDATES", "10"))

# Nearby hits within this distance are grouped into one CandidateRegion.
CANDIDATE_MERGE_GAP_MS = int(os.getenv("CANDIDATE_MERGE_GAP_MS", "2000"))

API_PREFIX = "/api/v1"


__all__ = [
    "API_PREFIX",
    "CANDIDATE_MERGE_GAP_MS",
    "SELECTIVE_VERIFIER_ENABLED",
    "VQA_MAX_CANDIDATES",
]
