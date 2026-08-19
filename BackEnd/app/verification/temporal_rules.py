"""Pure deterministic rules for temporal order and gap constraints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from BackEnd.app.contracts.models import TemporalConstraint
from BackEnd.app.verification.enums import ClaimStatus


def evaluate_temporal_order(
    constraint: TemporalConstraint,
    events: Mapping[str, Any],
) -> tuple[ClaimStatus, str]:
    timestamps = _event_timestamps(constraint, events)
    if timestamps is None:
        return ClaimStatus.UNKNOWN, "missing or invalid temporal event"

    before_start_ms, before_end_ms, after_start_ms, after_end_ms = timestamps
    if after_end_ms < before_start_ms:
        return ClaimStatus.CONTRADICTED, "events reversed"

    gap_ms = after_start_ms - before_end_ms
    if gap_ms < 0 and not constraint.allow_overlap:
        return ClaimStatus.CONTRADICTED, "events overlap"
    return ClaimStatus.SUPPORTED, "temporal order passed"


def evaluate_temporal_gap(
    constraint: TemporalConstraint,
    events: Mapping[str, Any],
) -> tuple[ClaimStatus, str]:
    order_status, _ = evaluate_temporal_order(constraint, events)
    if order_status != ClaimStatus.SUPPORTED:
        return ClaimStatus.UNKNOWN, "temporal order must pass before gap evaluation"

    timestamps = _event_timestamps(constraint, events)
    if timestamps is None:
        return ClaimStatus.UNKNOWN, "missing or invalid temporal event"
    _, before_end_ms, after_start_ms, _ = timestamps
    gap_ms = after_start_ms - before_end_ms

    if constraint.min_gap_ms is not None and gap_ms < constraint.min_gap_ms:
        return ClaimStatus.CONTRADICTED, "gap below minimum"
    if constraint.max_gap_ms is not None and gap_ms > constraint.max_gap_ms:
        return ClaimStatus.CONTRADICTED, "gap above maximum"
    return ClaimStatus.SUPPORTED, "temporal gap passed"


def _event_timestamps(
    constraint: TemporalConstraint,
    events: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    before = events.get(constraint.before)
    after = events.get(constraint.after)
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    try:
        values = (
            int(before["start_ms"]),
            int(before["end_ms"]),
            int(after["start_ms"]),
            int(after["end_ms"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if any(value < 0 for value in values):
        return None
    if values[1] < values[0] or values[3] < values[2]:
        return None
    return values
