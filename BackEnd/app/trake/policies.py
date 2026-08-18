"""Final sequence sorting policies for TRAKE."""

from __future__ import annotations

from BackEnd.app.contracts.models import RankedCandidateRegion, TemporalSequence
from BackEnd.app.trake.config import TrakeAlignerConfig


def candidate_identity_key(candidate: RankedCandidateRegion) -> str:
    return "|".join(
        (
            candidate.video_id,
            candidate.event_id or "",
            candidate.candidate_id,
            str(candidate.start_ms),
            str(candidate.end_ms),
        )
    )


def sequence_candidate_key(sequence: TemporalSequence) -> str:
    return "||".join(
        "|".join(
            (
                sequence.video_id,
                event.event_id,
                event.candidate_id,
                str(event.start_ms),
                str(event.end_ms),
            )
        )
        for event in sequence.events
    )


def default_sequence_sort_key(sequence: TemporalSequence) -> tuple[float, int, int, str, str]:
    first_start = sequence.events[0].start_ms if sequence.events else 0
    total_gap = 0
    for prev, curr in zip(sequence.events, sequence.events[1:]):
        total_gap += max(0, curr.start_ms - prev.end_ms)
    return (
        -sequence.sequence_score,
        first_start,
        total_gap,
        sequence.video_id,
        sequence_candidate_key(sequence),
    )


def sort_sequences(
    sequences: list[TemporalSequence],
    config: TrakeAlignerConfig,
) -> list[TemporalSequence]:
    """Sort final sequences using score or first-occurrence policy."""

    if config.first_occurrence_mode == "strict":
        return sorted(
            sequences,
            key=lambda sequence: (
                sequence.events[0].start_ms if sequence.events else 0,
                -sequence.sequence_score,
                sequence.video_id,
                sequence_candidate_key(sequence),
            ),
        )

    if config.first_occurrence_mode == "soft":
        return sorted(
            sequences,
            key=lambda sequence: (
                -(
                    sequence.sequence_score
                    - (
                        (sequence.events[0].start_ms if sequence.events else 0)
                        * config.first_occurrence_soft_weight
                    )
                ),
                sequence.video_id,
                sequence_candidate_key(sequence),
            ),
        )

    return sorted(sequences, key=default_sequence_sort_key)
