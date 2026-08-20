"""Build atomic verification claims from pipeline query/result contracts."""

from __future__ import annotations

import re

from BackEnd.app.contracts.models import (
    KISResult,
    StructuredQuery,
    TemporalSequence,
    VQAResult,
)
from BackEnd.app.verification.contracts import ObjectCountSpec, VerificationClaim
from BackEnd.app.verification.enums import ClaimImportance, ClaimType


class ClaimBuilder:
    def build_claims(
        self,
        query: StructuredQuery,
        result: KISResult | VQAResult | TemporalSequence,
    ) -> list[VerificationClaim]:
        claims: list[VerificationClaim] = []
        claims.extend(self._constraint_claims(query))

        if isinstance(result, VQAResult) and result.status == "answered" and result.answer:
            claims.append(
                VerificationClaim(
                    claim_id="claim-vqa-answer",
                    claim_type=ClaimType.VQA_ANSWER_CLAIM,
                    text=result.answer,
                    importance=ClaimImportance.HARD,
                    metadata={"evidence_ids": list(result.evidence_ids)},
                )
            )

        if isinstance(result, TemporalSequence):
            event_by_id = {
                event.event_id: {
                    "start_ms": event.start_ms,
                    "end_ms": event.end_ms,
                    "candidate_id": event.candidate_id,
                }
                for event in result.events
            }
            for constraint in query.temporal_constraints:
                metadata = {"constraint": constraint, "events": event_by_id}
                claims.append(
                    VerificationClaim(
                        claim_id=f"claim-temporal-order-{constraint.before}-{constraint.after}",
                        claim_type=ClaimType.TEMPORAL_ORDER,
                        text=f"{constraint.before} before {constraint.after}",
                        importance=ClaimImportance.HARD,
                        metadata=metadata,
                    )
                )
                if constraint.min_gap_ms is not None or constraint.max_gap_ms is not None:
                    claims.append(
                        VerificationClaim(
                            claim_id=f"claim-temporal-gap-{constraint.before}-{constraint.after}",
                            claim_type=ClaimType.TEMPORAL_GAP,
                            text=f"{constraint.before} gap {constraint.after}",
                            importance=ClaimImportance.HARD,
                            metadata=metadata,
                        )
                    )

        return claims

    @staticmethod
    def _constraint_claims(query: StructuredQuery) -> list[VerificationClaim]:
        claims: list[VerificationClaim] = []
        for index, text in enumerate(query.ocr_constraints, start=1):
            claims.append(
                VerificationClaim(
                    claim_id=f"claim-ocr-{index}",
                    claim_type=ClaimType.OCR_EXACT,
                    text=text,
                    importance=ClaimImportance.HARD,
                )
            )
        for index, text in enumerate(query.asr_constraints, start=1):
            claims.append(
                VerificationClaim(
                    claim_id=f"claim-asr-{index}",
                    claim_type=ClaimType.ASR_EXACT,
                    text=text,
                    importance=ClaimImportance.HARD,
                )
            )
        for index, text in enumerate(query.object_constraints, start=1):
            count_spec = _parse_object_count(text)
            is_unparsed_count = count_spec is None and any(
                character.isdigit() for character in text
            )
            claims.append(
                VerificationClaim(
                    claim_id=f"claim-object-{index}",
                    claim_type=(
                        ClaimType.OBJECT_COUNT
                        if count_spec is not None or is_unparsed_count
                        else ClaimType.OBJECT_PRESENCE
                    ),
                    text=text,
                    importance=ClaimImportance.HARD,
                    metadata=(
                        {"count_spec": count_spec}
                        if count_spec is not None
                        else ({"count_parse_error": True} if is_unparsed_count else {})
                    ),
                )
            )
        for index, text in enumerate(query.negative_constraints, start=1):
            claims.append(
                VerificationClaim(
                    claim_id=f"claim-negative-{index}",
                    claim_type=ClaimType.NEGATIVE_CONSTRAINT,
                    text=text,
                    importance=ClaimImportance.HARD,
                )
            )
        return claims


_COUNT_PATTERNS = (
    ("at_least", re.compile(r"^at\s+least\s+(\d+)\s+(.+)$", re.IGNORECASE)),
    ("at_least", re.compile(r"^ít\s+nhất\s+(\d+)\s+(.+)$", re.IGNORECASE)),
    ("at_most", re.compile(r"^at\s+most\s+(\d+)\s+(.+)$", re.IGNORECASE)),
    ("at_most", re.compile(r"^không\s+quá\s+(\d+)\s+(.+)$", re.IGNORECASE)),
    ("exact", re.compile(r"^(?:exactly\s+)?(\d+)\s+(.+)$", re.IGNORECASE)),
)


def _parse_object_count(text: str) -> ObjectCountSpec | None:
    normalized = " ".join(text.split())
    for operator, pattern in _COUNT_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match is None:
            continue
        expected_count = int(match.group(1))
        if expected_count < 1:
            return None
        return ObjectCountSpec(
            operator=operator,
            expected_count=expected_count,
            object_label=_singularize_label(match.group(2)),
        )
    return None


def _singularize_label(label: str) -> str:
    words = label.strip().casefold().split()
    if not words:
        return label.strip()
    last = words[-1]
    if last in {"people", "persons"}:
        words[-1] = "person"
    elif last.endswith("ies") and len(last) > 3:
        words[-1] = f"{last[:-3]}y"
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 1:
        words[-1] = last[:-1]
    return " ".join(words)
