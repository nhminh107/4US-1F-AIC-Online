"""Rule-based verification gate."""

from __future__ import annotations

from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.contracts import (
    VerificationContext,
    VerificationGateDecision,
)
from BackEnd.app.verification.enums import VerificationStatus


class RuleBasedVerificationGate:
    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        reasons: list[str] = []

        if context.hard_contradicted > 0:
            return VerificationGateDecision(
                should_verify=False,
                reasons=["hard_constraint_contradicted"],
                direct_status=VerificationStatus.REJECTED,
            )
        if context.hard_unknown > 0:
            reasons.append("hard_constraint_unknown")
        if (
            context.score_margin is not None
            and context.score_margin < self.config.gate.min_score_margin
        ):
            reasons.append("small_top1_top2_margin")
        if context.task == "VQA":
            if (
                context.answer_confidence is not None
                and context.answer_confidence < self.config.gate.vqa_min_answer_confidence
            ):
                reasons.append("low_vqa_confidence")
            if context.answer_evidence_count == 0:
                reasons.append("vqa_answer_has_no_evidence")
        if context.task == "TRAKE":
            if (
                context.weakest_event_score is not None
                and context.weakest_event_score < self.config.gate.trake_min_event_score
            ):
                reasons.append("weak_trake_event")
            if (
                context.sequence_margin is not None
                and context.sequence_margin < self.config.gate.trake_min_sequence_margin
            ):
                reasons.append("ambiguous_sequence")

        return VerificationGateDecision(
            should_verify=bool(reasons),
            reasons=reasons,
        )
