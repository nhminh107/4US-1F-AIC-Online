"""Gate adapter interface."""

from __future__ import annotations

from typing import Protocol

from BackEnd.app.verification.contracts import (
    VerificationContext,
    VerificationGateDecision,
)


class VerificationGate(Protocol):
    def decide(self, context: VerificationContext) -> VerificationGateDecision:
        ...
