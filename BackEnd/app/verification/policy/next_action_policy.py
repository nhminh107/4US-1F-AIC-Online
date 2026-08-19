"""Placeholder next-action policy module for future orchestration expansion."""

from __future__ import annotations

from BackEnd.app.verification.enums import NextAction, VerificationStatus


def default_next_action(status: VerificationStatus) -> NextAction:
    if status == VerificationStatus.ACCEPTED:
        return NextAction.RETURN_RESULT
    if status == VerificationStatus.REJECTED:
        return NextAction.TRY_NEXT_CANDIDATE
    return NextAction.EXPAND_LOCAL_CONTEXT
