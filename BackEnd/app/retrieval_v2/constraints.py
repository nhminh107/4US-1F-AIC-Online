from __future__ import annotations

from BackEnd.app.retrieval_v2.contracts import ConstraintDecision, MomentBand, QueryAtom


_ALLOWED_RETRIEVERS = {
    "visual": {"frame_search", "clip_search", "shot_search"},
    "ocr": {"ocr_search"},
    "asr": {"asr_search"},
    "object": {"object_search", "track_search", "frame_search"},
}


class HardConstraintEngine:
    """Deterministic tri-state checks shared by planning and candidate gates.

    Implements the 4-gate architecture from pipeline.md §6G:
      - Plan gate: tool/modality/alias validation
      - Candidate gate: required atom presence, contradiction
      - Task gate: temporal order, TRAKE arity
      - Submission gate: official frame, duplicate row, format
    """

    # ---------- Plan gate ----------

    def validate_retriever(
        self,
        atom: QueryAtom,
        retriever: str,
    ) -> ConstraintDecision:
        # Check atom's own forbidden list first (new in Phase B)
        if atom.forbidden_retrievers and retriever in atom.forbidden_retrievers:
            return ConstraintDecision(
                constraint_id=f"retriever:{atom.atom_id}",
                status="FAIL",
                scope="PLAN",
                reason_code="FORBIDDEN_MODALITY",
            )
        # Check atom's allowed list if populated
        if atom.allowed_retrievers and retriever not in atom.allowed_retrievers:
            return ConstraintDecision(
                constraint_id=f"retriever:{atom.atom_id}",
                status="FAIL",
                scope="PLAN",
                reason_code="FORBIDDEN_MODALITY",
            )
        # Fall back to modality-based allowed set
        allowed = _ALLOWED_RETRIEVERS[atom.modality]
        if retriever in allowed:
            return ConstraintDecision(
                constraint_id=f"retriever:{atom.atom_id}",
                status="PASS",
                scope="PLAN",
                reason_code="ALLOWED_RETRIEVER",
            )
        return ConstraintDecision(
            constraint_id=f"retriever:{atom.atom_id}",
            status="FAIL",
            scope="PLAN",
            reason_code="FORBIDDEN_MODALITY",
        )

    # ---------- Candidate gate ----------

    def evaluate_required_atom(
        self,
        *,
        atom_id: str,
        evidence_ids: list[str],
        contradiction_evidence_ids: list[str],
        scope: str,
        verified_status: str = "UNKNOWN",
    ) -> ConstraintDecision:
        if contradiction_evidence_ids:
            return ConstraintDecision(
                constraint_id=f"required:{atom_id}",
                status="FAIL",
                scope=scope,
                reason_code="POSITIVE_CONSTRAINT_CONTRADICTION",
                evidence_ids=contradiction_evidence_ids,
            )
        if verified_status == "FAIL":
            return ConstraintDecision(
                constraint_id=f"required:{atom_id}",
                status="FAIL",
                scope=scope,
                reason_code="REQUIRED_ATOM_VERIFIED_MISMATCH",
                evidence_ids=evidence_ids,
            )
        if evidence_ids and verified_status == "PASS":
            return ConstraintDecision(
                constraint_id=f"required:{atom_id}",
                status="PASS",
                scope=scope,
                reason_code="REQUIRED_ATOM_SUPPORTED",
                evidence_ids=evidence_ids,
            )
        return ConstraintDecision(
            constraint_id=f"required:{atom_id}",
            status="UNKNOWN",
            scope=scope,
            reason_code=(
                "REQUIRED_ATOM_RETRIEVED_UNVERIFIED"
                if evidence_ids
                else "MISSING_REQUIRED_ATOM"
            ),
            confidence=0.0,
        )

    def evaluate_band(
        self,
        band: MomentBand,
        atoms: list[QueryAtom],
    ) -> list[ConstraintDecision]:
        decisions: list[ConstraintDecision] = []
        for atom in atoms:
            if atom.operator == "MUST_NOT":
                cell = band.contradictions.get(atom.atom_id)
                if cell is not None and cell.status == "PASS":
                    decisions.append(ConstraintDecision(
                        constraint_id=f"negative:{atom.atom_id}",
                        status="FAIL",
                        scope="MOMENT_BAND",
                        reason_code="VERIFIED_NEGATIVE_CONSTRAINT",
                        evidence_ids=cell.evidence_ids,
                    ))
                elif cell is not None and cell.status == "FAIL":
                    decisions.append(ConstraintDecision(
                        constraint_id=f"negative:{atom.atom_id}",
                        status="PASS",
                        scope="MOMENT_BAND",
                        reason_code="NEGATIVE_CONSTRAINT_CLEARED",
                        evidence_ids=cell.evidence_ids,
                    ))
                else:
                    decisions.append(ConstraintDecision(
                        constraint_id=f"negative:{atom.atom_id}",
                        status="UNKNOWN",
                        scope="MOMENT_BAND",
                        reason_code="NEGATIVE_CONSTRAINT_UNVERIFIED",
                        evidence_ids=cell.evidence_ids if cell else [],
                        confidence=0.0,
                    ))
                continue
            if atom.required and atom.operator == "MUST":
                cell = band.coverage.get(atom.atom_id)
                decisions.append(self.evaluate_required_atom(
                    atom_id=atom.atom_id,
                    evidence_ids=cell.evidence_ids if cell else [],
                    contradiction_evidence_ids=[],
                    scope="MOMENT_BAND",
                    verified_status=cell.status if cell else "UNKNOWN",
                ))
        return decisions

    # ---------- Task gate ----------

    def validate_temporal_order(
        self,
        bands: list[MomentBand],
        event_order: list[str],
    ) -> ConstraintDecision:
        """Check that events within a single video appear in the expected order.

        Returns PASS if bands matching the events appear in increasing start_ms.
        Returns FAIL if a clear order violation exists.
        Returns UNKNOWN if some events are missing.
        """
        if len(event_order) < 2:
            return ConstraintDecision(
                constraint_id="temporal_order",
                status="PASS",
                scope="VIDEO",
                reason_code="SINGLE_EVENT_NO_ORDER",
            )

        # Group bands by video
        by_video: dict[str, dict[str, int]] = {}
        for band in bands:
            if band.event_id in event_order:
                by_video.setdefault(band.video_id, {})
                existing = by_video[band.video_id].get(band.event_id, band.start_ms)
                by_video[band.video_id][band.event_id] = min(existing, band.start_ms)

        violations: list[str] = []
        for video_id, event_times in by_video.items():
            if len(event_times) < 2:
                continue
            ordered_events = [e for e in event_order if e in event_times]
            times = [event_times[e] for e in ordered_events]
            for i in range(len(times) - 1):
                if times[i] > times[i + 1]:
                    violations.append(f"{video_id}:{ordered_events[i]}>{ordered_events[i+1]}")

        if violations:
            return ConstraintDecision(
                constraint_id="temporal_order",
                status="FAIL",
                scope="VIDEO",
                reason_code="TEMPORAL_ORDER_VIOLATION",
                evidence_ids=violations,
            )
        return ConstraintDecision(
            constraint_id="temporal_order",
            status="PASS",
            scope="VIDEO",
            reason_code="TEMPORAL_ORDER_VALID",
        )

    def validate_trake_arity(
        self,
        bands: list[MomentBand],
        expected_event_count: int,
        video_id: str,
    ) -> ConstraintDecision:
        """Check that a video has bands covering all expected events."""
        video_bands = [b for b in bands if b.video_id == video_id]
        covered_events = {b.event_id for b in video_bands if b.event_id is not None}
        if len(covered_events) >= expected_event_count:
            return ConstraintDecision(
                constraint_id=f"trake_arity:{video_id}",
                status="PASS",
                scope="VIDEO",
                reason_code="TRAKE_ARITY_SATISFIED",
            )
        if not covered_events:
            return ConstraintDecision(
                constraint_id=f"trake_arity:{video_id}",
                status="UNKNOWN",
                scope="VIDEO",
                reason_code="TRAKE_NO_EVENTS_FOUND",
                confidence=0.0,
            )
        return ConstraintDecision(
            constraint_id=f"trake_arity:{video_id}",
            status="UNKNOWN",
            scope="VIDEO",
            reason_code="TRAKE_EVENT_ARITY_MISMATCH",
            confidence=round(len(covered_events) / expected_event_count, 3),
        )

    # ---------- Submission gate ----------

    def validate_submission_row(
        self,
        *,
        video_id: str,
        frame_idx: int,
        is_official_frame: bool,
        seen_pairs: set[tuple[str, int]] | None = None,
    ) -> ConstraintDecision:
        """Validate a single submission row for BTC format compliance."""
        pair = (video_id, frame_idx)

        if not is_official_frame:
            return ConstraintDecision(
                constraint_id=f"submission:{video_id}:{frame_idx}",
                status="FAIL",
                scope="SUBMISSION_ROW",
                reason_code="INVALID_OFFICIAL_FRAME",
            )
        if seen_pairs is not None and pair in seen_pairs:
            return ConstraintDecision(
                constraint_id=f"submission:{video_id}:{frame_idx}",
                status="FAIL",
                scope="SUBMISSION_ROW",
                reason_code="DUPLICATE_SUBMISSION_ROW",
            )
        return ConstraintDecision(
            constraint_id=f"submission:{video_id}:{frame_idx}",
            status="PASS",
            scope="SUBMISSION_ROW",
            reason_code="VALID_SUBMISSION_ROW",
        )


__all__ = ["HardConstraintEngine"]
