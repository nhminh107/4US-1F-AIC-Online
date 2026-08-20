"""Readable orchestration of the online retrieval pipeline described in the proposal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Fusion.fusion_and_ranking import FusionRanking
from BackEnd.app.KIS.kis_handler import KISHandler
from BackEnd.app.aggregator import Aggregator
from BackEnd.app.api.config import (
    CANDIDATE_MERGE_GAP_MS,
    SELECTIVE_VERIFIER_ENABLED,
)
from BackEnd.app.api.models import (
    QueryRequest,
    QueryResponse,
    TopKParameters,
    TrakeEventResult,
    TrakeSequenceResult,
    VerificationItem,
    VerificationSummary,
    VisualResult,
)
from BackEnd.app.contracts.models import (
    KISResult,
    RawQuery,
    RankedCandidateRegion,
    SearchHit,
    StructuredQuery,
    TemporalSequence,
    ToolCall,
    VerifiedResult,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.fast_path.runner import run_fast_path
from BackEnd.app.intent_extractor.extractor import extract_intent
from BackEnd.app.query_planner import execute_tool_calls, run_query_planner
from BackEnd.app.services.evidence_service import get_evidence_bundle
from BackEnd.app.trake import TrakeAlignerResult, TrakeTemporalAligner
from BackEnd.app.verification import VerificationService
from BackEnd.app.verification.config import VerificationConfig
from BackEnd.app.verification.evidence.provider import DatabaseEvidenceProvider


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ResolvedFrame:
    """The official frame that the API will actually display."""

    metadata: FrameMetadata
    img_url: str


class FrameResolver:
    """Resolve task results to stable, displayable official frames."""

    def __init__(self, db_mng: PostgreManager) -> None:
        self.db_mng = db_mng

    def from_frame_id(self, frame_id: str) -> ResolvedFrame | None:
        try:
            selected = self.db_mng.get_frame_record_by_frame_id(frame_id)
        except ValueError:
            return None
        return self._official_display_frame(selected)

    def from_time_range(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> ResolvedFrame | None:
        """Use the evidence service to inspect frames inside a TRAKE event."""

        bundle = get_evidence_bundle(
            video_id,
            start_ms,
            end_ms,
            self.db_mng,
            modalities={"frame"},
        )
        midpoint_ms = (start_ms + end_ms) // 2
        if bundle.frames:
            selected = min(
                bundle.frames,
                key=lambda frame: (
                    abs(frame.timestamp_ms - midpoint_ms),
                    0 if frame.source == "official" else 1,
                    frame.frame_idx,
                ),
            )
            return self._official_display_frame(selected)

        nearest = self.db_mng.get_nearest_official_frame(video_id, midpoint_ms)
        return self._as_resolved(nearest) if nearest is not None else None

    def media_path(self, frame_id: str) -> Path:
        """Resolve a DB-backed media ID to a local file for ``FileResponse``."""

        frame = self.db_mng.get_frame_record_by_frame_id(frame_id)
        if frame.frame_path is None:
            raise FileNotFoundError(f"Frame '{frame_id}' has no image path.")
        path = frame.frame_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image for frame '{frame_id}' does not exist.")
        return path

    def _official_display_frame(
        self,
        selected: FrameMetadata,
    ) -> ResolvedFrame | None:
        if selected.source == "official" and selected.frame_path is not None:
            return self._as_resolved(selected)

        # Extracted frames are retrieval evidence only. The UI always receives
        # the nearest official frame so its URL and frame_idx remain canonical.
        official = self.db_mng.get_nearest_official_frame(
            selected.video_id,
            selected.timestamp_ms,
        )
        return self._as_resolved(official) if official is not None else None

    @staticmethod
    def _as_resolved(frame: FrameMetadata) -> ResolvedFrame:
        if frame.frame_path is None:
            raise ValueError(
                f"Official frame '{frame.frame_id}' has no frame_path in PostgreSQL."
            )
        return ResolvedFrame(
            metadata=frame,
            # Required API contract: img_url is the frame_path persisted in
            # PostgreSQL, not a URL synthesized from the frame ID.
            img_url=str(frame.frame_path),
        )


class OnlinePipeline:
    """Intent → retrieval → aggregation → fusion → task handler → verifier."""

    def __init__(
        self,
        db_mng: PostgreManager,
        *,
        selective_verifier_enabled: bool = SELECTIVE_VERIFIER_ENABLED,
    ) -> None:
        self.db_mng = db_mng
        self.selective_verifier_enabled = selective_verifier_enabled
        self.aggregator = Aggregator(db_mng)
        self.fusion = FusionRanking()
        self.kis_handler = KISHandler(db_mng)
        self.trake_aligner = TrakeTemporalAligner()
        self.frame_resolver = FrameResolver(db_mng)

        verifier_config = VerificationConfig(enabled=selective_verifier_enabled)
        evidence_provider = DatabaseEvidenceProvider(
            db_mng,
            config=verifier_config,
        )
        self.verifier = VerificationService(
            evidence_provider,
            config=verifier_config,
        )

    async def execute(self, request: QueryRequest) -> QueryResponse:
        structured_query = await extract_intent(
            RawQuery(
                text=request.prompt,
                feedback=request.feedback,
                session_id=request.session_id,
            )
        )
        hits, tool_calls, execution_path = await self._retrieve(
            structured_query,
            request.top_k,
        )
        candidate_regions = self.aggregator.execute(
            hits,
            merge_gap=CANDIDATE_MERGE_GAP_MS,
        )
        ranked_candidates = self.fusion.fusion_and_ranking(
            candidate_regions,
            self._fusion_weight_flags(hits),
            structured_query,
        )

        warnings: list[str] = []
        trake_result: TrakeAlignerResult | None = None
        raw_results: list[KISResult] | list[TemporalSequence]

        if structured_query.task == "KIS":
            raw_visual_results = self.kis_handler.execute(
                structured_query,
                ranked_candidates,
                top_n=request.top_k.result_top_k,
            )
            visual_results, raw_results = self._enrich_visual_results(
                raw_visual_results,
                warnings,
            )
            api_results: list[VisualResult] | list[TrakeSequenceResult] = visual_results
        elif structured_query.task == "VQA":
            # Human-review VQA intentionally shares the complete KIS final
            # selection logic; only intent/retrieval differ between the tasks.
            raw_visual_results = self.kis_handler.execute(
                structured_query,
                ranked_candidates,
                top_n=request.top_k.result_top_k,
            )
            visual_results, raw_results = self._enrich_visual_results(
                raw_visual_results,
                warnings,
            )
            api_results = visual_results
        else:
            trake_aligner = TrakeTemporalAligner(
                self.trake_aligner.config.model_copy(
                    update={"top_k_sequences": request.top_k.result_top_k}
                )
            )
            trake_result = trake_aligner.align(
                ranked_candidates,
                [event.event_id for event in structured_query.events],
                structured_query.temporal_constraints,
            )
            sequence_results, raw_results = self._enrich_trake_sequences(
                trake_result.sequences,
                warnings,
            )
            api_results = sequence_results

        verification, verification_warnings = await self._verify_results(
            structured_query,
            raw_results,
            ranked_candidates,
        )
        warnings.extend(verification_warnings)

        return QueryResponse(
            query_id=structured_query.query_id,
            task=structured_query.task,
            structured_query=structured_query,
            top_k=request.top_k,
            execution_path=execution_path,
            tool_calls=tool_calls,
            search_hit_count=len(hits),
            candidate_count=len(ranked_candidates),
            results=api_results,
            trake_status=trake_result.status if trake_result is not None else None,
            replan_required=(
                trake_result.replan_required if trake_result is not None else False
            ),
            missing_event_ids=(
                trake_result.missing_event_ids if trake_result is not None else []
            ),
            verification=verification,
            warnings=warnings,
        )

    async def _retrieve(
        self,
        query: StructuredQuery,
        top_k: TopKParameters,
    ) -> tuple[
        list[SearchHit],
        list[ToolCall],
        str,
    ]:
        if not self._requires_query_planner(query):
            return (
                await run_fast_path(
                    query,
                    top_k=top_k.model_dump(exclude={"result_top_k"}),
                ),
                [],
                "fast_path",
            )

        tool_calls = await run_query_planner(query)
        execution_path = "query_planner"
        if not tool_calls:
            tool_calls = self._fallback_tool_calls(query, top_k)
            execution_path = "query_planner_fallback"
        else:
            tool_calls = self._apply_top_k(tool_calls, top_k)
        return await execute_tool_calls(tool_calls), tool_calls, execution_path

    @staticmethod
    def _requires_query_planner(query: StructuredQuery) -> bool:
        """TRAKE and object/negative constraints require explicit tool planning."""

        return bool(
            query.task == "TRAKE"
            or query.events
            or query.object_constraints
            or query.negative_constraints
        )

    @staticmethod
    def _fallback_tool_calls(
        query: StructuredQuery,
        top_k: TopKParameters,
    ) -> list[ToolCall]:
        """Deterministic fallback used only when the planner returns no calls."""

        planned: list[tuple[str, dict[str, object], str | None]] = []
        if query.events:
            planned.extend(
                (
                    "clip_search",
                    {
                        "query": event.description,
                        "top_k": top_k.clip_search,
                    },
                    event.event_id,
                )
                for event in query.events
            )
        else:
            visual_queries = query.visual_queries or (
                [query.question] if query.question else []
            )
            planned.extend(
                (
                    "clip_search",
                    {"query": text, "top_k": top_k.clip_search},
                    None,
                )
                for text in visual_queries
            )

        planned.extend(
            (
                "ocr_search",
                {
                    "query": text,
                    "top_k": top_k.ocr_search,
                    "mode": "fuzzy",
                },
                None,
            )
            for text in query.ocr_constraints
        )
        planned.extend(
            (
                "asr_search",
                {
                    "query": text,
                    "top_k": top_k.asr_search,
                    "mode": "fuzzy",
                },
                None,
            )
            for text in query.asr_constraints
        )
        planned.extend(
            (
                "object_search",
                {
                    "object_class": object_class,
                    "top_k": top_k.object_search,
                    "min_count": 1,
                },
                None,
            )
            for object_class in query.object_constraints
        )
        return [
            ToolCall(
                tool_call_id=f"fallback_{index:03d}",
                tool_name=tool_name,
                parameters=parameters,
                event_id=event_id,
            )
            for index, (tool_name, parameters, event_id) in enumerate(
                planned,
                start=1,
            )
        ]

    @staticmethod
    def _apply_top_k(
        tool_calls: list[ToolCall],
        top_k: TopKParameters,
    ) -> list[ToolCall]:
        """Make request-level K authoritative over values chosen by the LLM."""

        return [
            call.model_copy(
                update={
                    "parameters": {
                        **call.parameters,
                        "top_k": top_k.for_tool(call.tool_name),
                    }
                }
            )
            for call in tool_calls
        ]

    def _fusion_weight_flags(self, hits: list[SearchHit]) -> dict[str, bool]:
        present_entity_types = {hit.entity_type for hit in hits}
        return {
            entity_type: entity_type in present_entity_types
            for entity_type in self.fusion.weight_mapping
        }

    def _enrich_visual_results(
        self,
        results: list[KISResult],
        warnings: list[str],
    ) -> tuple[list[VisualResult], list[KISResult]]:
        enriched: list[VisualResult] = []
        retained: list[KISResult] = []
        for result in results:
            display = self.frame_resolver.from_frame_id(
                result.representative_frame_id
            )
            if display is None:
                warnings.append(
                    "Skipped result because no displayable official frame exists: "
                    f"{result.representative_frame_id}"
                )
                continue
            enriched.append(
                VisualResult(
                    **result.model_dump(),
                    display_frame_id=display.metadata.frame_id,
                    frame_idx=display.metadata.frame_idx,
                    img_url=display.img_url,
                )
            )
            retained.append(result)
        return enriched, retained

    def _enrich_trake_sequences(
        self,
        sequences: list[TemporalSequence],
        warnings: list[str],
    ) -> tuple[list[TrakeSequenceResult], list[TemporalSequence]]:
        enriched: list[TrakeSequenceResult] = []
        retained: list[TemporalSequence] = []
        for sequence in sequences:
            events: list[TrakeEventResult] = []
            for event in sequence.events:
                display = self.frame_resolver.from_time_range(
                    sequence.video_id,
                    event.start_ms,
                    event.end_ms,
                )
                if display is None:
                    warnings.append(
                        "Skipped TRAKE sequence because no displayable official "
                        f"frame exists for event {event.event_id}."
                    )
                    events = []
                    break
                events.append(
                    TrakeEventResult(
                        **event.model_dump(),
                        display_frame_id=display.metadata.frame_id,
                        frame_idx=display.metadata.frame_idx,
                        img_url=display.img_url,
                    )
                )
            if not events:
                continue
            enriched.append(
                TrakeSequenceResult(
                    video_id=sequence.video_id,
                    events=events,
                    sequence_score=sequence.sequence_score,
                )
            )
            retained.append(sequence)
        return enriched, retained

    async def _verify_results(
        self,
        query: StructuredQuery,
        results: list[KISResult] | list[TemporalSequence],
        ranked_candidates: list[RankedCandidateRegion],
    ) -> tuple[VerificationSummary, list[str]]:
        if not self.selective_verifier_enabled:
            return (
                VerificationSummary(
                    enabled=False,
                    applied=False,
                    reason="selective_verifier_disabled",
                ),
                [],
            )
        if query.task == "VQA":
            return (
                VerificationSummary(
                    enabled=True,
                    applied=False,
                    reason="human_review_vqa_has_no_generated_answer",
                ),
                [],
            )
        if not results:
            return (
                VerificationSummary(
                    enabled=True,
                    applied=False,
                    reason="no_displayable_results",
                ),
                [],
            )

        limit = self.verifier.config.budget.max_candidates_to_verify
        items: list[VerificationItem] = []
        warnings: list[str] = []
        for index, result in enumerate(results[:limit], start=1):
            target_id = self._verification_target_id(result, index)
            try:
                verified = await self.verifier.verify(
                    query,
                    result,
                    ranked_candidates,
                )
            except Exception:
                logger.exception("Selective verifier failed for target=%s", target_id)
                warnings.append(f"Selective verifier failed for {target_id}.")
                verified = VerifiedResult(
                    status="uncertain",
                    confidence=0.0,
                    failed_constraints=["verification_error"],
                )
            items.append(VerificationItem(target_id=target_id, result=verified))

        return (
            VerificationSummary(
                enabled=True,
                applied=True,
                reason="limited_to_top_candidates" if len(results) > limit else None,
                items=items,
            ),
            warnings,
        )

    @staticmethod
    def _verification_target_id(
        result: KISResult | TemporalSequence,
        index: int,
    ) -> str:
        if isinstance(result, KISResult):
            return result.representative_frame_id
        candidate_ids = "-".join(event.candidate_id for event in result.events)
        return candidate_ids or f"{result.video_id}-sequence-{index}"


__all__ = ["FrameResolver", "OnlinePipeline", "ResolvedFrame"]
