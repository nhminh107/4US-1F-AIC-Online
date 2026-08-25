"""Readable orchestration of the online retrieval pipeline described in the proposal."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Fusion.fusion_and_ranking import FusionRanking
from BackEnd.app.KIS.kis_handler import KISHandler
from BackEnd.app.aggregator import Aggregator
from BackEnd.app.api.config import (
    CANDIDATE_MERGE_GAP_MS,
    RETRIEVAL_V2_CANDIDATE_BUDGET,
    RETRIEVAL_V2_CORPUS_STATS_PATH,
    RETRIEVAL_V2_ENABLED,
    RETRIEVAL_V2_VIDEO_INDEX_PATH,
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
    VQAVisualResult,
)
from BackEnd.app.contracts.models import (
    CandidateEvidence,
    ConstraintResult,
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
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.gateway import ToolSearchGateway
from BackEnd.app.retrieval_v2.contracts import SearchControllerResult
from BackEnd.app.retrieval_v2.adapters.reranker import (
    CandidateReranker,
    ClipOfficialFrameReranker,
)
from BackEnd.app.retrieval_v2.review import CandidateReviewer
from BackEnd.app.retrieval_v2.adapters.vlm_reviewer import DeterministicFallbackReviewer
from BackEnd.app.retrieval_v2.corpus_stats import CorpusStats
from BackEnd.app.retrieval_v2.prompt_planner import CorpusAwarePromptPlanner
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex
from BackEnd.app.vqa.provider import build_default_vqa_provider
from BackEnd.app.retrieval_v2.metrics import RetrievalMetricsCollector
from BackEnd.app.retrieval_v2.logging import emit_audit_log
from BackEnd.app.retrieval_v2.frame_selector import (
    FrameCandidate,
    QueryAwareOfficialFrameSelector,
)
from BackEnd.app.retrieval_v2.task_heads import (
    AnswerClaim,
    GroundedQAResult,
    OfficialFrame,
    aggregate_grounded_answer,
    validate_trake_sequence,
)


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
        if frame is None or frame.frame_path is None:
            raise FileNotFoundError(f"Frame '{frame_id}' not found or has no image path.")
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
            # PostgreSQL, formatted with canonical posix slashes.
            img_url=Path(str(frame.frame_path)).as_posix(),
        )


class DatabaseOfficialFrameProvider:
    """Cache official frame metadata per video for query-aware selection."""

    def __init__(self, db_mng: PostgreManager) -> None:
        self.db_mng = db_mng
        self._cache: dict[str, list[FrameCandidate]] = {}
        self._region_cache: dict[tuple[str, int, int], list[FrameCandidate]] = {}

    def prefetch(self, bands) -> None:
        regions = list(dict.fromkeys(
            (band.video_id, band.start_ms, band.end_ms) for band in bands
        ))
        if not regions:
            return
        batch_get = getattr(self.db_mng, "batch_get_frames_for_regions", None)
        if not callable(batch_get):
            return
        resolved = batch_get(regions)
        for region, frames in resolved.items():
            self._region_cache[region] = [
                self._as_candidate(frame)
                for frame in frames
                if frame.source == "official" and frame.frame_path is not None
            ]

    def get_official_frames(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> list[FrameCandidate]:
        region_key = (video_id, start_ms, end_ms)
        if region_key in self._region_cache:
            in_band = self._region_cache[region_key]
            if in_band:
                return in_band
        if video_id not in self._cache:
            try:
                frames = self.db_mng.get_frame_record_by_video_id(video_id)
            except ValueError:
                frames = []
            self._cache[video_id] = [self._as_candidate(frame) for frame in frames]
        in_band = [
            frame
            for frame in self._cache[video_id]
            if frame.is_official and start_ms <= frame.timestamp_ms <= end_ms
        ]
        if in_band:
            return in_band
        nearest = self.db_mng.get_nearest_official_frame(
            video_id,
            (start_ms + end_ms) // 2,
        )
        if nearest is None or nearest.frame_path is None:
            return []
        midpoint_ms = (start_ms + end_ms) // 2
        if abs(nearest.timestamp_ms - midpoint_ms) > 5_000:
            return []
        return [
            self._as_candidate(nearest)
        ]

    @staticmethod
    def _as_candidate(frame: FrameMetadata) -> FrameCandidate:
        return FrameCandidate(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            img_url=(
                Path(str(frame.frame_path)).as_posix()
                if frame.frame_path is not None
                else None
            ),
            frame_id=frame.frame_id,
            is_official=frame.source == "official",
            exists=frame.frame_path is not None,
        )


class QAAnswerProvider(Protocol):
    async def answer_claims(
        self,
        question: str,
        frames: Sequence[VisualResult],
        allowed_evidence_ids: set[str],
    ) -> Sequence[AnswerClaim]: ...


class _LazyVisualEmbedder:
    def encode_texts(self, texts):
        from BackEnd.app.retrieval_tools.visual import get_visual_retrieval_tools

        return get_visual_retrieval_tools().embedder.encode_texts(texts)

    def encode_image(self, image_path):
        from BackEnd.app.retrieval_tools.visual import get_visual_retrieval_tools

        return get_visual_retrieval_tools().embedder.encode_image(image_path)


class OnlinePipeline:
    """Intent → retrieval → aggregation → fusion → task handler → verifier."""

    def __init__(
        self,
        db_mng: PostgreManager,
        *,
        selective_verifier_enabled: bool = SELECTIVE_VERIFIER_ENABLED,
        retrieval_v2_enabled: bool = RETRIEVAL_V2_ENABLED,
        search_controller: SearchController | None = None,
        candidate_reviewer: CandidateReviewer | None = None,
        candidate_reranker: CandidateReranker | None = None,
        qa_answer_provider: QAAnswerProvider | None = None,
    ) -> None:
        self.db_mng = db_mng
        self.selective_verifier_enabled = selective_verifier_enabled
        self.retrieval_v2_enabled = retrieval_v2_enabled
        self.qa_answer_provider = qa_answer_provider or build_default_vqa_provider(db_mng)
        self.retrieval_metrics = RetrievalMetricsCollector()
        self.retrieval_v2_degraded_reasons: list[str] = []
        self.search_controller = search_controller
        if retrieval_v2_enabled and self.search_controller is None:
            corpus_stats = None
            if RETRIEVAL_V2_CORPUS_STATS_PATH:
                try:
                    corpus_stats = CorpusStats.load(RETRIEVAL_V2_CORPUS_STATS_PATH)
                except Exception as exc:
                    logger.warning("Retrieval V2 corpus stats unavailable: %r", exc)
                    self.retrieval_v2_degraded_reasons.append("corpus_stats_unavailable")
            else:
                self.retrieval_v2_degraded_reasons.append("corpus_stats_not_configured")
            video_index = None
            if RETRIEVAL_V2_VIDEO_INDEX_PATH:
                try:
                    video_index = VideoLevelIndex.load_versioned(RETRIEVAL_V2_VIDEO_INDEX_PATH)
                except Exception as exc:
                    logger.warning("Retrieval V2 video index unavailable: %r", exc)
                    self.retrieval_v2_degraded_reasons.append("video_index_unavailable")
            else:
                self.retrieval_v2_degraded_reasons.append("video_index_not_configured")
            video_query_encoder = None
            if video_index is not None:
                from BackEnd.app.retrieval_tools.visual import get_visual_retrieval_tools

                video_query_encoder = get_visual_retrieval_tools().embedder.encode_text
            self.search_controller = SearchController(
                ToolSearchGateway(),
                budget=RETRIEVAL_V2_CANDIDATE_BUDGET,
                reviewer=candidate_reviewer or DeterministicFallbackReviewer(),
                reranker=candidate_reranker or ClipOfficialFrameReranker(
                    DatabaseOfficialFrameProvider(db_mng),
                    _LazyVisualEmbedder(),
                ),
                video_index=video_index,
                video_query_encoder=video_query_encoder,
                prompt_planner=CorpusAwarePromptPlanner(corpus_stats),
                metrics_collector=self.retrieval_metrics,
            )
        self.aggregator = Aggregator(db_mng)
        self.fusion = FusionRanking()
        self.kis_handler = KISHandler(db_mng)
        self.trake_aligner = TrakeTemporalAligner()
        self.frame_resolver = FrameResolver(db_mng)
        self.official_frame_selector = QueryAwareOfficialFrameSelector()

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
                task_hint=request.task_hint,
            )
        )
        retrieval_v2_result: SearchControllerResult | None = None
        warnings: list[str] = []
        if self.retrieval_v2_enabled:
            if self.search_controller is None:
                raise RuntimeError("V2 retrieval is enabled without a SearchController")
            try:
                retrieval_v2_result = await self.search_controller.search(structured_query)
            except ValueError as exc:
                if "translated to English" not in str(exc):
                    raise
                logger.warning("V2 visual translation boundary failed; using legacy path")
                warnings.append(
                    "Retrieval V2 could not obtain an English visual plan; used the legacy fallback."
                )
                hits, tool_calls, _ = await self._retrieve(
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
                execution_path = "query_planner_fallback"
            else:
                emit_audit_log(retrieval_v2_result)
                hits = self._hits_from_v2_result(retrieval_v2_result)
                tool_calls = self._tool_calls_from_v2_result(retrieval_v2_result)
                execution_path = "retrieval_v2"
                ranked_candidates = self._ranked_candidates_from_v2_result(
                    retrieval_v2_result
                )
        else:
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

        trake_result: TrakeAlignerResult | None = None
        qa_result: GroundedQAResult | None = None
        raw_results: list[KISResult] | list[TemporalSequence]

        if structured_query.task == "KIS":
            if retrieval_v2_result is not None:
                visual_results, raw_results = self._select_v2_official_frames(
                    retrieval_v2_result,
                    request.top_k.result_top_k,
                    warnings,
                )
            else:
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
            if retrieval_v2_result is not None:
                visual_results, raw_results = self._select_v2_official_frames(
                    retrieval_v2_result,
                    request.top_k.result_top_k,
                    warnings,
                )
            else:
                raw_visual_results = self.kis_handler.execute(
                    structured_query,
                    ranked_candidates,
                    top_n=request.top_k.result_top_k,
                )
                visual_results, raw_results = self._enrich_visual_results(
                    raw_visual_results,
                    warnings,
                )
            qa_result = await self._answer_vqa(
                structured_query,
                visual_results,
                retrieval_v2_result,
            )
            api_results = [
                VQAVisualResult(
                    **item.model_dump(),
                    answer=qa_result.answer,
                    answer_status=qa_result.status,
                )
                for item in visual_results
            ]
        else:
            trake_aligner = TrakeTemporalAligner(
                self.trake_aligner.config.model_copy(
                    update={"top_k_sequences": request.top_k.result_top_k}
                )
            )
            event_ids = [event.event_id for event in structured_query.events]
            trake_result = trake_aligner.align(
                ranked_candidates,
                event_ids,
                structured_query.temporal_constraints,
            )
            if (
                trake_result.replan_required
                and structured_query.events
            ):
                ranked_candidates, hits = await self._trake_fallback_second_pass(
                    structured_query,
                    ranked_candidates,
                    hits,
                    event_ids,
                    missing_event_ids=trake_result.missing_event_ids,
                )
                trake_result = trake_aligner.align(
                    ranked_candidates,
                    event_ids,
                    structured_query.temporal_constraints,
                )

            sequence_results, raw_results = self._enrich_trake_sequences(
                trake_result.sequences,
                warnings,
                event_descriptions={
                    event.event_id: event.description
                    for event in structured_query.events
                },
            )
            if retrieval_v2_result is not None:
                sequence_results, raw_results = self._validate_v2_trake_outputs(
                    sequence_results,
                    raw_results,
                    event_ids,
                    structured_query.temporal_constraints,
                    warnings,
                )
                trake_result = self._synchronize_trake_result(
                    trake_result,
                    raw_results,
                    event_ids,
                )
            api_results = sequence_results

        verification, verification_warnings = await self._verify_results(
            structured_query,
            raw_results,
            ranked_candidates,
        )
        warnings.extend(verification_warnings)
        if verification.applied:
            rejected = {
                index
                for index, item in enumerate(verification.items)
                if item.result.status == "rejected"
            }
            if rejected:
                api_results = [
                    item for index, item in enumerate(api_results) if index not in rejected
                ]
                raw_results = [
                    item for index, item in enumerate(raw_results) if index not in rejected
                ]
                warnings.append(
                    f"Selective verification rejected {len(rejected)} output candidate(s)."
                )

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
            retrieval_v2_plan=(
                retrieval_v2_result.plan if retrieval_v2_result is not None else None
            ),
            retrieval_v2_session=(
                retrieval_v2_result.session if retrieval_v2_result is not None else None
            ),
            warnings=warnings,
            answer=qa_result.answer if qa_result is not None else None,
            answer_status=qa_result.status if qa_result is not None else None,
        )

    @staticmethod
    def _hits_from_v2_result(result: SearchControllerResult) -> list[SearchHit]:
        unique: dict[tuple[str, str, str | None], SearchHit] = {}
        for band in result.bands:
            for hit in band.hits:
                unique[(hit.entity_type, hit.entity_id, hit.atom_id)] = hit
        return list(unique.values())

    def _select_v2_official_frames(
        self,
        result: SearchControllerResult,
        limit: int,
        warnings: list[str],
    ) -> tuple[list[VisualResult], list[KISResult]]:
        moment_confidence = {
            band_id: hypothesis.moment_confidence
            for hypothesis in result.hypotheses
            for band_id in hypothesis.band_ids
        }
        for band in result.reranked_bands:
            moment_confidence.setdefault(
                band.band_id,
                1.0 - math.exp(-max(0.0, band.score)),
            )
        selected = self.official_frame_selector.select(
            bands=result.reranked_bands,
            atoms=result.plan.atoms,
            provider=DatabaseOfficialFrameProvider(self.db_mng),
            reviews=result.session.reviews,
            moment_confidence=moment_confidence,
            limit=limit,
        )
        visual_results: list[VisualResult] = []
        raw_results: list[KISResult] = []
        bands_by_id = {band.band_id: band for band in result.reranked_bands}
        for item in selected:
            frame = item.frame
            band = bands_by_id[item.source_band_id]
            if frame.frame_id is None or frame.img_url is None:
                warnings.append(
                    f"Official frame metadata is incomplete for {frame.video_id}:{frame.frame_idx}."
                )
                continue
            raw = KISResult(
                video_id=frame.video_id,
                start_ms=band.start_ms,
                end_ms=band.end_ms,
                representative_frame_id=frame.frame_id,
                score=1.0 - math.exp(-max(0.0, item.score)),
                evidence_ids=list(dict.fromkeys(hit.entity_id for hit in band.hits)),
            )
            raw_results.append(raw)
            visual_results.append(
                VisualResult(
                    **raw.model_dump(),
                    display_frame_id=frame.frame_id,
                    frame_idx=frame.frame_idx,
                    img_url=frame.img_url,
                )
            )
        return visual_results, raw_results

    async def _answer_vqa(
        self,
        query: StructuredQuery,
        frames: Sequence[VisualResult],
        retrieval_result: SearchControllerResult | None,
    ) -> GroundedQAResult:
        if retrieval_result is None:
            return GroundedQAResult(
                answer="uncertain",
                status="uncertain",
                evidence_ids=(),
            )
        evidence_source = (
            "ocr"
            if query.ocr_constraints
            else "asr"
            if query.asr_constraints
            else self._qa_evidence_source(query.question)
        )
        bands_by_frame = [
            [
                band
                for band in retrieval_result.reranked_bands
                if band.video_id == frame.video_id
                and band.start_ms <= frame.end_ms
                and band.end_ms >= frame.start_ms
            ]
            for frame in frames
        ]
        allowed_by_frame = [
            {
                frame.display_frame_id,
                *(hit.entity_id for band in bands for hit in band.hits),
            }
            for frame, bands in zip(frames, bands_by_frame, strict=True)
        ]
        all_allowed = set().union(*allowed_by_frame) if allowed_by_frame else set()
        provider_claims: Sequence[AnswerClaim] = ()
        if self.qa_answer_provider is not None:
            try:
                provider_claims = await self.qa_answer_provider.answer_claims(
                    query.question,
                    frames,
                    all_allowed,
                )
            except Exception:
                logger.exception("Grounded VQA provider failed")
        seen_evidence: list[str] = []
        for bands, allowed in zip(bands_by_frame, allowed_by_frame, strict=True):
            claims = [
                AnswerClaim(
                    evidence_id=hit.entity_id,
                    answer=hit.text_content,
                    confidence=max(0.0, min(1.0, hit.raw_score)),
                )
                for band in bands
                for hit in band.hits
                if hit.text_content and hit.entity_type == evidence_source
            ]
            claims.extend(
                claim for claim in provider_claims if claim.evidence_id in allowed
            )
            answer = aggregate_grounded_answer(claims, allowed)
            seen_evidence.extend(answer.evidence_ids)
            if answer.status == "answered":
                return answer
        return GroundedQAResult(
            answer="uncertain",
            status="uncertain",
            evidence_ids=tuple(dict.fromkeys(seen_evidence)),
        )

    @staticmethod
    def _qa_evidence_source(question: str) -> str | None:
        normalized = question.casefold()
        if any(token in normalized for token in ("written", "text", "sign", "read", "ghi", "viết", "chữ")):
            return "ocr"
        if any(token in normalized for token in ("said", "says", "speaker", "nói", "phát biểu")):
            return "asr"
        return None

    @staticmethod
    def _tool_calls_from_v2_result(result: SearchControllerResult) -> list[ToolCall]:
        return [
            ToolCall(
                tool_call_id=call.call_id,
                tool_name=call.retriever,
                event_id=call.event_id,
                parameters={
                    "query": call.query,
                    "top_k": call.top_k,
                    "video_ids": call.video_ids,
                    "atom_id": call.atom_id,
                    "prompt_role": call.prompt_role,
                    "min_count": call.min_count,
                },
            )
            for round_ in result.session.rounds
            for call in round_.calls
        ]

    @staticmethod
    def _ranked_candidates_from_v2_result(
        result: SearchControllerResult,
    ) -> list[RankedCandidateRegion]:
        ranked: list[RankedCandidateRegion] = []
        for band in result.reranked_bands:
            evidence = [
                CandidateEvidence(
                    source=hit.source,
                    entity_type=hit.entity_type,
                    entity_id=hit.entity_id,
                    start_ms=hit.start_ms,
                    end_ms=hit.end_ms,
                    rank=hit.rank,
                    raw_score=hit.raw_score,
                    tool_call_id=hit.tool_call_id,
                    atom_id=hit.atom_id,
                    prompt_role=hit.prompt_role,
                    retriever_family=hit.retriever_family,
                    text_content=hit.text_content,
                )
                for hit in band.hits
            ]
            has_fail = any(
                decision.status == "FAIL" for decision in band.constraint_decisions
            )
            has_negative_fail = any(
                decision.status == "FAIL"
                and decision.constraint_id.startswith("negative:")
                for decision in band.constraint_decisions
            )
            ranked.append(
                RankedCandidateRegion(
                    candidate_id=band.band_id,
                    event_id=band.event_id,
                    video_id=band.video_id,
                    start_ms=band.start_ms,
                    end_ms=band.end_ms,
                    fusion_score=band.score,
                    constraint_result=ConstraintResult(
                        hard_constraints_passed=not has_fail,
                        negative_constraints_passed=not has_negative_fail,
                    ),
                    evidence=evidence,
                )
            )
        return ranked

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
        event_descriptions: dict[str, str] | None = None,
    ) -> tuple[list[TrakeSequenceResult], list[TemporalSequence]]:
        enriched: list[TrakeSequenceResult] = []
        retained: list[TemporalSequence] = []
        for sequence in sequences:
            events: list[TrakeEventResult] = []
            displays = self._resolve_monotonic_trake_frames(
                sequence,
                event_descriptions=event_descriptions,
            )
            if displays is None:
                warnings.append(
                    "Skipped TRAKE sequence because no strictly increasing "
                    "official-frame assignment exists."
                )
                continue
            for event, display in zip(sequence.events, displays, strict=True):
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

    def _resolve_monotonic_trake_frames(
        self,
        sequence: TemporalSequence,
        *,
        event_descriptions: dict[str, str] | None = None,
    ) -> list[ResolvedFrame] | None:
        try:
            records = self.db_mng.get_frame_record_by_video_id(sequence.video_id)
        except ValueError:
            records = []
        official = [
            frame
            for frame in records
            if frame.source == "official" and frame.frame_path is not None
        ]
        options: list[list[FrameMetadata]] = []
        for event in sequence.events:
            in_band = [
                frame
                for frame in official
                if event.start_ms <= frame.timestamp_ms <= event.end_ms
            ]
            if not in_band:
                nearest = self.db_mng.get_nearest_official_frame(
                    sequence.video_id,
                    (event.start_ms + event.end_ms) // 2,
                )
                if nearest is None or nearest.frame_path is None:
                    return None
                midpoint = (event.start_ms + event.end_ms) // 2
                if abs(nearest.timestamp_ms - midpoint) > 5_000:
                    return None
                in_band = [nearest]
            options.append(in_band)

        states: dict[int, tuple[int, list[FrameMetadata]]] = {}
        for frame in options[0] if options else []:
            target = self._trake_event_target_ms(
                sequence.events[0],
                event_descriptions or {},
            )
            states[frame.frame_idx] = (abs(frame.timestamp_ms - target), [frame])
        for event, candidates in zip(sequence.events[1:], options[1:], strict=True):
            target = self._trake_event_target_ms(event, event_descriptions or {})
            next_states: dict[int, tuple[int, list[FrameMetadata]]] = {}
            for frame in candidates:
                predecessors = [
                    value
                    for frame_idx, value in states.items()
                    if frame_idx < frame.frame_idx
                ]
                if not predecessors:
                    continue
                prior_cost, prior_path = min(
                    predecessors,
                    key=lambda item: (item[0], tuple(f.frame_idx for f in item[1])),
                )
                next_states[frame.frame_idx] = (
                    prior_cost + abs(frame.timestamp_ms - target),
                    [*prior_path, frame],
                )
            states = next_states
            if not states:
                return None
        if not states:
            return None
        _, selected = min(
            states.values(),
            key=lambda item: (item[0], tuple(frame.frame_idx for frame in item[1])),
        )
        return [FrameResolver._as_resolved(frame) for frame in selected]

    @staticmethod
    def _trake_event_target_ms(event, descriptions: dict[str, str]) -> int:
        description = descriptions.get(event.event_id, "").casefold()
        if any(token in description for token in ("first", "earliest", "initial")):
            return event.start_ms
        if any(token in description for token in ("completes", "completed", "finishes", "finished")):
            return event.end_ms
        return (event.start_ms + event.end_ms) // 2

    def _validate_v2_trake_outputs(
        self,
        enriched: list[TrakeSequenceResult],
        retained: list[TemporalSequence],
        event_ids: list[str],
        temporal_constraints,
        warnings: list[str],
    ) -> tuple[list[TrakeSequenceResult], list[TemporalSequence]]:
        valid_enriched: list[TrakeSequenceResult] = []
        valid_retained: list[TemporalSequence] = []
        for api_sequence, raw_sequence in zip(enriched, retained, strict=True):
            interval_reasons = self._trake_interval_violations(
                raw_sequence,
                temporal_constraints,
            )
            if interval_reasons:
                warnings.append(
                    "Rejected TRAKE sequence at interval gate: "
                    + ",".join(interval_reasons)
                )
                continue
            frames: list[OfficialFrame] = []
            for event in api_sequence.events:
                try:
                    metadata = self.db_mng.get_frame_record_by_frame_id(
                        event.display_frame_id
                    )
                except ValueError:
                    frames = []
                    break
                frames.append(
                    OfficialFrame(
                        evidence_id=event.display_frame_id,
                        video_id=api_sequence.video_id,
                        frame_idx=event.frame_idx,
                        timestamp_ms=metadata.timestamp_ms,
                        event_id=event.event_id,
                        official=metadata.source == "official",
                        score=event.fusion_score or 0.0,
                    )
                )
            decision = validate_trake_sequence(
                frames,
                event_ids,
                temporal_constraints=(),
            )
            if not decision.valid:
                warnings.append(
                    "Rejected TRAKE sequence at task gate: "
                    + ",".join(decision.reasons)
                )
                continue
            valid_enriched.append(api_sequence)
            valid_retained.append(raw_sequence)
        return valid_enriched, valid_retained

    @staticmethod
    def _synchronize_trake_result(trake_result, sequences, event_ids):
        return trake_result.model_copy(
            update={
                "sequences": sequences,
                "status": "success" if sequences else "no_valid_sequence",
                "replan_required": not bool(sequences),
                "missing_event_ids": [] if sequences else list(event_ids),
            }
        )

    @staticmethod
    def _trake_interval_violations(sequence, constraints) -> list[str]:
        by_event = {event.event_id: event for event in sequence.events}
        reasons: list[str] = []
        for constraint in constraints:
            before = by_event.get(constraint.before)
            after = by_event.get(constraint.after)
            if before is None or after is None:
                continue
            gap_ms = after.start_ms - before.end_ms
            if gap_ms < 0 and not constraint.allow_overlap:
                reasons.append(f"OVERLAP_NOT_ALLOWED:{constraint.before}:{constraint.after}")
            if constraint.min_gap_ms is not None and gap_ms < constraint.min_gap_ms:
                reasons.append(f"MIN_GAP_VIOLATION:{constraint.before}:{constraint.after}")
            if constraint.max_gap_ms is not None and gap_ms > constraint.max_gap_ms:
                reasons.append(f"MAX_GAP_VIOLATION:{constraint.before}:{constraint.after}")
        return reasons

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
                    reason="grounded_vqa_answer_not_reverified",
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

    async def _trake_fallback_second_pass(
        self,
        query: StructuredQuery,
        ranked_candidates: list[RankedCandidateRegion],
        existing_hits: list[SearchHit],
        event_ids: list[str],
        missing_event_ids: list[str] | None = None,
    ) -> tuple[list[RankedCandidateRegion], list[SearchHit]]:
        """Second-pass retrieval to rescue missing events on candidate videos."""
        candidate_video_ids = {c.video_id for c in ranked_candidates if c.video_id}
        if not candidate_video_ids or not query.events:
            return ranked_candidates, existing_hits

        target_events = [
            e for e in query.events
            if missing_event_ids is None or e.event_id in missing_event_ids
        ]
        if not target_events:
            target_events = query.events

        from BackEnd.app.retrieval_tools.visual import clip_search
        import asyncio

        tasks = [
            clip_search(
                query=event.description,
                top_k=500,
                event_id=event.event_id,
            )
            for event in target_events
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        extra_hits: list[SearchHit] = []
        for res in results:
            if isinstance(res, list):
                filtered = [h for h in res if h.video_id in candidate_video_ids]
                extra_hits.extend(filtered)

        if not extra_hits:
            return ranked_candidates, existing_hits

        combined_hits = existing_hits + extra_hits
        candidate_regions = self.aggregator.execute(
            combined_hits,
            merge_gap=CANDIDATE_MERGE_GAP_MS,
        )
        new_ranked_candidates = self.fusion.fusion_and_ranking(
            candidate_regions,
            self._fusion_weight_flags(combined_hits),
            query,
        )
        return new_ranked_candidates, combined_hits

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
