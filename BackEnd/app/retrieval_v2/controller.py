from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.cache import QueryEmbeddingCache
from BackEnd.app.retrieval_v2.constraints import HardConstraintEngine
from BackEnd.app.retrieval_v2.video_index import VideoLevelIndex
from BackEnd.app.retrieval_v2.adapters.reranker import CandidateReranker
from BackEnd.app.retrieval_v2.prompt_planner import CorpusAwarePromptPlanner
from BackEnd.app.retrieval_v2.corpus_stats import CorpusStats
from BackEnd.app.retrieval_v2.metrics import RetrievalMetricsCollector
from BackEnd.app.retrieval_v2.contracts import (
    CandidateBudget,
    PromptVariant,
    QueryAtom,
    RetrievalPlan,
    SearchCall,
    SearchControllerResult,
    SearchRound,
    SearchSessionState,
    VideoHypothesis,
)
from BackEnd.app.retrieval_v2.moment_bands import build_moment_bands
from BackEnd.app.retrieval_v2.planning import build_retrieval_plan
from BackEnd.app.retrieval_v2.reservoir import select_candidate_reservoir
from BackEnd.app.retrieval_v2.ranking import (
    build_video_hypotheses,
    diagnose_hypotheses,
    rerank_bands,
)
from BackEnd.app.retrieval_v2.allocation import allocate_submission_bands
from BackEnd.app.retrieval_v2.sequence import collapse_kis_sequences
from BackEnd.app.retrieval_v2.review import (
    CandidateReviewer,
    apply_candidate_reviews,
    diagnose_candidate_reviews,
)


logger = logging.getLogger(__name__)


class SearchGateway(Protocol):
    async def search(self, call: SearchCall) -> list[SearchHit]: ...

    async def search_many(self, calls: list[SearchCall]) -> list[list[SearchHit]]: ...


_ROLE_RETRIEVER = {
    "global": "clip_search",
    "rare_detail": "frame_search",
    "action": "clip_search",
    "context": "shot_search",
    "contrast": "frame_search",
}
_MODALITY_RETRIEVER = {
    "ocr": "ocr_search",
    "asr": "asr_search",
    "object": "object_search",
}


class SearchController:
    """Bounded global -> local -> diagnosed retry retrieval loop."""

    def __init__(
        self,
        gateway: SearchGateway,
        *,
        budget: CandidateBudget | None = None,
        constraint_engine: HardConstraintEngine | None = None,
        reviewer: CandidateReviewer | None = None,
        reranker: CandidateReranker | None = None,
        video_index: VideoLevelIndex | None = None,
        video_query_encoder: Callable[[str], Any] | None = None,
        cache: QueryEmbeddingCache | None = None,
        prompt_planner: CorpusAwarePromptPlanner | None = None,
        metrics_collector: RetrievalMetricsCollector | None = None,
    ) -> None:
        self.gateway = gateway
        self.budget = budget or CandidateBudget()
        self.constraint_engine = constraint_engine or HardConstraintEngine()
        self.reviewer = reviewer
        self.reranker = reranker
        self.video_index = video_index
        self.video_query_encoder = video_query_encoder
        self.cache = cache or QueryEmbeddingCache()
        self.prompt_planner = prompt_planner or CorpusAwarePromptPlanner()
        self.metrics_collector = metrics_collector

    async def search(self, query: StructuredQuery) -> SearchControllerResult:
        started_at = time.perf_counter()
        session_metrics = (
            self.metrics_collector.start_session(query.query_id, query.task)
            if self.metrics_collector is not None
            else None
        )
        plan = build_retrieval_plan(query, prompt_planner=self.prompt_planner)
        coverage_atom_ids = [
            atom.atom_id for atom in plan.atoms if atom.operator != "MUST_NOT"
        ]
        rounds: list[SearchRound] = []
        all_raw_hits: list[SearchHit] = []
        diagnosis_history = []
        review_history = []
        stop_reason = "COMPLETED"

        global_calls = self._global_calls(plan)
        global_hits = await self._run_calls(global_calls)
        all_raw_hits.extend(global_hits)
        rounds.append(SearchRound(
            round_index=0,
            phase="GLOBAL",
            calls=global_calls,
            hit_count=len(global_hits),
            requested_k=sum(call.top_k for call in global_calls),
            unique_candidate_count=self._canonical_candidate_count(global_hits),
            new_video_gain=len({hit.video_id for hit in global_hits}),
            new_moment_gain=self._canonical_candidate_count(global_hits),
        ))

        # Requested top-k is only an upper bound. Correlated prompts can return
        # the same frame repeatedly, so deepen while each wave still adds new
        # canonical moments and the raw safety ceiling has not been reached.
        global_depth = sum(call.top_k for call in global_calls)
        unique_before = self._canonical_candidate_count(all_raw_hits)
        global_wave = 1
        while (
            global_hits
            and unique_before < self.budget.unique_candidate_min
            and global_depth < self.budget.raw_retrieval_max
        ):
            # FAISS has no cursor. A retry at the same/smaller top_k only
            # returns the old prefix, so every wave requests a strictly deeper
            # cumulative prefix and deduplication keeps only newly exposed hits.
            next_depth = min(
                self.budget.raw_retrieval_max,
                global_depth + self.budget.raw_retrieval_target,
            )
            deepen_calls = self._global_calls(
                plan,
                total_k=next_depth,
                prefix=f"global_deepen_{global_wave:02d}",
            )
            deepen_hits = await self._run_calls(deepen_calls)
            if not deepen_calls:
                break
            all_raw_hits.extend(deepen_hits)
            global_depth = next_depth
            unique_after = self._canonical_candidate_count(all_raw_hits)
            rounds[0] = rounds[0].model_copy(
                update={
                    "calls": [*rounds[0].calls, *deepen_calls],
                    "hit_count": rounds[0].hit_count + len(deepen_hits),
                    "requested_k": rounds[0].requested_k + sum(call.top_k for call in deepen_calls),
                    "unique_candidate_count": unique_after,
                    "new_video_gain": len({hit.video_id for hit in all_raw_hits}),
                    "new_moment_gain": unique_after,
                }
            )
            global_hits = deepen_hits
            if unique_after <= unique_before:
                break
            unique_before = unique_after
            global_wave += 1

        plan = self._calibrate_plan_selectivity(plan, all_raw_hits)
        deduplicated = self._limit_canonical_candidates(self._deduplicate(all_raw_hits), plan.atoms)
        bands = self._bands(deduplicated, coverage_atom_ids, plan.atoms)
        moment_hypotheses = build_video_hypotheses(
            bands,
            plan.atoms,
            self.budget.video_shortlist_limit,
        )
        video_hypotheses = await self._video_lane_hypotheses(plan)
        hypotheses = self._merge_hypothesis_lanes(moment_hypotheses, video_hypotheses)

        local_calls = self._local_calls(plan, hypotheses, bands)
        before_local_keys = self._canonical_candidate_keys(all_raw_hits)
        local_hits = await self._run_calls(local_calls)
        all_raw_hits.extend(local_hits)
        after_local_keys = self._canonical_candidate_keys(all_raw_hits)
        rounds.append(SearchRound(
            round_index=len(rounds),
            phase="LOCAL",
            calls=local_calls,
            hit_count=len(local_hits),
            requested_k=sum(call.top_k for call in local_calls),
            unique_candidate_count=len(after_local_keys),
            new_video_gain=len({key[0] for key in after_local_keys} - {key[0] for key in before_local_keys}),
            new_moment_gain=len(after_local_keys - before_local_keys),
        ))

        deduplicated = self._limit_canonical_candidates(self._deduplicate(all_raw_hits), plan.atoms)
        bands = self._bands(deduplicated, coverage_atom_ids, plan.atoms)
        hypotheses = self._merge_hypothesis_lanes(
            build_video_hypotheses(
                bands,
                plan.atoms,
                self.budget.video_shortlist_limit,
            ),
            video_hypotheses,
        )
        preliminary = self._rank_for_task(plan, bands, hypotheses)
        preliminary = await self._cheap_rerank(plan, preliminary)
        reviews = await self._review(plan, preliminary)
        review_history.extend(reviews)
        reranked = self._apply_candidate_gates(
            apply_candidate_reviews(preliminary, reviews, plan.atoms),
            plan.atoms,
        )
        diagnoses = diagnose_hypotheses(hypotheses, plan.atoms)
        diagnoses.extend(diagnose_candidate_reviews(reviews))
        diagnosis_history.extend(diagnoses)

        for retry_index in range(1, self.budget.max_retry_rounds + 1):
            if not diagnoses:
                stop_reason = "SUFFICIENT_EVIDENCE"
                break
            all_raw_hits = self._reject_diagnosed_scopes(
                all_raw_hits,
                bands,
                diagnoses,
            )
            # Rejection mutates the candidate corpus. Materialize immediately
            # so a no-op retry cannot leak a previously rejected band.
            deduplicated = self._limit_canonical_candidates(self._deduplicate(all_raw_hits), plan.atoms)
            bands = self._bands(deduplicated, coverage_atom_ids, plan.atoms)
            hypotheses = self._merge_hypothesis_lanes(
                build_video_hypotheses(
                    bands,
                    plan.atoms,
                    self.budget.video_shortlist_limit,
                ),
                video_hypotheses,
            )
            preliminary = self._rank_for_task(plan, bands, hypotheses)
            preliminary = await self._cheap_rerank(plan, preliminary)
            reranked = self._apply_candidate_gates(
                apply_candidate_reviews(preliminary, review_history, plan.atoms),
                plan.atoms,
            )
            retry_calls = self._retry_calls(
                plan,
                diagnoses,
                retry_index=retry_index,
            )
            if not retry_calls:
                stop_reason = "NO_RETRY_ACTION"
                break
            before_retry_keys = self._canonical_candidate_keys(all_raw_hits)
            retry_hits = await self._run_calls(retry_calls)
            all_raw_hits.extend(retry_hits)
            after_retry_keys = self._canonical_candidate_keys(all_raw_hits)
            rounds.append(
                SearchRound(
                    round_index=len(rounds),
                    phase="RETRY",
                    calls=retry_calls,
                    hit_count=len(retry_hits),
                    requested_k=sum(call.top_k for call in retry_calls),
                    unique_candidate_count=len(after_retry_keys),
                    new_video_gain=len({key[0] for key in after_retry_keys} - {key[0] for key in before_retry_keys}),
                    new_moment_gain=len(after_retry_keys - before_retry_keys),
                )
            )
            deduplicated = self._limit_canonical_candidates(self._deduplicate(all_raw_hits), plan.atoms)
            bands = self._bands(deduplicated, coverage_atom_ids, plan.atoms)
            hypotheses = self._merge_hypothesis_lanes(
                build_video_hypotheses(
                    bands,
                    plan.atoms,
                    self.budget.video_shortlist_limit,
                ),
                video_hypotheses,
            )
            preliminary = self._rank_for_task(plan, bands, hypotheses)
            preliminary = await self._cheap_rerank(plan, preliminary)
            reviews = await self._review(plan, preliminary)
            review_history.extend(reviews)
            reranked = self._apply_candidate_gates(
                apply_candidate_reviews(preliminary, review_history, plan.atoms),
                plan.atoms,
            )
            diagnoses = diagnose_hypotheses(hypotheses, plan.atoms)
            diagnoses.extend(diagnose_candidate_reviews(reviews))
            diagnosis_history.extend(diagnoses)
            if not retry_hits or after_retry_keys == before_retry_keys:
                stop_reason = "NO_CANDIDATE_GAIN"
                break
        else:
            if diagnoses and self.budget.max_retry_rounds > 0:
                stop_reason = "MAX_RETRIES"

        session = SearchSessionState(
            query_id=query.query_id,
            rounds=rounds,
            diagnoses=diagnosis_history,
            raw_hit_count=len(all_raw_hits),
            deduplicated_hit_count=len(deduplicated),
            hypotheses=hypotheses,
            reviews=review_history,
            stop_reason=stop_reason,
        )
        result = SearchControllerResult(
            plan=plan,
            bands=bands,
            reranked_bands=reranked,
            hypotheses=hypotheses,
            session=session,
        )
        if session_metrics is not None:
            session_metrics.total_latency_ms = (time.perf_counter() - started_at) * 1000.0
            session_metrics.raw_hit_count = session.raw_hit_count
            session_metrics.dedup_hit_count = session.deduplicated_hit_count
            session_metrics.moment_band_count = len(bands)
            session_metrics.video_hypothesis_count = len(hypotheses)
            session_metrics.round_count = len(rounds)
            session_metrics.retry_reasons = [item.reason for item in diagnosis_history]
            session_metrics.cache_hit = self.cache.hits > 0
        return result

    def _rank_for_task(self, plan, bands, hypotheses):
        positive_required_ids = {
            atom.atom_id
            for atom in plan.atoms
            if atom.required and atom.operator == "MUST"
        }
        task_bands = [
            band
            for band in bands
            if not positive_required_ids
            or any(
                band.coverage.get(atom_id) is not None
                and band.coverage[atom_id].retrieval_status == "RETRIEVED"
                for atom_id in positive_required_ids
            )
        ]
        if plan.execution_profile == "KIS_SEQUENCE":
            event_ids = list(dict.fromkeys(atom.event_id for atom in plan.atoms if atom.event_id))
            complete_sequences = collapse_kis_sequences(
                bands,
                event_ids,
                temporal_constraints=plan.temporal_constraints,
                limit=self.budget.moment_band_limit,
            )
            complete_videos = {band.video_id for band in complete_sequences}
            partial_limit = max(1, self.budget.moment_band_limit // 5)
            partial_rescue = [
                band.model_copy(update={
                    "score": 0.60 * band.score,
                    "score_breakdown": {
                        **band.score_breakdown,
                        "partial_sequence_penalty": 0.40,
                    },
                })
                for band in rerank_bands(bands, plan.atoms, self.budget.moment_band_limit)
                if band.video_id not in complete_videos
            ][:partial_limit]
            task_bands = [*complete_sequences, *partial_rescue]
        cheap_reranked = rerank_bands(
            task_bands,
            plan.atoms,
            self.budget.moment_band_limit,
        )
        return allocate_submission_bands(
            cheap_reranked,
            hypotheses,
            limit=self.budget.rerank_limit,
        )

    def _global_calls(
        self,
        plan: RetrievalPlan,
        *,
        total_k: int | None = None,
        prefix: str = "global",
    ) -> list[SearchCall]:
        specs: list[tuple[QueryAtom, PromptVariant | None, str, float]] = []
        for atom in plan.atoms:
            if atom.modality == "visual":
                for prompt in atom.prompt_variants:
                    specs.append(
                        (
                            atom,
                            prompt,
                            _ROLE_RETRIEVER[prompt.role],
                            prompt.weight * atom.discriminative_weight,
                        )
                    )
            else:
                specs.append(
                    (atom, None, _MODALITY_RETRIEVER[atom.modality], atom.discriminative_weight)
                )
        return self._allocate_calls(
            specs,
            total_k or self.budget.raw_retrieval_target,
            prefix,
        )

    def _local_calls(self, plan: RetrievalPlan, hypotheses, bands) -> list[SearchCall]:
        video_ids = [hypothesis.video_id for hypothesis in hypotheses]
        if not video_ids:
            return []
        specs: list[tuple[QueryAtom, PromptVariant | None, str, float]] = []
        for atom in plan.atoms:
            prompt = self._preferred_local_prompt(atom)
            retriever = (
                _ROLE_RETRIEVER[prompt.role]
                if prompt is not None
                else _MODALITY_RETRIEVER[atom.modality]
            )
            specs.append((atom, prompt, retriever, atom.discriminative_weight))
        max_specs = max(1, self.budget.local_retrieval_k // max(1, len(video_ids) * 2))
        if len(specs) > max_specs:
            protected: list[tuple[QueryAtom, PromptVariant | None, str, float]] = []
            for scope in ("VIDEO_ANCHOR", "EVENT", "ANSWER_EVIDENCE"):
                scope_specs = [spec for spec in specs if spec[0].scope == scope]
                by_event: dict[str, tuple[QueryAtom, PromptVariant | None, str, float]] = {}
                for spec in sorted(scope_specs, key=lambda item: item[3], reverse=True):
                    event_key = spec[0].event_id or scope
                    by_event.setdefault(event_key, spec)
                protected.extend(by_event.values())
            protected_ids = {spec[0].atom_id for spec in protected}
            remainder = sorted(
                (spec for spec in specs if spec[0].atom_id not in protected_ids),
                key=lambda spec: spec[3],
                reverse=True,
            )
            specs = [*protected, *remainder][:max_specs]
        base_quota, extra = divmod(self.budget.local_retrieval_k, len(video_ids))
        band_by_id = {band.band_id: band for band in bands}
        hypothesis_by_video = {item.video_id: item for item in hypotheses}
        calls: list[SearchCall] = []
        for video_index, video_id in enumerate(video_ids):
            video_quota = base_quota + (1 if video_index < extra else 0)
            if video_quota <= 0:
                continue
            hypothesis = hypothesis_by_video[video_id]
            scoped_bands = [
                band_by_id[band_id]
                for band_id in hypothesis.band_ids
                if band_id in band_by_id
            ]
            local_calls = self._allocate_calls(
                specs,
                video_quota,
                f"local_{video_index + 1:03d}",
                video_ids=[video_id],
            )
            if not scoped_bands or hypothesis.moment_confidence < 0.55:
                calls.extend(local_calls)
                continue

            margin_ms = 60_000 if hypothesis.moment_confidence < 0.80 else 15_000
            atom_by_id = {atom.atom_id: atom for atom in plan.atoms}
            for call in local_calls:
                event_id = atom_by_id[call.atom_id].event_id
                event_bands = [
                    band
                    for band in scoped_bands
                    if event_id is None or band.event_id in {None, event_id}
                ] or scoped_bands
                calls.append(call.model_copy(update={
                    "start_ms": max(0, min(band.start_ms for band in event_bands) - margin_ms),
                    "end_ms": max(band.end_ms for band in event_bands) + margin_ms,
                }))
        return calls

    def _retry_calls(
        self,
        plan: RetrievalPlan,
        diagnoses,
        *,
        retry_index: int,
    ) -> list[SearchCall]:
        local_video_ids = list(
            dict.fromkeys(
                diagnosis.video_id
                for diagnosis in diagnoses
                if diagnosis.video_id
                and diagnosis.reason
                in {
                    "LOW_MOMENT_CONFIDENCE",
                    "WRONG_MOMENT",
                    "MISSING_ACTION",
                    "WRONG_RELATION_OR_COUNT",
                }
            )
        )

        def specs_for(video_id: str | None):
            relevant = [
                diagnosis
                for diagnosis in diagnoses
                if video_id is None or diagnosis.video_id in {None, video_id}
            ]
            weak_atom_ids = list(dict.fromkeys(
                diagnosis.atom_id for diagnosis in relevant if diagnosis.atom_id
            ))
            retry_atoms = [atom for atom in plan.atoms if atom.atom_id in weak_atom_ids]
            if not retry_atoms:
                retry_atoms = sorted(
                    (atom for atom in plan.atoms if atom.operator != "MUST_NOT"),
                    key=lambda atom: atom.discriminative_weight,
                    reverse=True,
                )[:1]
            retry_atoms = sorted(
                retry_atoms,
                key=lambda atom: atom.discriminative_weight,
                reverse=True,
            )[:3]
            resolved = []
            for atom in retry_atoms:
                prompt = self._diagnostic_retry_prompt(atom, relevant)
                retriever = (
                    _ROLE_RETRIEVER[prompt.role]
                    if prompt is not None
                    else _MODALITY_RETRIEVER[atom.modality]
                )
                resolved.append((atom, prompt, retriever, atom.discriminative_weight))
            return resolved

        prefix = f"retry_{retry_index:02d}"
        if not local_video_ids:
            return self._allocate_calls(
                specs_for(None),
                self.budget.retry_retrieval_k,
                prefix,
            )

        base_quota, extra = divmod(self.budget.retry_retrieval_k, len(local_video_ids))
        calls: list[SearchCall] = []
        for video_index, video_id in enumerate(local_video_ids):
            quota = base_quota + (1 if video_index < extra else 0)
            if quota <= 0:
                continue
            calls.extend(self._allocate_calls(
                specs_for(video_id),
                quota,
                f"{prefix}_{video_index + 1:03d}",
                video_ids=[video_id],
            ))
        return calls

    async def _review(self, plan: RetrievalPlan, bands):
        if self.reviewer is None or not bands:
            return []
        return await self.reviewer.review(
            plan,
            bands[: self.budget.review_limit],
        )

    async def _cheap_rerank(self, plan: RetrievalPlan, bands):
        if self.reranker is None or not bands:
            return bands
        return await self.reranker.rerank(
            bands[: max(self.budget.rerank_limit, self.budget.review_limit)],
            plan.atoms,
            self.budget.rerank_limit,
        )

    async def _video_lane_hypotheses(self, plan: RetrievalPlan):
        if self.video_index is None or self.video_query_encoder is None:
            return []
        visual_atoms = sorted(
            (atom for atom in plan.atoms if atom.modality == "visual" and atom.operator != "MUST_NOT"),
            key=lambda atom: atom.discriminative_weight,
            reverse=True,
        )[:8]
        scores: dict[str, float] = {}
        for atom in visual_atoms:
            prompt = self._preferred_local_prompt(atom)
            text = prompt.text if prompt is not None else atom.text
            normalized = " ".join(text.casefold().split())
            vector = self.cache.get("video_level_v1", normalized)
            if vector is None:
                vector = await asyncio.to_thread(self.video_query_encoder, text)
                self.cache.put("video_level_v1", normalized, vector)
            for video_id, score in self.video_index.search(
                vector,
                self.budget.video_shortlist_limit,
            ):
                scores[video_id] = max(scores.get(video_id, -1.0), score)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            VideoHypothesis(
                video_id=video_id,
                video_confidence=max(0.0, min(1.0, (score + 1.0) / 2.0)),
                moment_confidence=0.0,
                coverage={},
                band_ids=[],
                lane_sources=["video"],
            )
            for video_id, score in ordered[: self.budget.video_shortlist_limit]
        ]

    def _merge_hypothesis_lanes(self, moment_hypotheses, video_hypotheses):
        merged = {hypothesis.video_id: hypothesis for hypothesis in moment_hypotheses}
        for hypothesis in video_hypotheses:
            current = merged.get(hypothesis.video_id)
            if current is None:
                merged[hypothesis.video_id] = hypothesis
                continue
            merged[hypothesis.video_id] = current.model_copy(
                update={
                    "video_confidence": max(
                        current.video_confidence,
                        hypothesis.video_confidence,
                    ),
                    "lane_sources": list(
                        dict.fromkeys([*current.lane_sources, *hypothesis.lane_sources])
                    ),
                }
            )
        moment_order = [item.video_id for item in moment_hypotheses]
        video_only = sorted(
            (
                item for item in merged.values() if item.video_id not in set(moment_order)
            ),
            key=lambda item: (-item.video_confidence, item.video_id),
        )
        if not video_only:
            return [merged[video_id] for video_id in moment_order][
                : self.budget.video_shortlist_limit
            ]
        # Reserve at least 25% of the shortlist for the independent video lane;
        # otherwise a noisy moment lane can starve its recall-rescue purpose.
        reserve = max(1, self.budget.video_shortlist_limit // 4)
        moment_limit = max(0, self.budget.video_shortlist_limit - reserve)
        ordered = [merged[video_id] for video_id in moment_order[:moment_limit]]
        ordered.extend(video_only[:reserve])
        if len(ordered) < self.budget.video_shortlist_limit:
            ordered.extend(
                merged[video_id]
                for video_id in moment_order[moment_limit:]
                if video_id not in {item.video_id for item in ordered}
            )
        return ordered[: self.budget.video_shortlist_limit]

    @staticmethod
    def _reject_diagnosed_scopes(all_hits, bands, diagnoses):
        rejected_videos = {
            diagnosis.video_id
            for diagnosis in diagnoses
            if diagnosis.reason == "WRONG_VIDEO" and diagnosis.video_id
        }
        rejected_band_ids = {
            diagnosis.band_id
            for diagnosis in diagnoses
            if diagnosis.reason == "WRONG_MOMENT" and diagnosis.band_id
        }
        rejected_bands = {
            band.band_id: band for band in bands if band.band_id in rejected_band_ids
        }

        def keep(hit: SearchHit) -> bool:
            if hit.video_id in rejected_videos:
                return False
            for band in rejected_bands.values():
                if hit.video_id != band.video_id:
                    continue
                if hit.start_ms <= band.end_ms and hit.end_ms >= band.start_ms:
                    return False
            return True

        return [hit for hit in all_hits if keep(hit)]

    def _allocate_calls(
        self,
        specs: list[tuple[QueryAtom, PromptVariant | None, str, float]],
        total_k: int,
        prefix: str,
        *,
        video_ids: list[str] | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[SearchCall]:
        valid_specs = [
            spec
            for spec in specs
            if self.constraint_engine.validate_retriever(spec[0], spec[2]).status == "PASS"
        ]
        if not valid_specs:
            return []
        if total_k < len(valid_specs):
            valid_specs = sorted(
                valid_specs,
                key=lambda spec: spec[3],
                reverse=True,
            )[:total_k]
        weights = [max(spec[3], 0.01) for spec in valid_specs]
        weight_sum = sum(weights)
        remaining = min(total_k, self.budget.raw_retrieval_max if prefix == "global" else total_k)
        calls: list[SearchCall] = []
        for index, ((atom, prompt, retriever, _), weight) in enumerate(zip(valid_specs, weights), start=1):
            slots_left = len(valid_specs) - index
            if slots_left == 0:
                top_k = remaining
            else:
                top_k = max(1, int(total_k * weight / weight_sum))
                top_k = min(top_k, remaining - slots_left)
            remaining -= top_k
            query_text = prompt.text if prompt is not None else atom.text
            if retriever in {"object_search", "track_search"}:
                query_text = self._object_class_query(atom.object or atom.text)
            calls.append(
                SearchCall(
                    call_id=f"{prefix}_{index:03d}_{atom.atom_id}",
                    atom_id=atom.atom_id,
                    event_id=atom.event_id,
                    prompt_role=prompt.role if prompt is not None else None,
                    retriever=retriever,
                    query=query_text,
                    top_k=top_k,
                    video_ids=video_ids or [],
                    start_ms=start_ms,
                    end_ms=end_ms,
                    min_count=atom.count or 1,
                )
            )
        return calls

    @staticmethod
    def _object_class_query(text: str) -> str:
        normalized = text.casefold().strip()
        aliases = {
            "people": "person",
            "persons": "person",
            "men": "person",
            "women": "person",
            "cars": "car",
            "dogs": "dog",
            "cats": "cat",
        }
        words = normalized.split()
        return aliases.get(words[-1], words[-1]) if words else normalized

    @staticmethod
    def _calibrate_plan_selectivity(
        plan: RetrievalPlan,
        hits: list[SearchHit],
    ) -> RetrievalPlan:
        calibrated: list[QueryAtom] = []
        for atom in plan.atoms:
            atom_hits = [hit for hit in hits if hit.atom_id == atom.atom_id]
            if not atom_hits:
                calibrated.append(atom)
                continue
            atom_selectivity = CorpusStats.measure_selectivity(
                [hit.video_id for hit in atom_hits],
                [max(0.0, hit.raw_score) for hit in atom_hits],
            ).selectivity_score
            variants: list[PromptVariant] = []
            for prompt in atom.prompt_variants:
                prompt_hits = [
                    hit for hit in atom_hits if hit.prompt_role == prompt.role
                ]
                if not prompt_hits:
                    variants.append(prompt)
                    continue
                prompt_selectivity = CorpusStats.measure_selectivity(
                    [hit.video_id for hit in prompt_hits],
                    [max(0.0, hit.raw_score) for hit in prompt_hits],
                ).selectivity_score
                variants.append(
                    prompt.model_copy(
                        update={
                            "weight": min(
                                2.0,
                                prompt.weight * (0.8 + 0.4 * prompt_selectivity),
                            )
                        }
                    )
                )
            calibrated.append(
                atom.model_copy(
                    update={
                        "discriminative_weight": min(
                            2.0,
                            atom.discriminative_weight * (0.8 + 0.4 * atom_selectivity),
                        ),
                        "prompt_variants": variants,
                    }
                )
            )
        return plan.model_copy(update={"atoms": calibrated})

    async def _run_calls(self, calls: list[SearchCall]) -> list[SearchHit]:
        if not calls:
            return []
        search_many = getattr(self.gateway, "search_many", None)
        if callable(search_many):
            try:
                results = await search_many(calls)
            except Exception as exc:
                logger.warning("V2 batch retrieval failed: error=%r", exc)
                results = [exc for _ in calls]
            if len(results) != len(calls):
                logger.warning(
                    "V2 batch retrieval returned %s result groups for %s calls",
                    len(results),
                    len(calls),
                )
                results = list(results[: len(calls)]) + [
                    RuntimeError("missing batch result")
                    for _ in range(max(0, len(calls) - len(results)))
                ]
        else:
            results = await asyncio.gather(
                *(self.gateway.search(call) for call in calls),
                return_exceptions=True,
            )
        hits: list[SearchHit] = []
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                logger.warning("V2 retrieval call failed: call=%s error=%r", call.call_id, result)
                continue
            family = self._retriever_family(call.retriever)
            hits.extend(
                hit.model_copy(
                    update={
                        "tool_call_id": call.call_id,
                        "atom_id": call.atom_id,
                        "event_id": call.event_id,
                        "prompt_role": call.prompt_role,
                        "retriever_family": family,
                    }
                )
                for hit in result
            )
        return hits

    @staticmethod
    def _canonical_candidate_count(hits: list[SearchHit]) -> int:
        return len(SearchController._canonical_candidate_keys(hits))

    def _limit_canonical_candidates(
        self,
        hits: list[SearchHit],
        atoms: list[QueryAtom] | None = None,
    ) -> list[SearchHit]:
        return select_candidate_reservoir(
            hits,
            atoms or (),
            limit=self.budget.unique_candidate_max,
        )

    def _bands(
        self,
        hits: list[SearchHit],
        required_atom_ids: list[str],
        atoms: list[QueryAtom],
    ):
        negative_atom_ids = list(dict.fromkeys(
            hit.atom_id for hit in hits
            if hit.atom_id is not None and hit.atom_id not in required_atom_ids
        ))
        all_bands = build_moment_bands(
            hits,
            required_atom_ids=required_atom_ids,
            negative_atom_ids=negative_atom_ids,
        )
        return rerank_bands(all_bands, atoms, self.budget.moment_band_limit)

    def _apply_candidate_gates(self, bands, atoms):
        accepted = []
        rescue = []
        precision_atom_ids = {
            atom.atom_id
            for atom in atoms
            if atom.required and atom.atom_type in {"COUNT", "RELATION"}
            or atom.operator == "MUST_NOT"
        }
        for band in bands:
            decisions = self.constraint_engine.evaluate_band(band, atoms)
            gated = band.model_copy(update={"constraint_decisions": decisions})
            if any(decision.status == "FAIL" for decision in decisions):
                continue
            precision_decisions = [
                decision
                for decision in decisions
                if decision.constraint_id.split(":", 1)[-1] in precision_atom_ids
            ]
            if precision_decisions and any(
                decision.status != "PASS" for decision in precision_decisions
            ):
                rescue.append(gated)
            else:
                accepted.append(gated)
        if not accepted:
            return rescue[: self.budget.rerank_limit]
        rescue_limit = max(1, self.budget.rerank_limit // 5)
        return [*accepted, *rescue[:rescue_limit]][: self.budget.rerank_limit]

    @staticmethod
    def _deduplicate(hits: list[SearchHit]) -> list[SearchHit]:
        best: dict[tuple[str | None, str | None, str, str, str | None], SearchHit] = {}
        for hit in hits:
            # Preserve hits across distinct prompt roles / tool calls to allow multi-prompt RRF fusion
            call_scope = hit.prompt_role or hit.tool_call_id
            key = (call_scope, hit.retriever_family, hit.entity_type, hit.entity_id, hit.atom_id)
            current = best.get(key)
            if current is None or (hit.raw_score, -hit.rank) > (current.raw_score, -current.rank):
                best[key] = hit
        return list(best.values())

    @staticmethod
    def _preferred_local_prompt(atom: QueryAtom) -> PromptVariant | None:
        for role in ("rare_detail", "action", "global"):
            match = next((prompt for prompt in atom.prompt_variants if prompt.role == role), None)
            if match is not None:
                return match
        return None

    @staticmethod
    def _preferred_retry_prompt(atom: QueryAtom) -> PromptVariant | None:
        for role in ("contrast", "rare_detail", "global"):
            match = next((prompt for prompt in atom.prompt_variants if prompt.role == role), None)
            if match is not None:
                return match
        return None

    @staticmethod
    def _diagnostic_retry_prompt(atom: QueryAtom, diagnoses) -> PromptVariant | None:
        reasons = {
            diagnosis.reason
            for diagnosis in diagnoses
            if diagnosis.atom_id in {None, atom.atom_id}
        }
        if "MISSING_ACTION" in reasons:
            return PromptVariant(
                role="action",
                text=f"Decisive visible action: {atom.text}.",
                weight=1.35,
            )
        if "WRONG_RELATION_OR_COUNT" in reasons:
            return PromptVariant(
                role="contrast",
                text=f"Exact visible spatial configuration and count: {atom.text}.",
                weight=1.4,
            )
        if "WRONG_MOMENT" in reasons:
            return PromptVariant(
                role="rare_detail",
                text=f"Specific local moment visibly containing {atom.text}.",
                weight=1.35,
            )
        if reasons & {"WRONG_VIDEO", "PROMPT_TOO_BROAD", "MISSING_REQUIRED_ATOM"}:
            return PromptVariant(
                role="rare_detail",
                text=f"Distinctive visual evidence of {atom.text}.",
                weight=1.35,
            )
        return SearchController._preferred_retry_prompt(atom)

    @staticmethod
    def _canonical_key(hit: SearchHit) -> tuple[str, int]:
        midpoint_ms = hit.start_ms + max(0, hit.end_ms - hit.start_ms) // 2
        return hit.video_id, midpoint_ms // 2_000

    @staticmethod
    def _canonical_candidate_keys(hits: list[SearchHit]) -> set[tuple[str, int]]:
        return {SearchController._canonical_key(hit) for hit in hits}

    @staticmethod
    def _retriever_family(retriever: str) -> str:
        if retriever in {"frame_search", "clip_search", "shot_search"}:
            return "legacy_clip_b32"
        return retriever


__all__ = ["SearchController", "SearchGateway"]
