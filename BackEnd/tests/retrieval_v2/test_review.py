import asyncio

from BackEnd.app.contracts.models import SearchHit, StructuredQuery
from BackEnd.app.retrieval_v2.contracts import CandidateBudget, CandidateReview, SearchCall
from BackEnd.app.retrieval_v2.controller import SearchController
from BackEnd.app.retrieval_v2.review import apply_candidate_reviews, diagnose_candidate_reviews
from BackEnd.app.retrieval_v2.contracts import MomentBand


class Gateway:
    async def search(self, call: SearchCall):
        return [
            SearchHit(
                source="visual",
                entity_type="frame",
                entity_id=f"{call.call_id}-{index}",
                video_id=f"V{index % 3}",
                start_ms=index * 5_000,
                end_ms=index * 5_000,
                rank=index + 1,
                raw_score=0.9 - index * 0.01,
            )
            for index in range(min(call.top_k, 12))
        ]


class Reviewer:
    def __init__(self):
        self.seen_band_ids = []

    async def review(self, _plan, bands):
        self.seen_band_ids = [band.band_id for band in bands]
        return [
            CandidateReview(band_id=bands[0].band_id, verdict="mismatch", confidence=0.95),
            CandidateReview(band_id=bands[1].band_id, verdict="match", confidence=0.95),
        ]


def test_precise_reviewer_only_sees_budgeted_shortlist_and_can_rerank():
    reviewer = Reviewer()
    controller = SearchController(
        Gateway(),
        reviewer=reviewer,
        budget=CandidateBudget(
            raw_retrieval_target=50,
            raw_retrieval_max=80,
            moment_band_limit=20,
            video_shortlist_limit=3,
            local_retrieval_k=10,
            retry_retrieval_k=5,
            rerank_limit=10,
            review_limit=2,
            max_retry_rounds=0,
        ),
    )
    query = StructuredQuery(
        query_id="q-review",
        task="KIS",
        visual_queries=["a lion dancer biting a pumpkin on top of poles"],
    )

    result = asyncio.run(controller.search(query))

    assert len(reviewer.seen_band_ids) == 2
    assert result.session.reviews
    assert result.reranked_bands[0].band_id == reviewer.seen_band_ids[1]
    assert reviewer.seen_band_ids[0] not in {
        band.band_id for band in result.reranked_bands
    }


def test_rejected_scope_is_rematerialized_when_retry_adds_no_hits():
    class NoGainGateway:
        async def search(self, call: SearchCall):
            if call.call_id.startswith("retry_"):
                return []
            return [
                SearchHit(
                    source="visual",
                    entity_type="frame",
                    entity_id=f"{call.call_id}-bad",
                    video_id="V-bad",
                    start_ms=1_000,
                    end_ms=1_000,
                    rank=1,
                    raw_score=0.9,
                )
            ]

    class WrongMomentReviewer:
        async def review(self, _plan, bands):
            if not bands:
                return []
            return [
                CandidateReview(
                    band_id=bands[0].band_id,
                    verdict="mismatch",
                    confidence=0.99,
                    scope="MOMENT_BAND",
                    failure_reason="wrong_moment",
                    video_id=bands[0].video_id,
                )
            ]

    controller = SearchController(
        NoGainGateway(),
        reviewer=WrongMomentReviewer(),
        budget=CandidateBudget(
            raw_retrieval_target=10,
            raw_retrieval_max=10,
            unique_candidate_min=1,
            unique_candidate_max=5,
            moment_band_limit=5,
            video_shortlist_limit=2,
            local_retrieval_k=2,
            retry_retrieval_k=2,
            rerank_limit=5,
            review_limit=1,
            max_retry_rounds=1,
        ),
    )

    result = asyncio.run(
        controller.search(
            StructuredQuery(
                query_id="q-reject-no-gain",
                task="KIS",
                visual_queries=["a red car"],
            )
        )
    )

    assert result.reranked_bands == []
    assert result.session.stop_reason == "NO_CANDIDATE_GAIN"


def test_review_diagnosis_keeps_wrong_video_and_wrong_moment_scopes_separate():
    reviews = [
        CandidateReview(
            band_id="band-video",
            verdict="mismatch",
            confidence=0.98,
            scope="VIDEO",
            failure_reason="wrong_video",
            video_id="V1",
        ),
        CandidateReview(
            band_id="band-moment",
            verdict="mismatch",
            confidence=0.95,
            scope="MOMENT_BAND",
            failure_reason="wrong_moment",
            video_id="V2",
        ),
    ]

    diagnoses = diagnose_candidate_reviews(reviews)

    assert diagnoses[0].reason == "WRONG_VIDEO"
    assert diagnoses[0].action == "REJECT_VIDEO_AND_BROADEN"
    assert diagnoses[1].reason == "WRONG_MOMENT"
    assert diagnoses[1].action == "REJECT_BAND_AND_LOCAL_SEARCH"
    assert diagnoses[1].video_id == "V2"


def test_earlier_mismatch_cannot_be_overwritten_by_later_uncertain_review():
    band = MomentBand(
        band_id="band-1",
        video_id="V1",
        start_ms=1_000,
        end_ms=2_000,
        peak_ms=1_500,
        score=1.0,
    )

    result = apply_candidate_reviews(
        [band],
        [
            CandidateReview(
                band_id=band.band_id,
                verdict="mismatch",
                confidence=0.99,
            ),
            CandidateReview(
                band_id=band.band_id,
                verdict="uncertain",
                confidence=0.0,
            ),
        ],
    )

    assert result == []
