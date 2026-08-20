from BackEnd.app.contracts.models import CandidateEvidence, CandidateRegion, SearchHit
from BackEnd.app.services.evidence_service import get_evidence_bundle, get_temporal_neighbors
from BackEnd.app.Database.postgre_manager import PostgreManager
"""
Input: List of SearchHit

class SearchHit(CanonicalEntityRef):
    tool_call_id: str | None = None
    event_id: str | None = None
    source: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float

Output: 
class CandidateEvidence(ContractModel):
    source: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float


class CandidateRegion(TimeRangeModel):
    candidate_id: str = Field(min_length=1)
    event_id: str | None = None
    video_id: str = Field(min_length=1)
    evidence: list[CandidateEvidence] = Field(default_factory=list)

"""

class Aggregator:
    def __init__(self, db_mng: PostgreManager | None = None):
        self.db_mng = db_mng

    @staticmethod
    def __adapter_hit_to_candidate_evidence(
        hit: SearchHit,
    ) -> CandidateEvidence:
        return CandidateEvidence(
            source=hit.source,
            entity_type=hit.entity_type,
            entity_id=hit.entity_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            rank=hit.rank,
            raw_score=hit.raw_score,
            tool_call_id=hit.tool_call_id,
        )

    def execute(self, search_hits: list[SearchHit], merge_gap: int = 2000) -> list[CandidateRegion]:
        if not search_hits:
            return []

        """Step 1: Group by Video and Event"""
        group_video_id: dict[tuple[str, str | None], list[SearchHit]] = {}
        for item in search_hits: 
            key = (item.video_id, item.event_id)
            if key not in group_video_id:
                group_video_id[key] = []
            group_video_id[key].append(item)

        """Step 2: Combinator in each group"""
        list_region = []
        for (video_id, event_id), hits in group_video_id.items():
            hits.sort(key=lambda hit: (hit.start_ms, hit.end_ms))

            current_hits = [hits[0]]
            current_start = hits[0].start_ms
            current_end = hits[0].end_ms 
            count = 0

            for hit in hits[1:]:
                is_close = hit.start_ms <= current_end + merge_gap
                if is_close: 
                    current_hits.append(hit)
                    if hit.end_ms > current_end:
                        current_end = hit.end_ms
                else:
                    evidence_list = [
                        self.__adapter_hit_to_candidate_evidence(h) for h in current_hits
                    ]
                    candidate_id = f"{video_id}_{event_id}_{count:03d}" if event_id else f"{video_id}_{count:03d}"
                    region = CandidateRegion(
                        start_ms=current_start,
                        end_ms=current_end, 
                        candidate_id=candidate_id,
                        video_id=video_id,
                        event_id=event_id,
                        evidence=evidence_list
                    )
                    list_region.append(region)

                    current_hits = [hit]
                    current_start = hit.start_ms
                    current_end = hit.end_ms
                    count += 1

            if current_hits:
                evidence_list = [
                    self.__adapter_hit_to_candidate_evidence(h) for h in current_hits
                ]
                candidate_id = f"{video_id}_{event_id}_{count:03d}" if event_id else f"{video_id}_{count:03d}"
                region = CandidateRegion(
                    start_ms=current_start,
                    end_ms=current_end, 
                    candidate_id=candidate_id,
                    video_id=video_id,
                    event_id=event_id,
                    evidence=evidence_list
                )
                list_region.append(region)

        return list_region


if __name__ == "__main__":
    from random import randint, uniform

    video_ids = [f"L21_V{randint(1, 999):03d}" for _ in range(3)]
    search_hits = []

    for index, video_id in enumerate(
        [video_ids[0], video_ids[0], video_ids[1], video_ids[1], video_ids[2]],
        start=1,
    ):
        start_ms = randint(0, 300_000)
        frame_id = f"F{randint(1, 9999):04d}"
        search_hits.append(
            SearchHit(
                tool_call_id=f"TC{index:03d}",
                event_id=f"E{randint(1, 3)}",
                source="frame_embedding",
                entity_type="frame",
                entity_id=frame_id,
                video_id=video_id,
                frame_id=frame_id,
                start_ms=start_ms,
                end_ms=start_ms,
                rank=randint(1, 100),
                raw_score=round(uniform(0, 1), 4),
            )
        )

    agg = Aggregator()
    res = agg.execute(search_hits)
    print(f"Generated {len(res)} regions from {len(search_hits)} search hits.")

