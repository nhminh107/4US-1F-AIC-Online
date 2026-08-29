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
            atom_id=hit.atom_id,
            prompt_role=hit.prompt_role,
            retriever_family=hit.retriever_family,
        )

    @staticmethod
    def _split_long_cluster(
        hits: list[SearchHit],
        max_duration_ms: int,
    ) -> list[list[SearchHit]]:
        """Recursively split a cluster that exceeds max_duration_ms at its largest internal gap."""
        if not hits:
            return []
        start = hits[0].start_ms
        end = max(h.end_ms for h in hits)
        if end - start <= max_duration_ms or len(hits) <= 1:
            return [hits]

        best_split_idx = -1
        max_gap = -1
        for i in range(len(hits) - 1):
            gap = hits[i + 1].start_ms - hits[i].end_ms
            if gap > max_gap:
                max_gap = gap
                best_split_idx = i + 1

        if best_split_idx <= 0 or best_split_idx >= len(hits):
            best_split_idx = len(hits) // 2

        left = hits[:best_split_idx]
        right = hits[best_split_idx:]
        return (
            Aggregator._split_long_cluster(left, max_duration_ms)
            + Aggregator._split_long_cluster(right, max_duration_ms)
        )

    def execute(
        self,
        search_hits: list[SearchHit],
        merge_gap: int = 1000,
        max_region_duration_ms: int = 15_000,
    ) -> list[CandidateRegion]:
        if not search_hits:
            return []

        """Step 1: Group by Video and Event"""
        group_video_id: dict[tuple[str, str | None], list[SearchHit]] = {}
        for item in search_hits:
            key = (item.video_id, item.event_id)
            if key not in group_video_id:
                group_video_id[key] = []
            group_video_id[key].append(item)

        """Step 2: Combinator in each group with smart split"""
        list_region: list[CandidateRegion] = []
        for (video_id, event_id), hits in group_video_id.items():
            hits.sort(key=lambda hit: (hit.start_ms, hit.end_ms))

            clusters: list[list[SearchHit]] = []
            current_cluster = [hits[0]]
            current_end = hits[0].end_ms

            for hit in hits[1:]:
                if hit.start_ms <= current_end + merge_gap:
                    current_cluster.append(hit)
                    if hit.end_ms > current_end:
                        current_end = hit.end_ms
                else:
                    clusters.append(current_cluster)
                    current_cluster = [hit]
                    current_end = hit.end_ms
            if current_cluster:
                clusters.append(current_cluster)

            # Apply smart splitting to any oversized clusters
            final_clusters: list[list[SearchHit]] = []
            for cluster in clusters:
                final_clusters.extend(
                    self._split_long_cluster(cluster, max_region_duration_ms)
                )

            for count, cluster_hits in enumerate(final_clusters):
                evidence_list = [
                    self.__adapter_hit_to_candidate_evidence(h) for h in cluster_hits
                ]
                candidate_id = (
                    f"{video_id}_{event_id}_{count:03d}"
                    if event_id
                    else f"{video_id}_{count:03d}"
                )
                start_ms = min(h.start_ms for h in cluster_hits)
                end_ms = max(h.end_ms for h in cluster_hits)
                region = CandidateRegion(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    candidate_id=candidate_id,
                    video_id=video_id,
                    event_id=event_id,
                    evidence=evidence_list,
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

