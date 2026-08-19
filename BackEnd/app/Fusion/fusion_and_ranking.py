from BackEnd.CONFIG import (ASR_WEIGHT, CLIP_EMBEDDING_WEIGHT, OCR_WEIGHT, FRAME_EMBEDDING_WEIGHT, 
                            SHOT_EMBEDDING_WEIGHT, TRACK_WEIGHT, RRF_K, STD_BUFF, DETECT_WEIGHT) 
from BackEnd.app.contracts.models import CandidateRegion, StructuredQuery, RankedCandidateRegion, ConstraintResult


"""
class StructuredQuery(ContractModel):
    query_id: str = Field(min_length=1)
    task: Literal["KIS", "VQA", "TRAKE"]
    question: str = ""
    visual_queries: list[str] = Field(default_factory=list)
    ocr_constraints: list[str] = Field(default_factory=list)
    asr_constraints: list[str] = Field(default_factory=list)
    object_constraints: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(
        default_factory=list,
        validation_alias=AliasChoices("temporal_constraints", "temporal_constraint"),
    )
    negative_constraints: list[str] = Field(default_factory=list)

    CandidateEvidence(
            source=hit.source,
            entity_type=hit.entity_type,
            entity_id=hit.entity_id,
            start_ms=hit.start_ms,
            end_ms=hit.end_ms,
            rank=hit.rank,
            raw_score=hit.raw_score,
            tool_call_id=hit.tool_call_id,
        )
"""
class FusionRanking(): 
    def __init__(self): 
        pass
        self.weight_mapping = {
            'frame': FRAME_EMBEDDING_WEIGHT, 
            'ocr': OCR_WEIGHT, 
            'shot': SHOT_EMBEDDING_WEIGHT, 
            'clip': CLIP_EMBEDDING_WEIGHT, 
            'asr': ASR_WEIGHT, 
            'object_track': TRACK_WEIGHT,
            'object_detection': DETECT_WEIGHT
        }

    def __prepare_weight_buff(self, weight_buff: dict): 
        result = {}
        count = 0
        for key in weight_buff: 
            if weight_buff[key] == True: 
                count += 1 

        for key in weight_buff: 
            if weight_buff[key] == True: 
                if key not in result: 
                    result[key] = 1.0*STD_BUFF / count
            else: 
                if key not in result: 
                    result[key] = 0 

        return result
    def __region_scoring(self, candidate_region: CandidateRegion, weight_buff: dict): 
        score = 0.0
        for candidate in candidate_region.evidence:
            candidate_score = (self.weight_mapping[candidate.entity_type] + weight_buff[candidate.entity_type])/ (RRF_K + candidate.rank)
            score += candidate_score

        return score

    def __contraint_result(self): 
        return ConstraintResult(hard_constraints_passed=True, negative_constraints_passed=True)
    
    def fusion_and_ranking(self, region_list: list[CandidateRegion], weight_buff: dict, query: StructuredQuery = None):
        buff_weight = self.__prepare_weight_buff(weight_buff)
        list_candidate_ranking = []

        for region in region_list: 
            score = self.__region_scoring(region, buff_weight)
            cs = self.__contraint_result()
            """class RankedCandidateRegion(TimeRangeModel):
                candidate_id: str = Field(min_length=1)
                event_id: str | None = None
                video_id: str = Field(min_length=1)
                fusion_score: float
                constraint_result: ConstraintResult
                evidence: list[CandidateEvidence] = Field(default_factory=list)
            """
            candidate_ranking = RankedCandidateRegion(
                event_id=region.event_id,
                video_id=region.video_id, 
                fusion_score=score, 
                constraint_result=cs,
                candidate_id=region.candidate_id,
                evidence=region.evidence,
                start_ms=region.start_ms, 
                end_ms=region.end_ms
            )

            list_candidate_ranking.append(candidate_ranking)

        list_candidate_ranking.sort(key=lambda c: c.fusion_score, reverse=True)
        return list_candidate_ranking



        


    




    