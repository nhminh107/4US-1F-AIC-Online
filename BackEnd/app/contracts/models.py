from pydantic import BaseModel
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RawQuery(BaseModel):
    query_id: str 
    session_id: str
    text: str 
    feedback: str | None

class Event(BaseModel):
    event_id: str 
    description: str 

@dataclass(frozen=True, slots=True)
class StructuredQuery(BaseModel):
    query_id: str
    task: str

    visual_queries: list[str]
    ocr_constraints: list[str]
    asr_constraints: list[str]
    object_constraints: list[str]
    feedback: list[str]
    events: list[Event]
    temporal_constraint: list[(Event, Event)]
    negative_constraints: list[str]

@dataclass(frozen=True, slots=True)
class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    event_id: str | None = None
    parameters: dict

@dataclass(frozen=True, slots=True)
class SearchHit(BaseModel):
    tool_call_id: str | None = None
    event_id: str | None = None
    source: str
    entity_type: str
    entity_id: str
    video_id: str
    start_ms: int
    end_ms: int
    rank: int
    raw_score: float

@dataclass(frozen=True, slots=True)
class CandidateEvidence(BaseModel):
    source: str
    entity_id: str
    rank: int
    raw_score: float

@dataclass(frozen=True, slots=True)
class CandidateRegion(BaseModel):
    candidate_id: str
    event_id: str | None
    video_id: str
    start_ms: int
    end_ms: int
    evidence: list[CandidateEvidence]

@dataclass(frozen=True, slots=True)
class ConstraintResult(BaseModel):
    hard_constraints_passed: bool
    negative_constraints_passed: bool

@dataclass(frozen=True, slots=True)
class RankedCandidateRegion(BaseModel):
    candidate_id: str
    event_id: str | None

    video_id: str

    start_ms: int
    end_ms: int

    fusion_score: float

    constraint_result: ConstraintResult

    evidence: list[CandidateEvidence]

@dataclass(frozen=True, slots=True)
class EvidenceBundle(BaseModel):
    video_id: str

    start_ms: int
    end_ms: int

    frames: list
    ocr: list
    asr: list
    captions: list
    objects: list
    tracks: list

@dataclass(frozen=True, slots=True)
class KISResult(BaseModel):
    video_id: str

    start_ms: int
    end_ms: int

    representative_frame_id: str

    score: float

    evidence_ids: list[str]

@dataclass(frozen=True, slots=True)
class VQAResult(BaseModel):
    answer: str
    confidence: float
    evidence_ids: list[str]
    status: str

@dataclass(frozen=True, slots=True)
class TemporalEventResult(BaseModel):
    event_id: str
    candidate_id: str

    start_ms: int
    end_ms: int

@dataclass(frozen=True, slots=True)
class TemporalSequence(BaseModel):
    video_id: str
    events: list[TemporalEventResult]
    sequence_score: float

@dataclass(frozen=True, slots=True)
class VerifiedResult(BaseModel):
    status: str

    confidence: float

    supporting_evidence_ids: list[str]

    failed_constraints: list[str]