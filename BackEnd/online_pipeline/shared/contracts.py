# shared/contracts.py
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# --- StructuredQuery ---
class KISQuery(BaseModel):
    task: Literal["KIS"]
    visual_queries: list[str] = []
    ocr_constraints: list[str] = []
    negative_constraints: list[str] = []

class VQAQuery(BaseModel):
    task: Literal["VQA"]
    question: str
    visual_queries: list[str] = []
    ocr_constraints: list[str] = []

StructuredQuery = Annotated[
    KISQuery | VQAQuery,
    Field(discriminator="task")
]

# --- SearchHit ---
class SearchHit(BaseModel):
    source: str
    entity_type: str
    entity_id: str
    video_id: str
    start_ms: int
    end_ms: int
    rank: int
    raw_score: float
    event_id: str | None = None

# --- ToolCall ---
ToolName = Literal[
    "clip_search", "frame_search", "shot_search",
    "ocr_search", "asr_search", "caption_search",
]
