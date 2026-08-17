from typing import Annotated, Literal

from pydantic import BaseModel, Field

from online_pipeline.shared.contracts import SearchHit


class KISQuery(BaseModel):
    task: Literal["KIS"]
    visual_queries: list[str] = Field(default_factory=list)
    ocr_constraints: list[str] = Field(default_factory=list)
    asr_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)


class VQAQuery(BaseModel):
    task: Literal["VQA"]
    question: str
    visual_queries: list[str] = Field(default_factory=list)
    ocr_constraints: list[str] = Field(default_factory=list)


class TaskClassification(BaseModel):
    task: Literal["KIS", "VQA"]


StructuredQuery = Annotated[
    KISQuery | VQAQuery,
    Field(discriminator="task"),
]


__all__ = [
    "KISQuery",
    "SearchHit",
    "StructuredQuery",
    "TaskClassification",
    "VQAQuery",
]
