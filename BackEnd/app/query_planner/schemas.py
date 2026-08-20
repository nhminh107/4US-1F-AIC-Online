from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from BackEnd.CONFIG import TOP_K_DEFAULTS
from BackEnd.app.contracts.models import ToolCall


class ClipSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["clip_search"]


class FrameSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["frame_search"]


class ShotSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["shot_search"]


class OCRSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["ocr_search"]
    mode: Literal["exact", "fuzzy", "similarity"] = "similarity"


class ASRSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["asr_search"]
    mode: Literal["exact", "fuzzy", "similarity"] = "similarity"


class CaptionSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["caption_search"]
    mode: Literal["exact", "fuzzy", "similarity"] = "similarity"


class ObjectSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_class: str
    top_k: int = TOP_K_DEFAULTS["object_search"]
    min_count: int = 1


class TrackSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_class: str
    top_k: int = TOP_K_DEFAULTS["track_search"]
    relation: str = "continuous_track"


ToolName = Literal[
    "clip_search",
    "frame_search",
    "shot_search",
    "ocr_search",
    "asr_search",
    "caption_search",
    "object_search",
    "track_search",
]


TOOL_PARAMS: dict[str, type[BaseModel]] = {
    "clip_search": ClipSearchParams,
    "frame_search": FrameSearchParams,
    "shot_search": ShotSearchParams,
    "ocr_search": OCRSearchParams,
    "asr_search": ASRSearchParams,
    "caption_search": CaptionSearchParams,
    "object_search": ObjectSearchParams,
    "track_search": TrackSearchParams,
}


class PlannedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    parameters: dict[str, Any]
    event_id: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "PlannedToolCall":
        schema = TOOL_PARAMS[self.tool]
        self.parameters = schema.model_validate(self.parameters).model_dump()
        return self

    def to_contract(self, index: int) -> ToolCall:
        return ToolCall(
            tool_call_id=f"tc_{index:03d}",
            tool_name=self.tool,
            event_id=self.event_id,
            parameters=self.parameters,
        )


def to_contract_tool_calls(planned_calls: list[PlannedToolCall]) -> list[ToolCall]:
    return [
        planned_call.to_contract(index)
        for index, planned_call in enumerate(planned_calls, start=1)
    ]


__all__ = [
    "ASRSearchParams",
    "CaptionSearchParams",
    "ClipSearchParams",
    "FrameSearchParams",
    "OCRSearchParams",
    "ObjectSearchParams",
    "PlannedToolCall",
    "ShotSearchParams",
    "TOOL_PARAMS",
    "ToolName",
    "TrackSearchParams",
    "to_contract_tool_calls",
]
