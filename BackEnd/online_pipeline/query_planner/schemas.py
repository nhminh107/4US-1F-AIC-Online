from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from online_pipeline.shared.config import TOP_K_DEFAULTS


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


class CaptionSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = TOP_K_DEFAULTS["caption_search"]


class ObjectSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int


class TrackSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int


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


class ToolCall(BaseModel):
    tool: ToolName
    parameters: dict[str, Any]
    event_id: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "ToolCall":
        schema = TOOL_PARAMS[self.tool]
        self.parameters = schema.model_validate(self.parameters).model_dump()
        return self


__all__ = [
    "ASRSearchParams",
    "CaptionSearchParams",
    "ClipSearchParams",
    "FrameSearchParams",
    "OCRSearchParams",
    "ObjectSearchParams",
    "ShotSearchParams",
    "TOOL_PARAMS",
    "ToolCall",
    "ToolName",
    "TrackSearchParams",
]
