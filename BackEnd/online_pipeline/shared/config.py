from dataclasses import dataclass


TOOL_TIMEOUTS: dict[str, float] = {
    "clip_search": 2.0,
    "frame_search": 2.0,
    "shot_search": 2.0,
    "ocr_search": 1.5,
    "asr_search": 1.5,
    "caption_search": 1.5,
}

TOP_K_DEFAULTS: dict[str, int] = {
    "clip_search": 200,
    "frame_search": 200,
    "shot_search": 200,
    "ocr_search": 100,
    "asr_search": 100,
    "caption_search": 100,
}


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "gpt-4o"
    max_retries: int = 3
    temperature: float = 0.0


LLM_CONFIG = LLMConfig()


__all__ = ["LLMConfig", "LLM_CONFIG", "TOP_K_DEFAULTS", "TOOL_TIMEOUTS"]
