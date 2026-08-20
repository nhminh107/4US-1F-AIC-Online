FRAME_EMBEDDING_WEIGHT = 1.0
SHOT_EMBEDDING_WEIGHT = 1.0 
CLIP_EMBEDDING_WEIGHT = 1.0
OCR_WEIGHT = 1.0
ASR_WEIGHT = 1.0
TRACK_WEIGHT = 1.0
RRF_K = 60
RANKING_MODEL = "gpt-oss-20b"
STD_BUFF = 1.5
DETECT_WEIGHT = 1.0

from pydantic_settings import BaseSettings

FPT_SYSTEM_PROMPT = (
    "Respond only with valid JSON. Do not include reasoning, thinking blocks, "
    "or any text outside the JSON object."
)

TOOL_TIMEOUTS: dict[str, float] = {
    "clip_search": 30.0,
    "frame_search": 30.0,
    "shot_search": 30.0,
    "ocr_search": 1.5,
    "asr_search": 1.5,
    "object_search": 1.5,
    "track_search": 1.5,
}

TOP_K_DEFAULTS: dict[str, int] = {
    "clip_search": 200,
    "frame_search": 200,
    "shot_search": 200,
    "ocr_search": 100,
    "asr_search": 100,
    "object_search": 100,
    "track_search": 100,
}


class LLMConfig(BaseSettings):
    api_key: str
    base_url: str

    # Text reasoning — Intent Extractor, Query Planner
    llm_model: str = "Qwen3.6-27B"

    # Vision-language — VQA, Verifier (module của teammate)
    vlm_model: str = "Qwen2.5-VL-7B-Instruct"

    max_retries: int = 3
    temperature: float = 0.0
    fpt_system_prompt: str = FPT_SYSTEM_PROMPT

    class Config:
        env_prefix = "FPT_"
        env_file = ".env"
        extra = "ignore"

LLM_CONFIG = LLMConfig()


__all__ = [
    "FPT_SYSTEM_PROMPT",
    "LLMConfig",
    "LLM_CONFIG",
    "TOP_K_DEFAULTS",
    "TOOL_TIMEOUTS",
]
