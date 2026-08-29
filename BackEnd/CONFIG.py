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

# ==== Module 6A - KIS Handler ====
TOP_N_KIS = 100
KIS_EDGE_RATIO = 0.1
KIS_NEIGHBOR_SHOT_COUNT = 1

import os
from pydantic import BaseModel

try:
    from pydantic_settings import BaseSettings
except ImportError:
    class BaseSettings(BaseModel):  # type: ignore[no-redef]
        def __init__(self, **values):
            env_prefix = getattr(getattr(self, "Config", None), "env_prefix", "")
            for field_name, field_info in self.model_fields.items():
                if field_name not in values:
                    env_key = f"{env_prefix}{field_name}".upper()
                    env_val = os.getenv(env_key)
                    if env_val is not None:
                        values[field_name] = env_val
                    elif field_info.default is not None:
                        values[field_name] = field_info.default
                    else:
                        values[field_name] = ""
            super().__init__(**values)

FPT_SYSTEM_PROMPT = (
    "Respond only with valid JSON. Do not include reasoning, thinking blocks, "
    "or any text outside the JSON object."
)

TOOL_TIMEOUTS: dict[str, float] = {
    "clip_search": 30.0,
    "frame_search": 30.0,
    "shot_search": 30.0,
    "ocr_search": 3.0,
    "asr_search": 3.0,
    "object_search": 5.0,
    "track_search": 5.0,
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
    api_key: str = ""
    base_url: str = "https://mkp-api.fptcloud.com"

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
    "TOP_N_KIS",
    "KIS_EDGE_RATIO",
    "KIS_NEIGHBOR_SHOT_COUNT",
]
