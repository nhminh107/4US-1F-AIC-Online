from __future__ import annotations

import asyncio
import logging
from typing import Any

from online_pipeline.intent_extractor.prompts import (
    classify_task_prompt,
    extract_query_prompt,
)
from online_pipeline.intent_extractor.schemas import (
    KISQuery,
    StructuredQuery,
    TaskClassification,
)
from online_pipeline.shared.config import LLM_CONFIG

try:
    from instructor.exceptions import InstructorRetryException
except ImportError:
    try:
        from instructor import InstructorRetryException
    except ImportError:
        class InstructorRetryException(Exception):
            pass


DEFAULT_MODEL = LLM_CONFIG.model_name
DEFAULT_MAX_RETRIES = LLM_CONFIG.max_retries

logger = logging.getLogger(__name__)


def build_instructor_client() -> Any:
    import instructor
    from openai import OpenAI

    return instructor.from_openai(OpenAI())


def _render_prompt(prompt: str, raw_text: str) -> str:
    return prompt.replace("{raw_query}", raw_text)


def _coerce_raw_text(raw_query: Any) -> str:
    if isinstance(raw_query, str):
        return raw_query
    return raw_query.text


def _fallback_to_kis(raw_text: str) -> KISQuery:
    logger.warning(
        "Intent extraction failed, using raw text fallback for raw_text=%r",
        raw_text,
        exc_info=True,
    )
    return KISQuery(task="KIS", visual_queries=[raw_text])


def extract_intent_sync(
    raw_text: Any,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = LLM_CONFIG.temperature,
) -> StructuredQuery:
    raw_text = _coerce_raw_text(raw_text)
    client = client or build_instructor_client()

    try:
        classification = client.chat.completions.create(
            model=model,
            response_model=TaskClassification,
            messages=[
                {
                    "role": "user",
                    "content": _render_prompt(classify_task_prompt(), raw_text),
                }
            ],
            max_retries=max_retries,
            temperature=temperature,
        )

        return client.chat.completions.create(
            model=model,
            response_model=StructuredQuery,
            messages=[
                {
                    "role": "user",
                    "content": _render_prompt(
                        extract_query_prompt(classification.task),
                        raw_text,
                    ),
                }
            ],
            max_retries=max_retries,
            temperature=temperature,
        )
    except InstructorRetryException:
        return _fallback_to_kis(raw_text)


async def extract_intent(
    raw_text: Any,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = LLM_CONFIG.temperature,
) -> StructuredQuery:
    return await asyncio.to_thread(
        extract_intent_sync,
        raw_text,
        client=client,
        model=model,
        max_retries=max_retries,
        temperature=temperature,
    )


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "InstructorRetryException",
    "build_instructor_client",
    "extract_intent",
    "extract_intent_sync",
]
