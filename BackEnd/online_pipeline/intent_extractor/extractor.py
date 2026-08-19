from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import TypeAdapter

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
from online_pipeline.shared.utils import strip_thinking_blocks

try:
    from instructor.exceptions import InstructorRetryException
except ImportError:
    try:
        from instructor import InstructorRetryException
    except ImportError:
        class InstructorRetryException(Exception):
            pass


DEFAULT_MODEL = LLM_CONFIG.llm_model
DEFAULT_MAX_RETRIES = LLM_CONFIG.max_retries

logger = logging.getLogger(__name__)


def build_instructor_client() -> Any:
    import instructor
    from openai import OpenAI

    return instructor.from_openai(
        OpenAI(
            api_key=LLM_CONFIG.api_key,
            base_url=LLM_CONFIG.base_url,
        ),
        mode=instructor.Mode.JSON,
    )


def _render_prompt(prompt: str, raw_text: str) -> str:
    return prompt.replace("{raw_query}", raw_text)


def _coerce_raw_text(raw_query: Any) -> str:
    if isinstance(raw_query, str):
        return raw_query
    return raw_query.text


def _with_fpt_system_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            updated = list(messages)
            updated[index] = {
                **message,
                "content": (
                    f"{LLM_CONFIG.fpt_system_prompt}\n\n"
                    f"{message.get('content', '')}"
                ),
            }
            return updated
    return [
        {"role": "system", "content": LLM_CONFIG.fpt_system_prompt},
        *messages,
    ]


def _completion_content(error: Exception) -> str | None:
    for attribute in ("last_completion", "last_response"):
        completion = getattr(error, attribute, None)
        if completion is None:
            continue
        try:
            message = completion.choices[0].message
            content = message.content
            if content is None:
                content = getattr(message, "reasoning_content", None)
        except (AttributeError, IndexError, TypeError):
            continue
        if isinstance(content, str):
            return content
    return None


def _parse_stripped_completion(
    error: Exception,
    response_model: Any,
) -> Any | None:
    raw_content = _completion_content(error)
    if raw_content is None:
        return None

    try:
        cleaned_content = strip_thinking_blocks(raw_content)
        if response_model is TaskClassification:
            return TaskClassification.model_validate_json(cleaned_content)
        if response_model is StructuredQuery:
            return TypeAdapter(StructuredQuery).validate_json(cleaned_content)
    except Exception:
        return None
    return None


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
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _render_prompt(classify_task_prompt(), raw_text),
                    }
                ]
            ),
            max_retries=max_retries,
            temperature=temperature,
        )
    except InstructorRetryException as error:
        classification = _parse_stripped_completion(error, TaskClassification)
        if classification is None:
            return _fallback_to_kis(raw_text)

    try:
        return client.chat.completions.create(
            model=model,
            response_model=StructuredQuery,
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _render_prompt(
                            extract_query_prompt(classification.task),
                            raw_text,
                        ),
                    }
                ]
            ),
            max_retries=max_retries,
            temperature=temperature,
        )
    except InstructorRetryException as error:
        parsed_query = _parse_stripped_completion(error, StructuredQuery)
        if parsed_query is not None:
            return parsed_query
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
