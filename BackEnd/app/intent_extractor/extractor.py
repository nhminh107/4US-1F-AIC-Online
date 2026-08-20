from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from BackEnd.CONFIG import LLM_CONFIG
from BackEnd.app.contracts.models import StructuredQuery
from BackEnd.app.intent_extractor.prompts import (
    classify_task_prompt,
    extract_query_prompt,
)
from BackEnd.app.intent_extractor.schemas import TaskClassification
from BackEnd.app.intent_extractor.utils import strip_thinking_blocks

try:
    from instructor.core import InstructorRetryException
except ImportError:
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
    if not LLM_CONFIG.api_key or not LLM_CONFIG.base_url:
        raise RuntimeError(
            "FPT_API_KEY and FPT_BASE_URL must be configured before building "
            "the Intent Extractor LLM client."
        )

    import instructor
    from openai import OpenAI

    return instructor.from_openai(
        OpenAI(
            api_key=LLM_CONFIG.api_key,
            base_url=LLM_CONFIG.base_url,
        ),
        mode=instructor.Mode.JSON,
    )


def _stable_query_id(raw_text: str) -> str:
    digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]
    return f"query_{digest}"


def _render_prompt(prompt: str, raw_text: str, query_id: str) -> str:
    return prompt.replace("{raw_query}", raw_text).replace("{query_id}", query_id)


def _coerce_raw_query(raw_query: Any) -> tuple[str, str]:
    if isinstance(raw_query, str):
        return raw_query, _stable_query_id(raw_query)

    raw_text = raw_query.text
    query_id = getattr(raw_query, "query_id", None) or _stable_query_id(raw_text)
    return raw_text, query_id


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


def _parse_stripped_completion(error: Exception, response_model: Any) -> Any | None:
    raw_content = _completion_content(error)
    if raw_content is None:
        return None

    try:
        cleaned_content = strip_thinking_blocks(raw_content)
        return response_model.model_validate_json(cleaned_content)
    except Exception:
        return None


def _normalize_structured_query(
    structured_query: StructuredQuery,
    *,
    task: str,
    query_id: str,
) -> StructuredQuery:
    payload = structured_query.model_dump()
    payload["task"] = task
    payload["query_id"] = query_id
    return StructuredQuery.model_validate(payload)


def _fallback_to_kis(raw_text: str, query_id: str) -> StructuredQuery:
    logger.warning(
        "Intent extraction failed, using raw text fallback for raw_text=%r",
        raw_text,
        exc_info=True,
    )
    return StructuredQuery(
        query_id=query_id,
        task="KIS",
        visual_queries=[raw_text],
    )


def extract_intent_sync(
    raw_query: Any,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = LLM_CONFIG.temperature,
) -> StructuredQuery:
    raw_text, query_id = _coerce_raw_query(raw_query)
    client = client or build_instructor_client()

    try:
        classification = client.chat.completions.create(
            model=model,
            response_model=TaskClassification,
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _render_prompt(
                            classify_task_prompt(),
                            raw_text,
                            query_id,
                        ),
                    }
                ]
            ),
            max_retries=max_retries,
            temperature=temperature,
        )
    except InstructorRetryException as error:
        classification = _parse_stripped_completion(error, TaskClassification)
        if classification is None:
            return _fallback_to_kis(raw_text, query_id)

    try:
        structured_query = client.chat.completions.create(
            model=model,
            response_model=StructuredQuery,
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _render_prompt(
                            extract_query_prompt(classification.task),
                            raw_text,
                            query_id,
                        ),
                    }
                ]
            ),
            max_retries=max_retries,
            temperature=temperature,
        )
        return _normalize_structured_query(
            structured_query,
            task=classification.task,
            query_id=query_id,
        )
    except InstructorRetryException as error:
        parsed_query = _parse_stripped_completion(error, StructuredQuery)
        if parsed_query is not None:
            return _normalize_structured_query(
                parsed_query,
                task=classification.task,
                query_id=query_id,
            )
        return _fallback_to_kis(raw_text, query_id)


async def extract_intent(
    raw_query: Any,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = LLM_CONFIG.temperature,
) -> StructuredQuery:
    return await asyncio.to_thread(
        extract_intent_sync,
        raw_query,
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
