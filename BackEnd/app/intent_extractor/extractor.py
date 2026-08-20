from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from BackEnd.CONFIG import LLM_CONFIG
from BackEnd.app.contracts.models import StructuredQuery
from BackEnd.app.intent_extractor.prompts import extract_structured_query_prompt
from BackEnd.app.intent_extractor.object_classes import normalize_object_constraints
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
SUPPORTED_TASKS = frozenset({"KIS", "VQA", "TRAKE"})

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


def _render_prompt(
    prompt: str,
    raw_text: str,
    query_id: str,
    feedback: str,
) -> str:
    return (
        prompt.replace("{raw_query}", raw_text)
        .replace("{query_id}", query_id)
        .replace("{feedback}", feedback)
    )


def _normalize_feedback(*feedback_groups: list[str]) -> list[str]:
    normalized: list[str] = []
    for feedback_group in feedback_groups:
        for item in feedback_group:
            cleaned_item = item.strip()
            if cleaned_item and cleaned_item not in normalized:
                normalized.append(cleaned_item)
    return normalized


def _coerce_raw_query(raw_query: Any) -> tuple[str, str, list[str]]:
    if isinstance(raw_query, str):
        return raw_query, _stable_query_id(raw_query), []

    raw_text = raw_query.text
    query_id = getattr(raw_query, "query_id", None) or _stable_query_id(raw_text)
    raw_feedback = getattr(raw_query, "feedback", None)
    feedback = _normalize_feedback([raw_feedback] if isinstance(raw_feedback, str) else [])
    return raw_text, query_id, feedback


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
    query_id: str,
    feedback: list[str],
) -> StructuredQuery:
    if structured_query.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task returned by intent extractor: {structured_query.task}")

    payload = structured_query.model_dump()
    payload["query_id"] = query_id
    payload["feedback"] = _normalize_feedback(feedback, structured_query.feedback)
    payload["object_constraints"] = normalize_object_constraints(
        structured_query.object_constraints
    )
    return StructuredQuery.model_validate(payload)


def _fallback_to_kis(
    raw_text: str,
    query_id: str,
    feedback: list[str],
) -> StructuredQuery:
    logger.warning(
        "Intent extraction failed, using raw text fallback for raw_text=%r",
        raw_text,
        exc_info=True,
    )
    return StructuredQuery(
        query_id=query_id,
        task="KIS",
        visual_queries=[raw_text],
        feedback=feedback,
    )


def extract_intent_sync(
    raw_query: Any,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = LLM_CONFIG.temperature,
) -> StructuredQuery:
    raw_text, query_id, feedback = _coerce_raw_query(raw_query)
    client = client or build_instructor_client()

    try:
        structured_query = client.chat.completions.create(
            model=model,
            response_model=StructuredQuery,
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _render_prompt(
                            extract_structured_query_prompt(),
                            raw_text,
                            query_id,
                            "\n".join(feedback),
                        ),
                    }
                ]
            ),
            max_retries=max_retries,
            temperature=temperature,
        )
        return _normalize_structured_query(
            structured_query,
            query_id=query_id,
            feedback=feedback,
        )
    except InstructorRetryException as error:
        parsed_query = _parse_stripped_completion(error, StructuredQuery)
        if parsed_query is not None:
            try:
                return _normalize_structured_query(
                    parsed_query,
                    query_id=query_id,
                    feedback=feedback,
                )
            except ValueError:
                logger.warning("Intent extractor returned an unsupported task", exc_info=True)
        return _fallback_to_kis(raw_text, query_id, feedback)
    except ValueError:
        logger.warning("Intent extractor returned an unsupported task", exc_info=True)
        return _fallback_to_kis(raw_text, query_id, feedback)


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
    "SUPPORTED_TASKS",
    "build_instructor_client",
    "extract_intent",
    "extract_intent_sync",
]
