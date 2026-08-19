from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import TypeAdapter

from online_pipeline.intent_extractor.schemas import (
    KISQuery,
    StructuredQuery,
    VQAQuery,
)
from online_pipeline.query_planner.schemas import ToolCall
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


logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "clip_search": "semantic visual retrieval for clips matching a natural-language query",
    "frame_search": "frame-level visual retrieval for specific visible content",
    "shot_search": "shot-level retrieval for scenes or coherent visual segments",
    "ocr_search": "retrieval over text detected inside video frames",
    "asr_search": "retrieval over spoken words and audio transcripts",
    "caption_search": "retrieval over generated captions describing video content",
    "object_search": "retrieval for frames containing a named object or entity",
    "track_search": "retrieval for object tracks and movement across frames",
}


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


def _parse_stripped_completion(error: Exception) -> list[ToolCall] | None:
    raw_content = _completion_content(error)
    if raw_content is None:
        return None

    try:
        cleaned_content = strip_thinking_blocks(raw_content)
        return TypeAdapter(list[ToolCall]).validate_json(cleaned_content)
    except Exception:
        return None


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


def _task_guidance(structured_query: StructuredQuery) -> str:
    if isinstance(structured_query, KISQuery):
        guidance = [
            "For KIS, prioritize clip_search for broad visual retrieval.",
        ]
        if structured_query.ocr_constraints:
            guidance.append(
                "Because ocr_constraints are present, prioritize ocr_search and use "
                "them as its text query when appropriate."
            )
        if structured_query.asr_constraints:
            guidance.append(
                "Use asr_search when asr_constraints contain a meaningful speech query."
            )
        return "\n".join(f"- {item}" for item in guidance)

    if isinstance(structured_query, VQAQuery):
        return "\n".join(
            [
                "- For VQA, prioritize clip_search to locate candidate visual regions.",
                "- Also prioritize caption_search to find regions likely to contain the answer.",
                "- Use ocr_search when the question or constraints require reading visible text.",
            ]
        )

    return "- Plan only with the supported v1 KIS and VQA task behavior."


def _build_prompt(structured_query: StructuredQuery) -> str:
    tool_list = "\n".join(
        f"- {tool}: {description}"
        for tool, description in TOOL_DESCRIPTIONS.items()
    )
    query_json = json.dumps(
        structured_query.model_dump(),
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are a single-shot query planner for a video retrieval pipeline.
Create the complete retrieval plan in one response as a JSON array of ToolCall objects.
Do not explain the plan and do not perform a multi-turn loop.

Allowed tools:
{tool_list}

Task-specific guidance:
{_task_guidance(structured_query)}

General planning rules:
- Use only the allowed tools above.
- Match every parameters object to the selected tool's schema.
- Set top_k according to the expected recall and query specificity. Do not always use
  the largest value; prefer a smaller value for narrow or highly specific queries.
- Use multiple tools only when they add a distinct retrieval signal.
- Preserve event_id only when it is present in the input.

Structured query:
{query_json}
"""


def _run_query_planner_sync(
    structured_query: StructuredQuery,
    *,
    client: Any | None = None,
) -> list[ToolCall]:
    client = client or build_instructor_client()

    try:
        return client.chat.completions.create(
            model=LLM_CONFIG.llm_model,
            response_model=list[ToolCall],
            messages=_with_fpt_system_prompt(
                [
                    {
                        "role": "user",
                        "content": _build_prompt(structured_query),
                    }
                ]
            ),
            max_retries=LLM_CONFIG.max_retries,
            temperature=LLM_CONFIG.temperature,
        )
    except InstructorRetryException as error:
        parsed_tool_calls = _parse_stripped_completion(error)
        if parsed_tool_calls is not None:
            return parsed_tool_calls
        logger.warning(
            "Query planner failed, using Fast Path only for task=%s",
            structured_query.task,
            exc_info=True,
        )
        return []


async def run_query_planner(
    structured_query: StructuredQuery,
) -> list[ToolCall]:
    return await asyncio.to_thread(_run_query_planner_sync, structured_query)


__all__ = [
    "InstructorRetryException",
    "TOOL_DESCRIPTIONS",
    "build_instructor_client",
    "run_query_planner",
]
