import json
from typing import Literal

from pydantic import BaseModel

from online_pipeline.intent_extractor.schemas import KISQuery, TaskClassification, VQAQuery


TaskName = Literal["KIS", "VQA"]


def classify_task_prompt() -> str:
    schema = json.dumps(
        TaskClassification.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are the task classifier for an online multimodal video retrieval pipeline.

Classify the user query into exactly one task:
- KIS: the user wants to find a known scene, object, action, text, speech, or moment in video.
- VQA: the user asks a question that must be answered from retrieved visual/video evidence.

Return only valid JSON matching this JSON Schema:
{schema}

User query:
{{raw_query}}
"""


def extract_query_prompt(task: TaskName) -> str:
    schemas: dict[TaskName, type[BaseModel]] = {
        "KIS": KISQuery,
        "VQA": VQAQuery,
    }

    try:
        schema_model = schemas[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported intent extraction task: {task}") from exc

    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False, indent=2)

    if task == "KIS":
        return f"""You extract a structured KIS retrieval query from a raw user query.

Focus on searchable evidence in the video: visible objects, scenes, actions, OCR text,
spoken words, and constraints that should be excluded.

Return only valid JSON matching this JSON Schema:
{schema}

User query:
{{raw_query}}
"""

    return f"""You extract a structured VQA retrieval query from a raw user question.

Preserve the user's question as the answer target, and add only retrieval hints that help
find evidence for answering it.

Return only valid JSON matching this JSON Schema:
{schema}

User query:
{{raw_query}}
"""


__all__ = [
    "TaskName",
    "classify_task_prompt",
    "extract_query_prompt",
]
