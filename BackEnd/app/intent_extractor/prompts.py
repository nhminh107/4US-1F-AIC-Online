import json

from BackEnd.app.contracts.models import StructuredQuery
from BackEnd.app.intent_extractor.schemas import TaskClassification, TaskName


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
    schema = json.dumps(
        StructuredQuery.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    common_instruction = f"""Return only valid JSON matching this JSON Schema:
{schema}

Use task exactly "{task}".
Use query_id exactly "{{query_id}}".
Do not add fields outside the schema.
"""

    if task == "KIS":
        return f"""You extract a structured KIS retrieval query from a raw user query.

Focus on searchable evidence in the video: visible objects, scenes, actions, OCR text,
spoken words, and constraints that should be excluded.

{common_instruction}

User query:
{{raw_query}}
"""

    if task == "VQA":
        return f"""You extract a structured VQA retrieval query from a raw user question.

Preserve the user's question as the answer target, and add only retrieval hints that help
find evidence for answering it.

{common_instruction}

User query:
{{raw_query}}
"""

    raise ValueError(f"Unsupported intent extraction task: {task}")


__all__ = [
    "classify_task_prompt",
    "extract_query_prompt",
]
