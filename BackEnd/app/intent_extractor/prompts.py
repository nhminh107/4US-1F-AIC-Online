import json

from BackEnd.app.contracts.models import StructuredQuery
from BackEnd.app.intent_extractor.object_classes import ALLOWED_OBJECT_CLASSES


def extract_structured_query_prompt() -> str:
    schema = json.dumps(
        StructuredQuery.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    allowed_object_classes = ", ".join(ALLOWED_OBJECT_CLASSES)

    return f"""You extract a complete structured query for an online multimodal video
retrieval pipeline in one response.

Return only valid JSON matching this JSON Schema:
{schema}

Use query_id exactly "{{query_id}}".
Choose task exactly "KIS", "VQA", or "TRAKE".
Do not add fields outside the schema.

Task rules:
- KIS: find a known scene, object, action, visible text, speech, or moment. Set question
  to an empty string and provide at least one useful retrieval signal.
- VQA: answer a question from video evidence. Preserve the user's question in question and
  add only retrieval hints needed to find supporting evidence.
- TRAKE: find a video containing a sequence of events in temporal order. Set question to an
  empty string, decompose the sequence into events with stable IDs E1, E2, ..., and add a
  temporal_constraints item for each stated or implied ordering relation. Use event IDs in
  before and after. Add visual, OCR, ASR, and object constraints only when they help retrieve
  the individual events.

Feedback rules:
- Feedback is an additional user correction or constraint. Apply it to all relevant
  retrieval fields. If it conflicts with the raw query, feedback takes precedence.
- Preserve the feedback text exactly as one item in the feedback list when it is non-empty.
- Put exclusions implied by feedback into negative_constraints when applicable.

Object constraint rules:
- object_constraints may contain only these exact English values:
  {allowed_object_classes}
- Translate Vietnamese object names to the matching English value above.
- Do not put scenes, places, attributes, or actions such as city, street, building,
  indoor, walking, or receiving into object_constraints. Keep them in visual_queries.
- If an object is not in the allowed list, omit it from object_constraints and preserve
  the concept in visual_queries.

Raw user query:
{{raw_query}}

User feedback (may be empty):
{{feedback}}
"""


__all__ = [
    "extract_structured_query_prompt",
]
