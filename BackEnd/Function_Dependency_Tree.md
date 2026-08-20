# Function And Dependency Tree

## Module tree

```text
online_pipeline/
├── pipeline.py
│   ├── RawQuery
│   └── run_pipeline
├── intent_extractor/
│   ├── schemas.py: legacy TaskClassification compatibility type (KIS/VQA/TRAKE)
│   ├── prompts.py: extract_structured_query_prompt
│   └── extractor.py: build_instructor_client, extract_intent_sync, extract_intent
├── fast_path/
│   └── runner.py: _planned_calls, _run_with_timeout, run_fast_path
├── query_planner/
│   ├── schemas.py: *SearchParams, ToolName, TOOL_PARAMS, ToolCall
│   ├── planner.py: build_instructor_client, _task_guidance, _build_prompt,
│   │              _run_query_planner_sync, run_query_planner
│   └── executor.py: _dispatch_tool, _execute_with_timeout,
│                   _with_event_id, execute_tool_calls
├── retrieval_tools/
│   ├── visual.py: embed_text, search_faiss, resolve_entity,
│   │              clip_search, frame_search, shot_search
│   ├── text.py: embed_text, ocr_search, asr_search, caption_search
│   └── object.py: object_search, track_search
└── shared/
    ├── contracts.py: SearchHit and legacy shared contracts
    └── config.py: TOOL_TIMEOUTS, TOP_K_DEFAULTS, LLM_CONFIG
```

## Runtime dependency tree

```text
run_pipeline(RawQuery)
├── extract_intent(RawQuery)
│   ├── instructor.from_openai(OpenAI())
│   └── extract_structured_query_prompt() -> StructuredQuery
├── asyncio.gather(...)
│   ├── run_fast_path(StructuredQuery)
│   │   ├── clip_search -> embed_text -> search_faiss -> resolve_entity
│   │   ├── ocr_search -> AsyncElasticsearch(ocr alias)
│   │   └── asr_search -> AsyncElasticsearch(transcript alias)
│   └── run_query_planner(StructuredQuery)
│       ├── instructor.from_openai(OpenAI())
│       └── list[ToolCall]
├── execute_tool_calls(list[ToolCall])
│   ├── clip_search
│   ├── frame_search
│   ├── shot_search
│   ├── ocr_search
│   ├── asr_search
│   ├── object_search -> PostgreManager.search_object_detections
│   └── track_search -> PostgreManager.search_object_tracks
└── return fast_hits + agent_hits
```

## Function contracts

| Function | Input | Output | Failure behavior |
|---|---|---|---|
| `extract_intent` | raw text or object with `.text` | `StructuredQuery` | KIS fallback after Instructor retry |
| `run_fast_path` | `StructuredQuery` | `list[SearchHit]` | log and skip failed tool |
| `run_query_planner` | `StructuredQuery` | `list[ToolCall]` | log and return `[]` |
| `execute_tool_calls` | `list[ToolCall]` | `list[SearchHit]` | log and skip failed call |
| `clip_search` | query, top_k, event_id | `list[SearchHit]` | backend placeholder may raise |
| `frame_search` | query, top_k, event_id | `list[SearchHit]` | backend placeholder may raise |
| `shot_search` | query, top_k, event_id | `list[SearchHit]` | backend placeholder may raise |
| `ocr_search` | query, top_k, mode, event_id | `list[SearchHit]` | ES/config error raises to caller |
| `asr_search` | query, top_k, mode, event_id | `list[SearchHit]` | ES/config error raises to caller |
| `caption_search` | query, top_k, mode, event_id | `list[SearchHit]` | ES/config error raises to caller |
| `object_search` | object class, top_k, min_count, event_id | `list[SearchHit]` | PostgreSQL error raises to caller |
| `track_search` | object class, top_k, relation, event_id | `list[SearchHit]` | PostgreSQL error raises to caller |

## Shared data flow

```text
RawQuery.text + RawQuery.feedback
  -> single-shot StructuredQuery prompt
  -> StructuredQuery
  -> ToolCall.parameters
  -> retrieval tool
  -> SearchHit
  -> pipeline merge
```

`SearchHit` fields:

```text
source, entity_type, entity_id, video_id,
start_ms, end_ms, rank, raw_score, event_id
```

## Configuration dependencies

```text
LLM_CONFIG
├── intent_extractor.extractor
└── query_planner.planner

TOP_K_DEFAULTS
└── query_planner.schemas

TOOL_TIMEOUTS
├── fast_path.runner
└── query_planner.executor
```

## Important v1 boundary

`pipeline.py` intentionally stops after concatenating Fast Path and agent hits. Candidate Aggregation, deduplication, temporal merging and ranking belong to a later module.
