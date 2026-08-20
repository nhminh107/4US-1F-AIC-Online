# VQA Handler

`VQAHandler` prepares visual evidence for a human reviewer. Retrieval and
fusion use the VQA question to find relevant regions; this handler then returns
the same `list[KISResult]` shape used by KIS. It does not call an AI API and does
not answer the question.

```python
from BackEnd.app.vqa import VQAHandler

handler = VQAHandler(max_candidates=5)
results = handler.handle(
    structured_query,
    ranked_candidate_regions,
    evidence_loader,
)
```

`evidence_loader` must implement:

```python
def evidence_loader(video_id: str, start_ms: int, end_ms: int) -> EvidenceBundle:
    ...
```

The handler keeps only candidates that passed hard and negative constraints,
orders them by fusion score, and chooses the frame nearest each candidate's
temporal midpoint. A candidate with no frame is omitted because a human reviewer
cannot inspect it.

`FPTVLMClient` remains available as an independent client for future use, but it
is not part of the active VQA pipeline.
