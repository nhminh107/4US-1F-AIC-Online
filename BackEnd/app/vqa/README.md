# VQA Handler

`VQAHandler` is the reusable entry point for Module 6B. It consumes the shared
`StructuredQuery`, `RankedCandidateRegion`, and `EvidenceBundle` contracts and
returns the shared `VQAResult`. It only calls a VLM on the bounded set of ranked
candidate frames; it never searches the full corpus.

```python
from BackEnd.app.vqa import VQAHandler

handler = VQAHandler(vlm_client, max_candidates=5)
result = handler.handle(
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

The handler returns `status="uncertain"` with zero confidence when no local
visual evidence is available. Model implementations are injected through the
`VQAModelClient` protocol so the handler is not coupled to one provider.

## FPT AI Marketplace

Configure secrets locally (never commit the real key):

```env
VQA_API_KEY=your-local-secret
VQA_BASE_URL=https://mkp-api.fptcloud.com
VQA_MODEL=your-vision-language-model-name
VQA_TIMEOUT_SECONDS=60
```

Then wire the concrete client:

```python
from BackEnd.app.vqa import FPTVLMClient, VQAHandler

handler = VQAHandler(FPTVLMClient.from_env())
result = handler.handle(query, ranked_candidates, evidence_loader)
```
