# Text/Object/Tracking Retrieval Tools

The reusable entry points are `TextRetrievalTools` and
`ObjectTrackingRetrievalTools` in `text_object_retrieval.py`. Every search
returns the shared `BackEnd.app.contracts.models.SearchHit` contract, ready for
Candidate Aggregator and Fusion & Ranking.

```python
text_tools = TextRetrievalTools(elasticsearch_client)
ocr_hits = text_tools.ocr_search("HCMC", top_k=100, event_id="E1")

object_tools = ObjectTrackingRetrievalTools(postgre_manager.session_factory)
allowed_classes = object_tools.supported_object_classes()
object_hits = object_tools.object_search("person", min_count=2, top_k=100)
track_hits = object_tools.track_search("person", min_duration_ms=1000)
```

The query planner must constrain object parameters with
`supported_object_classes()`. Retrieval also validates at execution time and
fails closed for an invented class. The canonical `person` query combines the
BTC labels `Person`, `Man`, `Woman`, `Boy`, and `Girl` before applying
`min_count`, avoiding split person counts.
