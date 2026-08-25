from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


SCHEMA_VERSION = "retrieval-corpus-stats-v1"
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE | re.IGNORECASE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return deterministic terms suitable for lightweight corpus statistics."""

    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    video_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")


@dataclass(frozen=True)
class SelectivityMetrics:
    """Observed dispersion of retrieval hits across videos.

    Both normalized entropy and unique-video spread approach one for broad,
    weakly selective queries. `selectivity_score` reverses that direction so a
    larger value consistently means a more useful prompt.
    """

    hit_count: int
    unique_video_count: int
    normalized_entropy: float
    unique_video_spread: float

    @property
    def selectivity_score(self) -> float:
        dispersion = (self.normalized_entropy + self.unique_video_spread) / 2.0
        return round(max(0.0, min(1.0, 1.0 - dispersion)), 6)


@dataclass
class CorpusStats:
    schema_version: str = SCHEMA_VERSION
    document_count: int = 0
    video_count: int = 0
    document_frequency: dict[str, int] = field(default_factory=dict)
    video_frequency: dict[str, int] = field(default_factory=dict)
    online_selectivity: dict[str, SelectivityMetrics] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, documents: Iterable[CorpusDocument]) -> "CorpusStats":
        document_frequency: Counter[str] = Counter()
        term_videos: dict[str, set[str]] = {}
        videos: set[str] = set()
        document_count = 0

        for document in documents:
            document_count += 1
            videos.add(document.video_id)
            terms = set(tokenize(document.text))
            document_frequency.update(terms)
            for term in terms:
                term_videos.setdefault(term, set()).add(document.video_id)

        return cls(
            document_count=document_count,
            video_count=len(videos),
            document_frequency=dict(sorted(document_frequency.items())),
            video_frequency={
                term: len(video_ids)
                for term, video_ids in sorted(term_videos.items())
            },
        )

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency, including unseen terms."""

        normalized_terms = tokenize(term)
        if not normalized_terms:
            return 1.0
        frequency = self.document_frequency.get(normalized_terms[0], 0)
        return math.log((self.document_count + 1.0) / (frequency + 1.0)) + 1.0

    def phrase_idf(self, phrase: str) -> float:
        terms = tokenize(phrase)
        if not terms:
            return 1.0
        return sum(self.idf(term) for term in set(terms)) / len(set(terms))

    @staticmethod
    def measure_selectivity(
        video_ids: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> SelectivityMetrics:
        if scores is not None and len(video_ids) != len(scores):
            raise ValueError("video_ids and scores must have the same length")
        if not video_ids:
            return SelectivityMetrics(
                hit_count=0,
                unique_video_count=0,
                normalized_entropy=0.0,
                unique_video_spread=0.0,
            )

        weights = scores if scores is not None else [1.0] * len(video_ids)
        mass_by_video: Counter[str] = Counter()
        for video_id, raw_weight in zip(video_ids, weights, strict=True):
            mass_by_video[video_id] += max(float(raw_weight), 0.0)
        if sum(mass_by_video.values()) <= 0.0:
            mass_by_video = Counter(video_ids)

        total_mass = sum(mass_by_video.values())
        probabilities = [mass / total_mass for mass in mass_by_video.values()]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        unique_count = len(mass_by_video)
        normalized_entropy = entropy / math.log(unique_count) if unique_count > 1 else 0.0

        return SelectivityMetrics(
            hit_count=len(video_ids),
            unique_video_count=unique_count,
            normalized_entropy=round(normalized_entropy, 6),
            unique_video_spread=round(unique_count / len(video_ids), 6),
        )

    def record_selectivity(
        self,
        key: str,
        video_ids: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> SelectivityMetrics:
        if not key.strip():
            raise ValueError("selectivity key must not be empty")
        metrics = self.measure_selectivity(video_ids, scores)
        self.online_selectivity[key] = metrics
        return metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_count": self.document_count,
            "video_count": self.video_count,
            "document_frequency": dict(sorted(self.document_frequency.items())),
            "video_frequency": dict(sorted(self.video_frequency.items())),
            "online_selectivity": {
                key: asdict(metrics)
                for key, metrics in sorted(self.online_selectivity.items())
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CorpusStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported corpus stats schema version {version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=version,
            document_count=int(payload.get("document_count", 0)),
            video_count=int(payload.get("video_count", 0)),
            document_frequency={
                str(term): int(count)
                for term, count in payload.get("document_frequency", {}).items()
            },
            video_frequency={
                str(term): int(count)
                for term, count in payload.get("video_frequency", {}).items()
            },
            online_selectivity={
                str(key): SelectivityMetrics(**metrics)
                for key, metrics in payload.get("online_selectivity", {}).items()
            },
        )


__all__ = [
    "CorpusDocument",
    "CorpusStats",
    "SCHEMA_VERSION",
    "SelectivityMetrics",
    "tokenize",
]
