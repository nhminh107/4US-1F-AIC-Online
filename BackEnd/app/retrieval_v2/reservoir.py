from __future__ import annotations

from collections.abc import Callable, Iterable

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.retrieval_v2.contracts import QueryAtom


CanonicalKey = tuple[str, int]


def canonical_key(hit: SearchHit) -> CanonicalKey:
    midpoint = (hit.start_ms + hit.end_ms) // 2
    return hit.video_id, midpoint // 2_000


def _group_score(hits: list[SearchHit], atom_weights: dict[str, float]) -> float:
    independent_support: dict[tuple[str | None, str], float] = {}
    for hit in hits:
        family = hit.retriever_family or hit.source
        weight = atom_weights.get(hit.atom_id or "", 1.0)
        contribution = weight / (10.0 + hit.rank)
        support_key = (hit.atom_id, family)
        independent_support[support_key] = max(
            independent_support.get(support_key, 0.0),
            contribution,
        )
    atom_count = len({hit.atom_id for hit in hits if hit.atom_id})
    event_count = len({hit.event_id for hit in hits if hit.event_id})
    family_count = len({hit.retriever_family or hit.source for hit in hits})
    return (
        sum(independent_support.values())
        + 0.015 * max(0, atom_count - 1)
        + 0.010 * max(0, event_count - 1)
        + 0.005 * max(0, family_count - 1)
    )


def select_candidate_reservoir(
    hits: list[SearchHit],
    atoms: Iterable[QueryAtom] = (),
    *,
    limit: int,
) -> list[SearchHit]:
    """Select canonical moments with protected evidence and diversity quotas."""

    if limit <= 0 or not hits:
        return []
    groups: dict[CanonicalKey, list[SearchHit]] = {}
    for hit in hits:
        groups.setdefault(canonical_key(hit), []).append(hit)

    atom_weights = {atom.atom_id: atom.discriminative_weight for atom in atoms}
    scores = {key: _group_score(group_hits, atom_weights) for key, group_hits in groups.items()}
    ordered_keys = sorted(
        groups,
        key=lambda key: (
            -scores[key],
            min(hit.rank for hit in groups[key]),
            key[0],
            key[1],
        ),
    )
    selected: list[CanonicalKey] = []
    selected_set: set[CanonicalKey] = set()

    def add(keys: Iterable[CanonicalKey]) -> None:
        for key in keys:
            if len(selected) >= limit:
                return
            if key not in selected_set:
                selected.append(key)
                selected_set.add(key)

    def matching(value_of: Callable[[SearchHit], str | None]) -> dict[str, list[CanonicalKey]]:
        values = list(dict.fromkeys(
            value
            for hit in hits
            if (value := value_of(hit)) is not None
        ))
        return {
            value: [
                key
                for key in ordered_keys
                if any(value_of(hit) == value for hit in groups[key])
            ]
            for value in values
        }

    # Atom quotas are the strongest protection. Round-robin by depth prevents
    # one generic prompt from consuming the entire 1600-2000 moment reservoir.
    by_atom = matching(lambda hit: hit.atom_id)
    atom_depth = max(1, min(25, limit // max(1, 2 * len(by_atom))))
    for depth in range(atom_depth):
        depth_keys = [keys[depth] for keys in by_atom.values() if depth < len(keys)]
        add(sorted(set(depth_keys), key=lambda key: ordered_keys.index(key)))

    # Ensure each sequence event and each independent retrieval family has a
    # chance to seed local search even when its rank is initially weak.
    for value_map in (
        matching(lambda hit: hit.event_id),
        matching(lambda hit: hit.retriever_family or hit.source),
    ):
        add(keys[0] for keys in value_map.values() if keys)

    # Preserve broad video diversity after evidence quotas. Limiting this lane
    # leaves most capacity for consensus-ranked moments inside likely videos.
    best_by_video: dict[str, CanonicalKey] = {}
    for key in ordered_keys:
        best_by_video.setdefault(key[0], key)
    video_quota = max(1, limit // 4)
    add(list(best_by_video.values())[:video_quota])
    add(ordered_keys)

    return [hit for key in selected for hit in groups[key]]


__all__ = ["canonical_key", "select_candidate_reservoir"]
