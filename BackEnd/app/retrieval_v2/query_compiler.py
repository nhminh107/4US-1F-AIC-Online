from __future__ import annotations

import re
from dataclasses import dataclass, field


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COLORS = (
    "black|blue|brown|green|grey|gray|orange|pink|purple|red|white|yellow"
)
_ACTION_RE = re.compile(
    r"\b(stands?|spins?|jumps?|dives?|bites?|biting|holds?|holding|walks?|walking|"
    r"rests?|resting|climbs?|climbing|weighs?|weighing|records?|recording|"
    r"adds?|adding|appears?|showing|shows?|pours?|pouring|takes?|taking|"
    r"talks?|talking|continues?|pans?|zooming|zooms?|puts?|putting|places?|"
    r"picks?|lifting|lifts?|carries?|wearing|wears?|opens?|closes|turns?|"
    r"rotates?|falling|falls?|rising|rises?|drives?|riding|rides?|sits?)\b",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"\b(on top of|in front of|next to|beside|behind|under|above|across|inside|within)\b"
    r"\s+([^,.;]+)",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"((?:[a-z]+\s+){0,2}[a-z]+)",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    rf"\b({_COLORS})\s+([a-z][a-z-]*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticAtomSpec:
    text: str
    atom_type: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    count: int | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    relation: str | None = None
    confidence: float = 1.0


def _clean_noun_phrase(value: str) -> str:
    cleaned = re.sub(
        r"\b(?:am|are|is|was|were|be|been|being)$",
        "",
        value.strip(" ,-"),
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def compile_visual_clause(clause: str) -> list[SemanticAtomSpec]:
    """Compile one English visual clause into independently testable evidence.

    This deliberately stays conservative. A shallow parse may create soft typed
    evidence, while unsupported grammar remains one CONTEXT atom instead of a
    fabricated hard relation.
    """

    cleaned = clause.strip(" ,-.")
    if not cleaned:
        return []

    specs: list[SemanticAtomSpec] = []
    action_matches = list(_ACTION_RE.finditer(cleaned))
    action_match = action_matches[0] if action_matches else None
    subject: str | None = None
    if action_match:
        subject = _clean_noun_phrase(cleaned[: action_match.start()])
        subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
        if subject:
            specs.append(
                SemanticAtomSpec(
                    text=subject,
                    atom_type="ENTITY",
                    subject=subject,
                    confidence=0.9,
                )
            )
        if len(action_matches) > 1:
            specs.append(
                SemanticAtomSpec(
                    text=cleaned,
                    atom_type="ACTION",
                    subject=subject,
                    predicate="compound_action",
                    confidence=0.85,
                )
            )
        for index, match in enumerate(action_matches):
            end = action_matches[index + 1].start() if index + 1 < len(action_matches) else len(cleaned)
            action_phrase = cleaned[match.start():end].strip(" ,-.")
            action_phrase = re.sub(r"\band\s*$", "", action_phrase, flags=re.IGNORECASE).strip()
            if action_phrase:
                specs.append(
                    SemanticAtomSpec(
                        text=(f"{subject} {action_phrase}" if subject else action_phrase),
                        atom_type="ACTION",
                        subject=subject,
                        predicate=match.group(1).lower(),
                        confidence=0.9,
                    )
                )

    for match in _COUNT_RE.finditer(cleaned):
        raw_count, noun_phrase = match.groups()
        count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count.lower()]
        words = noun_phrase.split()
        # Stop a permissive regex from swallowing a following action verb.
        action_index = next(
            (index for index, word in enumerate(words) if _ACTION_RE.fullmatch(word)),
            len(words),
        )
        noun_phrase = _clean_noun_phrase(" ".join(words[:action_index]))
        if noun_phrase:
            preceding_action = next(
                (
                    action
                    for action in reversed(action_matches)
                    if action.start() < match.start()
                ),
                None,
            )
            count_text = f"{raw_count} {noun_phrase}"
            if preceding_action is not None:
                count_text = cleaned[preceding_action.start():match.end()].strip(" ,-.")
            specs.append(
                SemanticAtomSpec(
                    text=count_text,
                    atom_type="COUNT",
                    object=noun_phrase,
                    count=count,
                    confidence=0.85,
                )
            )

    for match in _RELATION_RE.finditer(cleaned):
        relation, target = match.groups()
        target = re.split(r"\b(?:and|then|to)\b", target, maxsplit=1, flags=re.IGNORECASE)[0]
        target = target.strip(" ,-.")
        if target:
            specs.append(
                SemanticAtomSpec(
                    text=f"{relation} {target}",
                    atom_type="RELATION",
                    subject=subject,
                    object=target,
                    relation=relation.lower(),
                    confidence=0.85,
                )
            )

    for match in _ATTRIBUTE_RE.finditer(cleaned):
        color, noun = match.groups()
        specs.append(
            SemanticAtomSpec(
                text=f"{color} {noun}",
                atom_type="ATTRIBUTE",
                object=noun,
                attributes={"color": color.lower()},
                confidence=0.95,
            )
        )

    if not action_matches:
        noun = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
        if len(noun.split()) <= 5 and not re.search(
            r"\b(?:in|on|at|near|inside|outside|under|above|behind)\b",
            noun,
            flags=re.IGNORECASE,
        ):
            specs.append(
                SemanticAtomSpec(
                    text=noun,
                    atom_type="ENTITY",
                    subject=noun,
                    confidence=0.85,
                )
            )
        else:
            specs.append(SemanticAtomSpec(text=cleaned, atom_type="CONTEXT", confidence=0.7))

    deduplicated: list[SemanticAtomSpec] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        key = (spec.atom_type, spec.text.casefold())
        if key not in seen:
            seen.add(key)
            deduplicated.append(spec)
    return deduplicated


__all__ = ["SemanticAtomSpec", "compile_visual_clause"]
