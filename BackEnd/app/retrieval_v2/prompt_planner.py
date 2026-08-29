from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from BackEnd.app.retrieval_v2.contracts import PromptRole, PromptVariant
from BackEnd.app.retrieval_v2.corpus_stats import CorpusStats, SelectivityMetrics, tokenize


_NEGATION = re.compile(r"\b(?:not|no|without)\b", flags=re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_CONTEXT_PREPOSITIONS = {"at", "in", "inside", "near", "on", "outside", "under"}
_BASE_WEIGHTS: dict[PromptRole, float] = {
    "global": 1.0,
    "rare_detail": 1.25,
    "action": 1.15,
    "context": 0.85,
    "contrast": 1.05,
}


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value.strip(" ,.;:"))


def _positive_phrases(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in values if value and _NEGATION.search(value) is None)


def _positive_fallback(parts: "_SemanticParts") -> tuple[str, ...]:
    candidates = _positive_phrases((parts.subject, *parts.rare_details))
    return candidates or ("a visually specific subject-detail configuration",)


@dataclass(frozen=True)
class SemanticAtom:
    """English visual semantics supplied by a query compiler or fallback parser."""

    text_en: str
    subject: str | None = None
    action: str | None = None
    context: str | None = None
    rare_details: tuple[str, ...] = ()
    positive_discriminators: tuple[str, ...] = ()
    language: str = "en"

    def __post_init__(self) -> None:
        if self.language.casefold() != "en":
            raise ValueError("SemanticAtom visual prompts must use English input")
        fields = [
            self.text_en,
            self.subject or "",
            self.action or "",
            self.context or "",
            *self.rare_details,
            *self.positive_discriminators,
        ]
        if not _clean(self.text_en):
            raise ValueError("text_en must not be empty")
        if any(not value.isascii() for value in fields):
            raise ValueError("SemanticAtom visual prompts must use English ASCII text")
        object.__setattr__(self, "text_en", _clean(self.text_en))
        object.__setattr__(self, "subject", _clean(self.subject) if self.subject else None)
        object.__setattr__(self, "action", _clean(self.action) if self.action else None)
        object.__setattr__(
            self,
            "context",
            _clean(self.context) if self.context else None,
        )
        object.__setattr__(
            self,
            "rare_details",
            tuple(_clean(value) for value in self.rare_details if _clean(value)),
        )
        object.__setattr__(
            self,
            "positive_discriminators",
            tuple(
                _clean(value)
                for value in self.positive_discriminators
                if _clean(value)
            ),
        )


@dataclass(frozen=True)
class _SemanticParts:
    subject: str
    action: str
    context: str
    rare_details: tuple[str, ...]
    discriminators: tuple[str, ...]


class CorpusAwarePromptPlanner:
    def __init__(self, corpus_stats: CorpusStats | None = None) -> None:
        self.corpus_stats = corpus_stats

    def plan(
        self,
        atom: SemanticAtom,
        *,
        online_selectivity: Mapping[PromptRole, SelectivityMetrics] | None = None,
    ) -> list[PromptVariant]:
        parts = self._parts(atom)
        rare_details = self._rank_rare_details(parts.rare_details)
        rare_focus = rare_details[0]
        contrast_parts = _positive_phrases(parts.discriminators + rare_details)
        if not contrast_parts:
            contrast_parts = _positive_fallback(parts)

        texts: dict[PromptRole, str] = {
            "global": (
                f"Documentary view of {parts.subject} {parts.action}, "
                f"seen {parts.context}."
            ),
            "rare_detail": (
                f"Close visual evidence of {rare_focus}, associated with {parts.subject}."
            ),
            "action": f"The decisive moment when {parts.subject} {parts.action}.",
            "context": (
                f"Wide environmental view {parts.context}, with {parts.subject} visible."
            ),
            "contrast": "Distinctive visible combination: "
            + "; ".join(contrast_parts[:3])
            + ".",
        }

        variants: list[PromptVariant] = []
        for role in ("global", "rare_detail", "action", "context", "contrast"):
            text = texts[role]
            if role == "contrast" and _NEGATION.search(text):
                safe_parts = _positive_fallback(parts)
                text = "Distinctive visible combination: " + "; ".join(safe_parts) + "."
            variants.append(
                PromptVariant(
                    role=role,
                    text=text,
                    weight=self._weight(role, rare_focus, online_selectivity),
                )
            )
        return variants

    def _parts(self, atom: SemanticAtom) -> _SemanticParts:
        tokens = list(tokenize(atom.text_en))
        subject = atom.subject or " ".join(tokens[: min(4, len(tokens))])

        if atom.action:
            action = atom.action
            subject_prefix = f"{subject} "
            if action.casefold().startswith(subject_prefix.casefold()):
                action = action[len(subject_prefix):]
        else:
            action_tokens = tokens[min(2, len(tokens)) :]
            action = (
                "visibly participates in " + " ".join(action_tokens[:6])
                if action_tokens
                else "is clearly visible"
            )

        if atom.context:
            context = atom.context
        else:
            context_start = next(
                (index for index, token in enumerate(tokens) if token in _CONTEXT_PREPOSITIONS),
                max(0, len(tokens) - 4),
            )
            context = "within the scene around " + " ".join(tokens[context_start:])

        fallback_detail = " ".join(tokens[-min(5, len(tokens)) :])
        rare_details = atom.rare_details or (fallback_detail,)
        discriminators = atom.positive_discriminators or (
            f"{subject} together with {rare_details[0]}",
        )
        return _SemanticParts(
            subject=subject,
            action=action,
            context=context,
            rare_details=rare_details,
            discriminators=discriminators,
        )

    def _rank_rare_details(self, details: tuple[str, ...]) -> tuple[str, ...]:
        if self.corpus_stats is None:
            return details
        return tuple(
            sorted(
                details,
                key=lambda detail: (
                    -self.corpus_stats.phrase_idf(detail),
                    detail.casefold(),
                ),
            )
        )

    def _weight(
        self,
        role: PromptRole,
        rare_focus: str,
        online_selectivity: Mapping[PromptRole, SelectivityMetrics] | None,
    ) -> float:
        weight = _BASE_WEIGHTS[role]
        if role in {"rare_detail", "contrast"} and self.corpus_stats is not None:
            max_idf = self.corpus_stats.idf("__unseen_retrieval_term__")
            rarity = min(1.0, self.corpus_stats.phrase_idf(rare_focus) / max_idf)
            weight += 0.25 * rarity
        if online_selectivity and role in online_selectivity:
            weight *= 0.8 + 0.4 * online_selectivity[role].selectivity_score
        return round(max(0.0, min(2.0, weight)), 3)


__all__ = ["CorpusAwarePromptPlanner", "SemanticAtom"]
