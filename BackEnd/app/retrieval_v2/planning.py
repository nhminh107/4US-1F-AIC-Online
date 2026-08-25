from __future__ import annotations

import re

from BackEnd.app.contracts.models import StructuredQuery
from BackEnd.app.retrieval_v2.contracts import (
    AtomLink,
    PromptVariant,
    QueryAtom,
    RetrievalPlan,
)
from BackEnd.app.retrieval_v2.query_compiler import SemanticAtomSpec, compile_visual_clause
from BackEnd.app.retrieval_v2.answer_spec import infer_answer_spec
from BackEnd.app.retrieval_v2.corpus_stats import tokenize
from BackEnd.app.retrieval_v2.prompt_planner import (
    CorpusAwarePromptPlanner,
    SemanticAtom,
)


_COMMON_TERMS = {
    "a", "an", "the", "in", "on", "at", "of", "and", "with", "to",
    "person", "people", "man", "woman", "walking", "outside", "outdoor",
    "rain", "scene", "người", "đi", "bộ", "cảnh", "ngoài", "trời", "mưa",
    "có", "một", "hai", "và", "đang", "trong", "sau", "đó",
}
_TEMPORAL_SPLIT = re.compile(
    r"[.;]|\b(?:then|afterwards|next|finally|sau đó|tiếp theo|cuối cùng)\b",
    flags=re.IGNORECASE,
)
_ATOMIC_VISUAL_SPLIT = re.compile(
    r"[.;]|\s*,\s*(?:and\s+)?|"
    r"\b(?:then|afterwards|next|finally|and then|and finally|"
    r"sau đó|tiếp theo|cuối cùng)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_TEMPORAL_TRANSITION = re.compile(
    r"\b(?:then|afterwards|next|finally|and then|and finally|"
    r"sau đó|tiếp theo|cuối cùng)\b|;|\.\s+\S",
    flags=re.IGNORECASE,
)

_VISUAL_ALLOWED = ["frame_search", "clip_search", "shot_search"]
_VISUAL_FORBIDDEN = ["ocr_search", "asr_search"]
_OCR_ALLOWED = ["ocr_search"]
_OCR_FORBIDDEN = ["asr_search", "frame_search", "clip_search", "shot_search"]
_ASR_ALLOWED = ["asr_search"]
_ASR_FORBIDDEN = ["ocr_search", "frame_search", "clip_search", "shot_search"]
_OBJECT_ALLOWED = ["object_search", "track_search", "frame_search"]
_OBJECT_FORBIDDEN = ["ocr_search", "asr_search"]

_MODALITY_ROUTING = {
    "visual": (_VISUAL_ALLOWED, _VISUAL_FORBIDDEN, "MOMENT"),
    "ocr": (_OCR_ALLOWED, _OCR_FORBIDDEN, "FRAME"),
    "asr": (_ASR_ALLOWED, _ASR_FORBIDDEN, "MOMENT"),
    "object": (_OBJECT_ALLOWED, _OBJECT_FORBIDDEN, "FRAME"),
}

_MAX_VISUAL_ATOMS_PER_SCOPE = 6
_PROMPT_ROLES_BY_ATOM_TYPE = {
    "ENTITY": {"global", "rare_detail", "contrast"},
    "ACTION": {"global", "rare_detail", "action", "contrast"},
    "CONTEXT": {"global", "rare_detail", "context"},
    "ATTRIBUTE": {"global", "rare_detail", "contrast"},
    "RELATION": {"global", "rare_detail", "contrast"},
    "COUNT": {"global", "rare_detail", "contrast"},
}


def _tokens(text: str) -> list[str]:
    return list(tokenize(text))


def estimate_discriminative_weight(text: str) -> float:
    """Estimate how useful a phrase is for separating videos in this corpus."""

    tokens = _tokens(text)
    if not tokens:
        return 0.25
    informative = [token for token in tokens if token not in _COMMON_TERMS]
    informative_ratio = len(informative) / len(tokens)
    phrase_bonus = min(len(set(informative)) / 10.0, 0.75)
    specificity_bonus = 0.2 if any(len(token) >= 8 for token in informative) else 0.0
    return round(min(2.0, 0.25 + informative_ratio + phrase_bonus + specificity_bonus), 3)


def _rare_clause(text: str) -> str:
    clauses = [part.strip(" ,-") for part in _TEMPORAL_SPLIT.split(text)]
    clauses = [part for part in clauses if part]
    if not clauses:
        return text
    return max(clauses, key=lambda part: (estimate_discriminative_weight(part), len(part)))


def _atomic_visual_clauses(text: str) -> list[str]:
    """Split an event into independently retrievable visual clauses.

    Intent extraction normally emits concise English phrases, but ordered KIS and
    TRAKE events can still contain several actions separated by punctuation or
    temporal connectives. Keeping those actions in one CLIP prompt makes one easy
    noun hide the missing parts of the event.
    """

    clauses: list[str] = []
    for raw_clause in _ATOMIC_VISUAL_SPLIT.split(text):
        clause = re.sub(r"^(?:and|then)\s+", "", raw_clause.strip(" ,-"), flags=re.IGNORECASE)
        if not clause:
            continue
        if clause.casefold() not in {item.casefold() for item in clauses}:
            clauses.append(clause)
    return clauses or [text.strip()]


def _visual_prompts(
    text: str,
    semantic: SemanticAtomSpec | None,
    planner: CorpusAwarePromptPlanner,
) -> list[PromptVariant]:
    if not text.isascii():
        raise ValueError(
            "Visual query atoms must be translated to English before CLIP planning; "
            "OCR/ASR constraints may remain in the source language"
        )
    rare_details = []
    if semantic is not None:
        if semantic.object:
            rare_details.append(semantic.object)
        rare_details.extend(
            f"{value} {semantic.object or key}"
            for key, value in semantic.attributes.items()
        )
        if semantic.relation or semantic.count is not None:
            rare_details.append(semantic.text)
    variants = planner.plan(
        SemanticAtom(
            text_en=text,
            subject=semantic.subject if semantic else None,
            action=(semantic.text if semantic and semantic.atom_type == "ACTION" else None),
            context=(semantic.text if semantic and semantic.atom_type in {"CONTEXT", "RELATION"} else None),
            rare_details=tuple(dict.fromkeys(rare_details)) or (text,),
            positive_discriminators=(text,),
        )
    )
    allowed_roles = _PROMPT_ROLES_BY_ATOM_TYPE.get(
        semantic.atom_type if semantic is not None else "CONTEXT",
        {"global", "rare_detail", "contrast"},
    )
    return [variant for variant in variants if variant.role in allowed_roles]


def _bounded_visual_semantics(text: str) -> list[tuple[SemanticAtomSpec, str | None]]:
    """Keep type diversity and one composite retrieval group within a hard cap."""

    clauses = _atomic_visual_clauses(text)
    candidates: list[tuple[SemanticAtomSpec, str | None]] = []
    for clause_index, clause in enumerate(clauses, start=1):
        specs = compile_visual_clause(clause)
        group_id = f"G{clause_index}" if len(specs) > 1 else None
        candidates.extend((spec, group_id) for spec in specs)
        if group_id is not None:
            candidates.append((
                SemanticAtomSpec(
                    text=clause,
                    atom_type="CONTEXT",
                    confidence=0.75,
                ),
                group_id,
            ))

    deduplicated: list[tuple[SemanticAtomSpec, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for spec, group_id in candidates:
        key = (spec.atom_type, spec.text.casefold())
        if key not in seen:
            seen.add(key)
            deduplicated.append((spec, group_id))

    type_order = ("ACTION", "ATTRIBUTE", "COUNT", "RELATION", "ENTITY", "CONTEXT")
    selected: list[tuple[SemanticAtomSpec, str | None]] = []
    for atom_type in type_order:
        matches = [item for item in deduplicated if item[0].atom_type == atom_type]
        if matches:
            def selection_score(item: tuple[SemanticAtomSpec, str | None]) -> float:
                spec, group_id = item
                score = estimate_discriminative_weight(spec.text)
                if atom_type != "ACTION":
                    covered_groups = {
                        selected_group_id
                        for _, selected_group_id in selected
                        if selected_group_id is not None
                    }
                    if group_id is not None and group_id not in covered_groups:
                        score += 2.00
                    if atom_type == "RELATION" and any(
                        selected_spec.atom_type == "COUNT"
                        and selected_group_id == group_id
                        for selected_spec, selected_group_id in selected
                    ):
                        score -= 2.00
                    return score
                if group_id is None:
                    return score

                sibling_types = {
                    sibling.atom_type
                    for sibling, sibling_group_id in deduplicated
                    if sibling_group_id == group_id and sibling is not spec
                }
                # Actions tied to a distinctive attribute, count, or relation are
                # stronger retrieval pivots than generic motion at the start of a
                # sentence. This keeps details such as "bite a pumpkin with a
                # yellow flower" inside the bounded atom budget.
                score += 0.40 if "ATTRIBUTE" in sibling_types else 0.0
                score += 0.20 if "COUNT" in sibling_types else 0.0
                score += 0.10 if "RELATION" in sibling_types else 0.0
                if {"COUNT", "RELATION"}.issubset(sibling_types):
                    score -= 0.70
                if " and " in spec.text.casefold() and len(spec.text.split()) <= 16:
                    score += 0.50
                return score

            matches.sort(
                key=lambda item: (
                    -selection_score(item),
                    -len(item[0].text),
                )
            )
            type_limit = 2 if atom_type == "ACTION" else 1
            selected.extend(matches[:type_limit])
        if len(selected) == _MAX_VISUAL_ATOMS_PER_SCOPE:
            break
        if len(selected) > _MAX_VISUAL_ATOMS_PER_SCOPE:
            selected = selected[:_MAX_VISUAL_ATOMS_PER_SCOPE]
            break
    if len(selected) < _MAX_VISUAL_ATOMS_PER_SCOPE:
        remaining = [item for item in deduplicated if item not in selected]
        remaining.sort(
            key=lambda item: (
                -estimate_discriminative_weight(item[0].text),
                -len(item[0].text),
            )
        )
        selected.extend(remaining[: _MAX_VISUAL_ATOMS_PER_SCOPE - len(selected)])
    return selected


def _profile(query: StructuredQuery, atoms: list[QueryAtom] | None = None) -> str:
    if query.task == "TRAKE":
        return "TRAKE"
    if query.task == "VQA":
        return "VQA"
    event_ids = {
        atom.event_id
        for atom in (atoms or [])
        if atom.event_id is not None
    }
    return "KIS_SEQUENCE" if len(query.events) > 1 or len(event_ids) > 1 else "KIS_MOMENT"


def build_retrieval_plan(
    query: StructuredQuery,
    *,
    prompt_planner: CorpusAwarePromptPlanner | None = None,
) -> RetrievalPlan:
    atoms: list[QueryAtom] = []
    resolved_prompt_planner = prompt_planner or CorpusAwarePromptPlanner()

    def add(
        text: str,
        modality: str,
        event_id: str | None = None,
        *,
        semantic: SemanticAtomSpec | None = None,
        role: str = "REQUIRED",
        operator: str = "MUST",
        links: list[AtomLink] | None = None,
        scope: str = "EVENT",
        group_id: str | None = None,
    ) -> str:
        """Add an atom and return its atom_id."""
        cleaned = text.strip()
        if not cleaned:
            return ""
        effective_role = (
            "SUPPORTING"
            if semantic is not None and semantic.confidence < 0.8 and role == "REQUIRED"
            else role
        )
        prompts = (
            _visual_prompts(cleaned, semantic, resolved_prompt_planner)
            if modality == "visual"
            else []
        )
        allowed, forbidden, granularity = _MODALITY_ROUTING[modality]
        atom_id = f"A{len(atoms) + 1}"
        discriminative_weight = estimate_discriminative_weight(cleaned)
        stats = resolved_prompt_planner.corpus_stats
        if stats is not None and stats.document_count > 0:
            max_idf = stats.idf("__unseen_retrieval_term__")
            rarity = min(1.0, stats.phrase_idf(cleaned) / max_idf)
            discriminative_weight = min(
                2.0,
                0.7 * discriminative_weight + 0.3 * (0.5 + 1.5 * rarity),
            )
        atoms.append(
            QueryAtom(
                atom_id=atom_id,
                event_id=event_id,
                scope=scope,
                group_id=group_id,
                text=cleaned,
                modality=modality,
                atom_type=(semantic.atom_type if semantic else ("TEXT" if modality in {"ocr", "asr"} else "CONTEXT")),
                subject=semantic.subject if semantic else None,
                predicate=semantic.predicate if semantic else None,
                object=semantic.object if semantic else None,
                count=semantic.count if semantic else None,
                attributes=semantic.attributes if semantic else {},
                relation=semantic.relation if semantic else None,
                parse_confidence=semantic.confidence if semantic else 1.0,
                required=effective_role == "REQUIRED" and operator != "MUST_NOT",
                operator=operator,
                role=effective_role,
                granularity=granularity,
                discriminative_weight=discriminative_weight,
                prompt_variants=prompts,
                allowed_retrievers=list(allowed),
                forbidden_retrievers=list(forbidden),
                links=links or [],
            )
        )
        return atom_id

    if query.events:
        anchor_text = ". ".join(query.visual_queries)
        for semantic, group_id in _bounded_visual_semantics(anchor_text)[:4]:
            add(
                semantic.text,
                "visual",
                semantic=semantic,
                role="SUPPORTING",
                scope="VIDEO_ANCHOR",
                group_id=group_id,
            )
        for event in query.events:
            for semantic, group_id in _bounded_visual_semantics(event.description):
                add(
                    semantic.text,
                    "visual",
                    event.event_id,
                    semantic=semantic,
                    scope="EVENT",
                    group_id=f"{event.event_id}:{group_id}" if group_id else None,
                )
    else:
        synthetic_event_index = 0
        visual_inputs = query.visual_queries
        if not any(_EXPLICIT_TEMPORAL_TRANSITION.search(text) for text in visual_inputs):
            visual_inputs = [". ".join(visual_inputs)] if visual_inputs else []
        for text in visual_inputs:
            clauses = _atomic_visual_clauses(text)
            has_explicit_sequence = bool(_EXPLICIT_TEMPORAL_TRANSITION.search(text))
            for clause in clauses:
                event_id = None
                if len(clauses) > 1 and has_explicit_sequence:
                    synthetic_event_index += 1
                    event_id = f"E{synthetic_event_index}"
                for semantic, group_id in _bounded_visual_semantics(clause):
                    add(
                        semantic.text,
                        "visual",
                        event_id,
                        semantic=semantic,
                        scope="ANSWER_EVIDENCE" if query.task == "VQA" else "EVENT",
                        group_id=(f"{event_id}:{group_id}" if event_id and group_id else group_id),
                    )

    for text in query.ocr_constraints:
        ocr_id = add(
            text,
            "ocr",
            scope="ANSWER_EVIDENCE" if query.task == "VQA" else "EVENT",
        )
        # Link OCR atoms to the nearest visual atom via same_moment if exists
        if ocr_id and atoms:
            visual_atoms = [a for a in atoms if a.modality == "visual"]
            if visual_atoms:
                target = visual_atoms[-1]
                atoms[-1] = atoms[-1].model_copy(update={
                    "links": [AtomLink(
                        target_atom_id=target.atom_id,
                        relation="same_moment",
                    )],
                })

    for text in query.asr_constraints:
        add(text, "asr", scope="ANSWER_EVIDENCE" if query.task == "VQA" else "EVENT")

    for text in query.object_constraints:
        semantic_specs = compile_visual_clause(text) if text.isascii() else []
        semantic = next(
            (spec for spec in semantic_specs if spec.atom_type == "COUNT"),
            semantic_specs[0] if semantic_specs else None,
        )
        add(
            semantic.text if semantic else text,
            "object",
            semantic=semantic,
            scope="ANSWER_EVIDENCE" if query.task == "VQA" else "EVENT",
        )

    # Negative constraints as MUST_NOT atoms
    for text in getattr(query, "negative_constraints", []) or []:
        add(text, "visual", operator="MUST_NOT", role="REQUIRED")

    if not atoms and query.question:
        add(
            query.question,
            "visual",
            scope="ANSWER_EVIDENCE" if query.task == "VQA" else "EVENT",
        )

    return RetrievalPlan(
        query_id=query.query_id,
        task=query.task,
        execution_profile=_profile(query, atoms),
        atoms=atoms,
        answer_spec=(
            infer_answer_spec(
                query.question,
                has_ocr=bool(query.ocr_constraints),
                has_asr=bool(query.asr_constraints),
            )
            if query.task == "VQA"
            else None
        ),
        temporal_constraints=query.temporal_constraints,
    )


__all__ = ["build_retrieval_plan", "estimate_discriminative_weight"]
