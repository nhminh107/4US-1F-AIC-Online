"""Enums used by the Selective Verifier."""

from __future__ import annotations

from enum import Enum


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    NOT_CHECKED = "NOT_CHECKED"


class ClaimImportance(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class ClaimType(str, Enum):
    VISUAL_ENTITY = "visual_entity"
    VISUAL_ATTRIBUTE = "visual_attribute"
    ACTION = "action"
    SCENE = "scene"
    OCR_EXACT = "ocr_exact"
    OCR_SEMANTIC = "ocr_semantic"
    ASR_EXACT = "asr_exact"
    ASR_SEMANTIC = "asr_semantic"
    OBJECT_PRESENCE = "object_presence"
    OBJECT_COUNT = "object_count"
    SPATIAL_RELATION = "spatial_relation"
    TEMPORAL_ORDER = "temporal_order"
    TEMPORAL_GAP = "temporal_gap"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    VQA_ANSWER_CLAIM = "vqa_answer_claim"
    TRAKE_EVENT = "trake_event"
    KIS_MOMENT = "kis_moment"


class VerificationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"
    NEED_REPLAN = "NEED_REPLAN"


class VerificationLevel(str, Enum):
    SKIPPED = "skipped"
    DETERMINISTIC = "deterministic"
    RERANKER = "reranker"
    VLM = "vlm"


class NextAction(str, Enum):
    RETURN_RESULT = "RETURN_RESULT"
    TRY_NEXT_CANDIDATE = "TRY_NEXT_CANDIDATE"
    EXPAND_LOCAL_CONTEXT = "EXPAND_LOCAL_CONTEXT"
    REPLAN = "REPLAN"
