"""Canonical object vocabulary shared by Intent Extractor and retrieval."""

from __future__ import annotations


# These 30 common classes exist in the project's Open Images ClassID table and
# overlap with common YOLO vocabularies. Keep canonical values in English because
# PostgreSQL object search performs an exact case-insensitive class-name match.
ALLOWED_OBJECT_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "traffic light",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cattle",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "ball",
    "chair",
    "couch",
    "table",
    "bottle",
)


_OBJECT_CLASS_ALIASES: dict[str, str] = {
    # English variants commonly produced by an LLM.
    "people": "person",
    "human": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "motor cycle": "motorcycle",
    "cow": "cattle",
    "sofa": "couch",
    "dining table": "table",
    "sports ball": "ball",
    # Vietnamese variants needed before exact PostgreSQL class lookup.
    "người": "person",
    "con người": "person",
    "xe đạp": "bicycle",
    "ô tô": "car",
    "xe hơi": "car",
    "xe máy": "motorcycle",
    "mô tô": "motorcycle",
    "xe buýt": "bus",
    "xe bus": "bus",
    "xe tải": "truck",
    "tàu hỏa": "train",
    "tàu lửa": "train",
    "thuyền": "boat",
    "đèn giao thông": "traffic light",
    "biển báo dừng": "stop sign",
    "ghế dài": "bench",
    "chim": "bird",
    "mèo": "cat",
    "chó": "dog",
    "ngựa": "horse",
    "cừu": "sheep",
    "bò": "cattle",
    "voi": "elephant",
    "gấu": "bear",
    "ngựa vằn": "zebra",
    "hươu cao cổ": "giraffe",
    "ba lô": "backpack",
    "ô": "umbrella",
    "dù": "umbrella",
    "túi xách": "handbag",
    "va li": "suitcase",
    "vali": "suitcase",
    "quả bóng": "ball",
    "bóng": "ball",
    "ghế": "chair",
    "ghế sofa": "couch",
    "bàn": "table",
    "chai": "bottle",
}


def normalize_object_constraints(values: list[str]) -> list[str]:
    """Translate known aliases and drop values outside the supported 30 classes."""

    allowed = set(ALLOWED_OBJECT_CLASSES)
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().casefold().split())
        canonical = _OBJECT_CLASS_ALIASES.get(cleaned, cleaned)
        if canonical in allowed and canonical not in normalized:
            normalized.append(canonical)
    return normalized


__all__ = ["ALLOWED_OBJECT_CLASSES", "normalize_object_constraints"]
