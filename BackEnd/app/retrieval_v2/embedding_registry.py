from __future__ import annotations

from typing import Literal

from pydantic import Field

from BackEnd.app.contracts.models import ContractModel


class EmbeddingBackendSpec(ContractModel):
    backend_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    entity_type: Literal["frame", "clip", "shot", "video", "audio"]
    status: Literal["active", "shadow", "disabled"] = "shadow"
    index_version: str = "0"


class EmbeddingBackendRegistry:
    """Versioned active/shadow registry; model download/build happens offline."""

    def __init__(self) -> None:
        self._backends: dict[str, EmbeddingBackendSpec] = {}

    def register(self, spec: EmbeddingBackendSpec) -> None:
        if spec.backend_id in self._backends:
            raise ValueError(f"Embedding backend already registered: {spec.backend_id}")
        if spec.status == "active" and any(
            item.entity_type == spec.entity_type and item.status == "active"
            for item in self._backends.values()
        ):
            raise ValueError(f"An active backend already exists for {spec.entity_type}")
        self._backends[spec.backend_id] = spec

    def active(self, entity_type: str) -> EmbeddingBackendSpec:
        for spec in self._backends.values():
            if spec.entity_type == entity_type and spec.status == "active":
                return spec
        raise KeyError(f"No active embedding backend for {entity_type}")

    def promote(self, backend_id: str, *, approved: bool) -> None:
        if not approved:
            raise ValueError("Shadow backend promotion requires benchmark approval")
        target = self._backends[backend_id]
        for current_id, spec in list(self._backends.items()):
            if spec.entity_type == target.entity_type and spec.status == "active":
                self._backends[current_id] = spec.model_copy(update={"status": "shadow"})
        self._backends[backend_id] = target.model_copy(update={"status": "active"})


__all__ = ["EmbeddingBackendRegistry", "EmbeddingBackendSpec"]
