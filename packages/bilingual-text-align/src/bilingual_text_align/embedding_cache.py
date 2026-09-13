"""Cache boundary used by semantic embedding computation."""

from __future__ import annotations

from typing import Any, Protocol


class EmbeddingCache(Protocol):
    """Store embeddings without exposing a persistence format to the worker."""

    def load(self, key: str, expected_prefix: tuple[int, ...]) -> Any | None: ...

    def store(self, key: str, embeddings: Any) -> None: ...


class DisabledEmbeddingCache:
    """Cache implementation for callers that disable embedding persistence."""

    def load(self, key: str, expected_prefix: tuple[int, ...]) -> Any | None:
        return None

    def store(self, key: str, embeddings: Any) -> None:
        return None
