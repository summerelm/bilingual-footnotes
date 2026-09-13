"""Structural ports owned by the EPUB application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .model import Artifact, Book

ProgressReporter = Callable[[str], None]


class BookReader(Protocol):
    """Read an EPUB into the application-owned whole-book representation."""

    def __call__(self, path: str | Path, /) -> Book: ...


class EpubRenderer(Protocol):
    """Render a validated artifact without changing the source EPUB."""

    def __call__(
        self,
        base_epub: str | Path,
        artifact: Artifact,
        output_epub: str | Path,
        /,
    ) -> None: ...
