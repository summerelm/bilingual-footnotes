"""Dependency-free alignment domain values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextUnit:
    """One ordered unit offered to an alignment algorithm."""

    index: int
    text: str
    group: int


@dataclass(frozen=True)
class AlignmentLink:
    """A monotonic many-to-many link between unit indices."""

    first_indices: tuple[int, ...]
    second_indices: tuple[int, ...]
    score: float | None
