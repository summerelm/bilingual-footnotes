"""EPUB-owned anchors and alignment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bilingual_text_align import AlignmentLink, TextUnit


@dataclass(frozen=True)
class Anchor:
    document: str
    block: int
    start: int
    end: int


@dataclass(frozen=True)
class Book:
    path: str
    sha256: str
    units: tuple[TextUnit, ...]
    anchors: tuple[Anchor, ...]


@dataclass(frozen=True)
class ScopeSelection:
    """Corresponding global ranges selected before fine alignment."""

    classification: str
    base_start: int
    base_end: int
    footnote_start: int
    footnote_end: int
    evidence: dict[str, object]


@dataclass(frozen=True)
class Artifact:
    schema_version: int
    base_sha256: str
    footnote_sha256: str
    aligner: dict[str, object]
    base_units: tuple[TextUnit, ...]
    base_anchors: tuple[Anchor, ...]
    footnote_units: tuple[TextUnit, ...]
    links: tuple[AlignmentLink, ...]
    scope: ScopeSelection | None = None
    verification: dict[str, object] | None = None
    run: dict[str, object] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
