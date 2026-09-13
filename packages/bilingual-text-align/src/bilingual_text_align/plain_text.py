"""Plain-text input and stable mapping output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from .protocol import AlignmentAlgorithm
from .segmentation import segment_text
from .service import align_units


class FootnoteSource(StrEnum):
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True)
class MappingEntry:
    base_indices: tuple[int, ...]
    footnote_indices: tuple[int, ...]
    base_text: tuple[str, ...]
    footnote_text: tuple[str, ...]
    score: float | None


@dataclass(frozen=True)
class TextMapping:
    schema_version: int
    footnote_source: str
    aligner: dict[str, object]
    mappings: tuple[MappingEntry, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def align_texts(
    first_text: str,
    second_text: str,
    *,
    footnote_source: FootnoteSource,
    aligner: AlignmentAlgorithm,
    unit: str = "sentence",
) -> TextMapping:
    first = segment_text(first_text, unit)
    second = segment_text(second_text, unit)
    links = align_units(first, second, aligner)
    entries = []
    for link in links:
        first_texts = tuple(first[index].text for index in link.first_indices)
        second_texts = tuple(second[index].text for index in link.second_indices)
        if footnote_source is FootnoteSource.SECOND:
            base_indices, note_indices = link.first_indices, link.second_indices
            base_text, note_text = first_texts, second_texts
        else:
            base_indices, note_indices = link.second_indices, link.first_indices
            base_text, note_text = second_texts, first_texts
        entries.append(MappingEntry(base_indices, note_indices, base_text, note_text, link.score))
    return TextMapping(1, footnote_source.value, dict(aligner.metadata), tuple(entries))
