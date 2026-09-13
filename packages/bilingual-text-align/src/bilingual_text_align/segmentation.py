"""Deterministic, source-preserving plain-text segmentation."""

from __future__ import annotations

import re

from .model import TextUnit

_ABBREVIATIONS = {"dr", "etc", "m", "mme", "mlle", "mr", "mrs", "ms", "prof", "st", "ste"}
_CLOSERS = "\"'»\u2019”)]}"


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] not in ".!?…":
            index += 1
            continue
        punctuation_start = index
        while index + 1 < len(text) and text[index + 1] in ".!?…":
            index += 1
        end = index + 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        match = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ]+)$", text[start:punctuation_start])
        token = match.group(1).lower() if match else ""
        abbreviation = text[punctuation_start] == "." and (
            token in _ABBREVIATIONS or len(token) == 1
        )
        next_nonspace = end
        while next_nonspace < len(text) and text[next_nonspace].isspace():
            next_nonspace += 1
        next_char = text[next_nonspace] if next_nonspace < len(text) else ""
        plausible_start = (
            not next_char
            or next_char.isupper()
            or next_char.isdigit()
            or next_char in "\"'«“\u2018—\u2013"
        )
        if not abbreviation and plausible_start:
            left, right = _trim_span(text, start, end)
            if left < right:
                spans.append((left, right))
            start = end
        index = end
    left, right = _trim_span(text, start, len(text))
    if left < right:
        spans.append((left, right))
    return tuple(spans)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment_text(text: str, unit: str = "sentence") -> tuple[TextUnit, ...]:
    if unit not in {"sentence", "line"}:
        raise ValueError(f"Unsupported text unit: {unit}")
    units: list[TextUnit] = []
    paragraphs = re.split(r"(?:\r?\n){2,}", text)
    for group, paragraph in enumerate(paragraphs):
        if unit == "line":
            values = [line.strip() for line in paragraph.splitlines() if line.strip()]
        else:
            values = [paragraph[start:end] for start, end in sentence_spans(paragraph)]
        units.extend(TextUnit(len(units), value, group) for value in values)
    return tuple(units)
