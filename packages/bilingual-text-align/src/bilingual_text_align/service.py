"""Validated orchestration shared by all frontends."""

from __future__ import annotations

from collections.abc import Sequence

from .model import AlignmentLink, TextUnit
from .protocol import AlignmentAlgorithm


def align_units(
    first: Sequence[TextUnit], second: Sequence[TextUnit], algorithm: AlignmentAlgorithm
) -> tuple[AlignmentLink, ...]:
    _validate_units(first, "first")
    _validate_units(second, "second")
    links = tuple(algorithm.align(first, second))
    _validate_links(links, len(first), len(second))
    return links


def _validate_units(units: Sequence[TextUnit], label: str) -> None:
    if not units:
        raise ValueError(f"{label} text contains no units")
    if [item.index for item in units] != list(range(len(units))):
        raise ValueError(f"{label} unit indices must be contiguous and ordered")
    if any(not item.text.strip() for item in units):
        raise ValueError(f"{label} units must not be blank")


def _validate_links(links: Sequence[AlignmentLink], first_size: int, second_size: int) -> None:
    if not links:
        raise ValueError("aligner returned no links")
    first = [index for link in links for index in link.first_indices]
    second = [index for link in links for index in link.second_indices]
    if first != list(range(first_size)):
        raise ValueError("aligner links do not cover the first text exactly once in order")
    if second != list(range(second_size)):
        raise ValueError("aligner links do not cover the second text exactly once in order")
    if any(not link.first_indices and not link.second_indices for link in links):
        raise ValueError("alignment links cannot be empty on both sides")
