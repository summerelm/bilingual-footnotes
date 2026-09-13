"""Public structural contract implemented by alignment algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .model import AlignmentLink, TextUnit


class AlignmentAlgorithm(Protocol):
    """A statically checked, structurally implemented alignment strategy.

    Inputs are ordered, non-empty sequences with contiguous indices. An
    implementation must not mutate them. Returned links must be monotonic and
    cover both input sequences exactly once; gap links may leave one side
    empty. :func:`bilingual_text_align.service.align_units` validates these
    invariants at the application boundary.

    ``metadata`` identifies the implementation and configuration used to
    produce an artifact. Its values must be JSON serializable.

    The protocol intentionally is not ``runtime_checkable``: runtime protocol
    checks only test member presence, not these signatures or invariants.
    """

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit], /
    ) -> Sequence[AlignmentLink]: ...
