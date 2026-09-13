"""Separate unmatched containment context from boundary alignment links."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .model import AlignmentLink, TextUnit

BoundaryEdge = Literal["leading", "trailing"]
AlignmentSide = Literal["first", "second"]


@runtime_checkable
class TextSimilarityScorer(Protocol):
    """Return pairwise semantic similarities for two ordered text lists."""

    def similarities(
        self, query_texts: Sequence[str], corpus_texts: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class BoundaryReconciliationConfig:
    """Acceptance thresholds for trimming fine-alignment boundary links."""

    minimum_similarity: float = 0.50
    similarity_tolerance: float = 0.03

    def __post_init__(self) -> None:
        if not -1 <= self.minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be between -1 and 1")
        if not 0 <= self.similarity_tolerance <= 2:
            raise ValueError("similarity_tolerance must be between zero and two")


@dataclass(frozen=True)
class BoundarySelection:
    """Evidence for one attempted boundary-context selection."""

    edge: BoundaryEdge
    side: AlignmentSide
    original_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    candidate_similarities: tuple[float, ...]
    accepted: bool


@dataclass(frozen=True)
class BoundaryReconciliation:
    """Reconciled links and auditable edge-selection evidence."""

    links: tuple[AlignmentLink, ...]
    selections: tuple[BoundarySelection, ...]


def reconcile_boundary_context(
    links: Sequence[AlignmentLink],
    first: Sequence[TextUnit],
    second: Sequence[TextUnit],
    context_side: AlignmentSide,
    scorer: TextSimilarityScorer,
    config: BoundaryReconciliationConfig | None = None,
) -> BoundaryReconciliation:
    """Split unmatched paragraph groups from the first and last matched links.

    A conservative containment range deliberately includes surrounding text.
    Fine aligners may absorb that context into a many-to-one edge link. This
    pass examines only distinct leading and trailing matched links and keeps
    all interior alignment decisions unchanged.
    """

    if context_side not in {"first", "second"}:
        raise ValueError("context_side must be 'first' or 'second'")
    result = list(links)
    aligned = [
        index for index, link in enumerate(result) if link.first_indices and link.second_indices
    ]
    if len(aligned) < 2:
        return BoundaryReconciliation(tuple(result), ())

    selections = []
    offset = 0
    edges: tuple[tuple[BoundaryEdge, int], ...] = (
        ("leading", aligned[0]),
        ("trailing", aligned[-1]),
    )
    for edge, original_position in edges:
        position = original_position + offset
        replacement, selection = _reconcile_link(
            result[position], first, second, context_side, edge, scorer, config
        )
        if selection is None:
            continue
        selections.append(selection)
        result[position : position + 1] = replacement
        offset += len(replacement) - 1
    return BoundaryReconciliation(tuple(result), tuple(selections))


def _reconcile_link(
    link: AlignmentLink,
    first: Sequence[TextUnit],
    second: Sequence[TextUnit],
    side: AlignmentSide,
    edge: BoundaryEdge,
    scorer: TextSimilarityScorer,
    config: BoundaryReconciliationConfig | None,
) -> tuple[tuple[AlignmentLink, ...], BoundarySelection | None]:
    context_units = first if side == "first" else second
    counterpart_units = second if side == "first" else first
    context_indices = link.first_indices if side == "first" else link.second_indices
    counterpart_indices = link.second_indices if side == "first" else link.first_indices
    candidates = _edge_candidates(context_units, context_indices, edge)
    if len(candidates) < 2:
        return (link,), None

    counterpart_text = " ".join(counterpart_units[index].text for index in counterpart_indices)
    similarities = _similarities(
        scorer,
        counterpart_text,
        [" ".join(context_units[index].text for index in candidate) for candidate in candidates],
    )
    settings = config or BoundaryReconciliationConfig()
    best = max(similarities)
    selected_position = next(
        index
        for index, similarity in enumerate(similarities)
        if similarity >= best - settings.similarity_tolerance
    )
    selected = candidates[selected_position]
    accepted = similarities[selected_position] >= settings.minimum_similarity
    selection = BoundarySelection(
        edge,
        side,
        tuple(context_indices),
        selected if accepted else tuple(context_indices),
        similarities,
        accepted,
    )
    if not accepted or selected == tuple(context_indices):
        return (link,), selection

    discarded = (
        tuple(index for index in context_indices if index < selected[0])
        if edge == "leading"
        else tuple(index for index in context_indices if index > selected[-1])
    )
    matched = (
        AlignmentLink(selected, link.second_indices, link.score)
        if side == "first"
        else AlignmentLink(link.first_indices, selected, link.score)
    )
    gap = (
        AlignmentLink(discarded, (), None)
        if side == "first"
        else AlignmentLink((), discarded, None)
    )
    return ((gap, matched) if edge == "leading" else (matched, gap)), selection


def _edge_candidates(
    units: Sequence[TextUnit], indices: Sequence[int], edge: BoundaryEdge
) -> tuple[tuple[int, ...], ...]:
    if not indices or any(index < 0 or index >= len(units) for index in indices):
        raise ValueError("boundary indices must refer to existing text units")
    if list(indices) != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("boundary indices must be contiguous and ordered")
    groups: list[list[int]] = []
    for index in indices:
        if not groups or units[groups[-1][-1]].group != units[index].group:
            groups.append([])
        groups[-1].append(index)
    candidates = []
    for size in range(1, len(groups) + 1):
        selected_groups = groups[-size:] if edge == "leading" else groups[:size]
        candidates.append(tuple(index for group in selected_groups for index in group))
    return tuple(candidates)


def _similarities(
    scorer: TextSimilarityScorer, query: str, candidates: Sequence[str]
) -> tuple[float, ...]:
    rows = scorer.similarities([query], candidates)
    if len(rows) != 1 or len(rows[0]) != len(candidates):
        raise RuntimeError("similarity scorer returned an invalid boundary matrix")
    values = tuple(float(value) for value in rows[0])
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
        raise RuntimeError("boundary similarities must be finite values between -1 and 1")
    return values
