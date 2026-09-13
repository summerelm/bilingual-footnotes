from __future__ import annotations

from collections.abc import Sequence

import pytest

from bilingual_text_align.boundary_reconciliation import (
    BoundaryReconciliationConfig,
    TextSimilarityScorer,
    reconcile_boundary_context,
)
from bilingual_text_align.model import AlignmentLink, TextUnit


class ScriptedScorer:
    def __init__(self, *rows: Sequence[float]) -> None:
        self.rows = list(rows)
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def similarities(
        self, query_texts: Sequence[str], corpus_texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        self.calls.append((tuple(query_texts), tuple(corpus_texts)))
        return [self.rows.pop(0)]


def units(*items: tuple[str, int]) -> tuple[TextUnit, ...]:
    return tuple(TextUnit(index, text, group) for index, (text, group) in enumerate(items))


def test_trims_unmatched_groups_from_both_containment_edges() -> None:
    first = units(
        ("preface", 0),
        ("opening", 1),
        ("middle", 2),
        ("ending", 3),
        ("FIN", 4),
        ("printer", 5),
    )
    second = units(("the opening", 0), ("the middle", 1), ("the ending", 2))
    links = (
        AlignmentLink((0, 1), (0,), 0.7),
        AlignmentLink((2,), (1,), 0.9),
        AlignmentLink((3, 4, 5), (2,), 0.7),
    )
    scorer = ScriptedScorer([0.91, 0.89], [0.88, 0.90, 0.71])

    result = reconcile_boundary_context(links, first, second, "first", scorer)

    assert result.links == (
        AlignmentLink((0,), (), None),
        AlignmentLink((1,), (0,), 0.7),
        AlignmentLink((2,), (1,), 0.9),
        AlignmentLink((3,), (2,), 0.7),
        AlignmentLink((4, 5), (), None),
    )
    assert [selection.selected_indices for selection in result.selections] == [(1,), (3,)]
    assert all(selection.accepted for selection in result.selections)
    assert scorer.calls == [
        (("the opening",), ("opening", "preface opening")),
        (("the ending",), ("ending", "ending FIN", "ending FIN printer")),
    ]


def test_reconciles_context_on_second_side() -> None:
    first = units(("opening", 0), ("middle", 1), ("ending", 2))
    second = units(
        ("preface", 0), ("the opening", 1), ("the middle", 2), ("the ending", 3), ("end", 4)
    )
    links = (
        AlignmentLink((0,), (0, 1), 0.8),
        AlignmentLink((1,), (2,), 0.9),
        AlignmentLink((2,), (3, 4), 0.8),
    )

    result = reconcile_boundary_context(
        links,
        first,
        second,
        "second",
        ScriptedScorer([0.9, 0.7], [0.9, 0.8]),
    )

    assert result.links == (
        AlignmentLink((), (0,), None),
        AlignmentLink((0,), (1,), 0.8),
        AlignmentLink((1,), (2,), 0.9),
        AlignmentLink((2,), (3,), 0.8),
        AlignmentLink((), (4,), None),
    )


def test_preserves_boundary_when_similarity_is_weak() -> None:
    first = units(("context", 0), ("possible match", 1), ("middle", 2), ("end", 3))
    second = units(("opening", 0), ("middle", 1), ("ending", 2))
    links = (
        AlignmentLink((0, 1), (0,), 0.4),
        AlignmentLink((2,), (1,), 0.9),
        AlignmentLink((3,), (2,), 0.9),
    )

    result = reconcile_boundary_context(
        links,
        first,
        second,
        "first",
        ScriptedScorer([0.4, 0.3]),
    )

    assert result.links == links
    assert not result.selections[0].accepted


def test_skips_single_match_or_single_group_boundaries() -> None:
    first = units(("one", 0), ("two", 0), ("three", 0))
    second = units(("un", 0), ("deux", 0), ("trois", 0))
    scorer = ScriptedScorer()

    one_link = reconcile_boundary_context(
        (AlignmentLink((0, 1, 2), (0, 1, 2), 0.8),), first, second, "first", scorer
    )
    separate = reconcile_boundary_context(
        (
            AlignmentLink((0,), (0,), 0.8),
            AlignmentLink((1,), (1,), 0.8),
            AlignmentLink((2,), (2,), 0.8),
        ),
        first,
        second,
        "first",
        scorer,
    )

    assert not one_link.selections
    assert not separate.selections
    assert not scorer.calls


@pytest.mark.parametrize(
    "config",
    [
        BoundaryReconciliationConfig(),
        BoundaryReconciliationConfig(minimum_similarity=-1, similarity_tolerance=2),
    ],
)
def test_config_accepts_valid_thresholds(config: BoundaryReconciliationConfig) -> None:
    assert config.similarity_tolerance >= 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"minimum_similarity": 2.0},
        {"similarity_tolerance": -0.1},
        {"similarity_tolerance": 2.1},
    ],
)
def test_config_rejects_invalid_thresholds(arguments: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BoundaryReconciliationConfig(**arguments)


def test_validates_side_indices_and_similarity_matrix() -> None:
    first = units(("one", 0), ("two", 1), ("three", 2))
    second = units(("un", 0), ("deux", 1), ("trois", 2))
    links = (
        AlignmentLink((0, 1), (0,), 0.8),
        AlignmentLink((2,), (1, 2), 0.8),
    )

    with pytest.raises(ValueError, match="context_side"):
        reconcile_boundary_context(links, first, second, "bad", ScriptedScorer())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="invalid boundary matrix"):
        reconcile_boundary_context(links, first, second, "first", ScriptedScorer([]))
    with pytest.raises(RuntimeError, match="finite values"):
        reconcile_boundary_context(
            links, first, second, "first", ScriptedScorer([float("nan"), 0.5])
        )


def test_similarity_scorer_protocol_is_structural() -> None:
    assert isinstance(ScriptedScorer([1.0]), TextSimilarityScorer)
