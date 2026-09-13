from __future__ import annotations

import pytest

from bilingual_text_align.containment import (
    ContainmentConfig,
    ContainmentMatch,
    TextWindowConfig,
    conservative_text_range,
    cosine_similarity_matrix,
    locate_contained_embeddings,
    locate_from_similarity,
    text_windows,
)


def accepted_match(start: int, end: int) -> ContainmentMatch:
    return ContainmentMatch(
        start,
        end,
        (),
        1.0,
        1.0,
        1.0,
        1.0,
        None,
        None,
        True,
        (),
    )


def test_locates_exact_sequence_inside_unrelated_corpus() -> None:
    query = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    corpus = [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), *query, (-1.0, -1.0, -1.0)]

    match = locate_contained_embeddings(query, corpus)

    assert (match.corpus_start, match.corpus_end) == (2, 5)
    assert match.pairs == ((0, 2), (1, 3), (2, 4))
    assert match.coverage == 1.0
    assert match.mean_similarity == pytest.approx(1.0)
    assert match.accepted


def test_allows_extra_corpus_content_inside_interval() -> None:
    match = locate_from_similarity(
        [
            [0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.90, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.92, 0.0],
        ]
    )

    assert (match.corpus_start, match.corpus_end) == (1, 5)
    assert match.accepted


def test_rejects_weak_and_ambiguous_matches() -> None:
    weak = locate_from_similarity(
        [[0.20, 0.25], [0.15, 0.20]],
        ContainmentConfig(minimum_coverage=0.5, minimum_mean_similarity=0.6),
    )
    assert not weak.accepted
    assert any("mean similarity" in reason for reason in weak.rejection_reasons)

    repeated = locate_contained_embeddings(
        [(1.0, 0.0), (0.0, 1.0)],
        [(1.0, 0.0), (0.0, 1.0), (-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)],
    )
    assert not repeated.accepted
    assert repeated.runner_up_margin == pytest.approx(0.0)
    assert any("runner-up margin" in reason for reason in repeated.rejection_reasons)


def test_windows_use_only_sequence_positions() -> None:
    windows = text_windows(
        ["A1", "A2", "A3", "A4", "A5"],
        TextWindowConfig(size=3, stride=2, sampled_texts=3),
    )
    assert [(item.start, item.end, item.text) for item in windows] == [
        (0, 3, "A1 A2 A3"),
        (2, 5, "A3 A4 A5"),
    ]
    assert (
        text_windows(["only"], TextWindowConfig(size=1, stride=1, sampled_texts=1))[0].text
        == "only"
    )
    assert [
        (item.start, item.end)
        for item in text_windows(["1", "2", "3", "4", "5", "6"], TextWindowConfig(size=3, stride=2))
    ] == [(0, 3), (2, 5), (3, 6)]


def test_window_match_maps_to_conservative_text_range() -> None:
    texts = [str(index) for index in range(1544)]
    config = TextWindowConfig()
    windows = text_windows(texts, config)

    assert (windows[4].start, windows[94].end) == (64, 1536)
    assert conservative_text_range(accepted_match(4, 95), windows, len(texts), config) == (
        16,
        1544,
    )


def test_conservative_text_range_validates_original_coordinates() -> None:
    windows = text_windows(["one", "two"], TextWindowConfig(size=1, stride=1))

    with pytest.raises(ValueError, match="text_count must be positive"):
        conservative_text_range(accepted_match(0, 1), windows, 0)
    with pytest.raises(ValueError, match="non-empty"):
        conservative_text_range(accepted_match(0, 1), (), 2)
    with pytest.raises(ValueError, match="invalid corpus interval"):
        conservative_text_range(accepted_match(0, 3), windows, 2)
    with pytest.raises(ValueError, match="outside the original text"):
        conservative_text_range(accepted_match(0, 1), windows, 1)


@pytest.mark.parametrize(
    "config",
    [
        TextWindowConfig(size=1, stride=1),
    ],
)
def test_window_config_accepts_valid_values(config: TextWindowConfig) -> None:
    assert config.size == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"size": 0},
        {"size": 1, "stride": 2},
    ],
)
def test_window_config_rejects_invalid_values(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TextWindowConfig(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query_gap_penalty": -1.0},
        {"minimum_coverage": 2.0},
        {"similarity_floor": 2.0},
        {"minimum_mean_similarity": -2.0},
        {"minimum_runner_up_margin": -1.0},
    ],
)
def test_containment_config_rejects_invalid_values(arguments: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ContainmentConfig(**arguments)


def test_validates_embedding_and_matrix_shapes() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        text_windows([])
    with pytest.raises(ValueError, match="must be non-empty"):
        cosine_similarity_matrix([], [(1.0,)])
    with pytest.raises(ValueError, match="one dimension"):
        cosine_similarity_matrix([()], [()])
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity_matrix([(1.0, 0.0)], [(1.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity_matrix([(float("nan"),)], [(1.0,)])
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarity_matrix([(0.0,)], [(1.0,)])
    with pytest.raises(ValueError, match="non-empty"):
        locate_from_similarity([])
    with pytest.raises(ValueError, match="rectangular"):
        locate_from_similarity([[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="finite"):
        locate_from_similarity([[float("inf")]])


def test_rejects_insufficient_query_coverage() -> None:
    match = locate_from_similarity(
        [[0.9, 0.0], [-1.0, -1.0]],
        ContainmentConfig(
            query_gap_penalty=0.0,
            minimum_coverage=1.0,
            minimum_mean_similarity=-1.0,
        ),
    )
    assert not match.accepted
    assert any("coverage" in reason for reason in match.rejection_reasons)
