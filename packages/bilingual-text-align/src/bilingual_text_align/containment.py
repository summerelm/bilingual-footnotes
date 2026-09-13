"""Locate one ordered semantic sequence inside another.

This module is independent of EPUB structure and embedding implementations.
Production callers can provide a :class:`ContainmentLocator`; deterministic
tests and alternative adapters can use the dependency-free functions directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Embedding = Sequence[float]
SimilarityMatrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class TextWindowConfig:
    """Structure-independent windows used as semantic containment anchors."""

    size: int = 32
    stride: int = 16
    sampled_texts: int = 8

    def __post_init__(self) -> None:
        if self.size < 1 or self.stride < 1 or self.sampled_texts < 1:
            raise ValueError("text-window sizes must be positive")
        if self.stride > self.size:
            raise ValueError("text-window stride must not exceed its size")


@dataclass(frozen=True)
class TextWindow:
    """One containment anchor and its half-open range in the original text."""

    start: int
    end: int
    text: str


def text_windows(
    texts: Sequence[str], config: TextWindowConfig | None = None
) -> tuple[TextWindow, ...]:
    """Create overlapping anchors without consulting document structure."""

    if not texts:
        raise ValueError("cannot create containment windows from an empty sequence")
    settings = config or TextWindowConfig()
    final_start = max(0, len(texts) - settings.size)
    starts = list(range(0, final_start + 1, settings.stride))
    if starts[-1] != final_start:
        starts.append(final_start)

    windows = []
    for start in starts:
        end = min(start + settings.size, len(texts))
        sample_count = min(settings.sampled_texts, end - start)
        sample_indices = (
            [start]
            if sample_count == 1
            else [
                start + round(index * (end - start - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        )
        windows.append(TextWindow(start, end, " ".join(texts[index] for index in sample_indices)))
    return tuple(windows)


@dataclass(frozen=True)
class ContainmentConfig:
    """Scoring and acceptance thresholds for semantic containment."""

    similarity_floor: float = 0.35
    query_gap_penalty: float = 0.45
    corpus_gap_penalty: float = 0.20
    minimum_coverage: float = 0.75
    minimum_mean_similarity: float = 0.50
    minimum_runner_up_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.query_gap_penalty < 0 or self.corpus_gap_penalty < 0:
            raise ValueError("gap penalties must be non-negative")
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between zero and one")
        if not -1 <= self.similarity_floor <= 1:
            raise ValueError("similarity_floor must be between -1 and 1")
        if not -1 <= self.minimum_mean_similarity <= 1:
            raise ValueError("minimum_mean_similarity must be between -1 and 1")
        if self.minimum_runner_up_margin < 0:
            raise ValueError("minimum_runner_up_margin must be non-negative")


@dataclass(frozen=True)
class ContainmentCandidate:
    """Summary of one ranked non-overlapping corpus interval."""

    corpus_start: int
    corpus_end: int
    raw_score: float
    normalized_score: float
    coverage: float
    mean_similarity: float


@dataclass(frozen=True)
class ContainmentMatch:
    """The best corpus interval and the evidence supporting the decision."""

    corpus_start: int
    corpus_end: int
    pairs: tuple[tuple[int, int], ...]
    raw_score: float
    normalized_score: float
    coverage: float
    mean_similarity: float
    runner_up_normalized_score: float | None
    runner_up_margin: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    pair_similarities: tuple[float, ...] = ()
    candidates: tuple[ContainmentCandidate, ...] = ()


@runtime_checkable
class ContainmentLocator(Protocol):
    """Locate an ordered query text inside an ordered corpus text."""

    def locate(
        self,
        query_texts: Sequence[str],
        corpus_texts: Sequence[str],
        config: ContainmentConfig | None = None,
    ) -> ContainmentMatch: ...


def conservative_text_range(
    match: ContainmentMatch,
    corpus_windows: Sequence[TextWindow],
    text_count: int,
    config: TextWindowConfig | None = None,
) -> tuple[int, int]:
    """Map a coarse window match to a safe range of original text units.

    Semantic windows are anchors, not exact sentence boundaries.  Preserve one
    window plus one placement stride on each side so a downstream fine-grained
    aligner can resolve translation-dependent segmentation drift.
    """

    if text_count < 1:
        raise ValueError("text_count must be positive")
    if not corpus_windows:
        raise ValueError("corpus windows must be non-empty")
    if not 0 <= match.corpus_start < match.corpus_end <= len(corpus_windows):
        raise ValueError("containment match has an invalid corpus interval")
    if any(not 0 <= window.start < window.end <= text_count for window in corpus_windows):
        raise ValueError("corpus window lies outside the original text")

    settings = config or TextWindowConfig()
    boundary_context = settings.size + settings.stride
    anchor_start = corpus_windows[match.corpus_start].start
    anchor_end = corpus_windows[match.corpus_end - 1].end
    return (
        max(0, anchor_start - boundary_context),
        min(text_count, anchor_end + boundary_context),
    )


def cosine_similarity_matrix(
    query_embeddings: Sequence[Embedding], corpus_embeddings: Sequence[Embedding]
) -> list[list[float]]:
    """Return cosine similarities for two non-empty embedding sequences."""

    if not query_embeddings or not corpus_embeddings:
        raise ValueError("query and corpus embeddings must be non-empty")
    dimensions = len(query_embeddings[0])
    if dimensions == 0:
        raise ValueError("embeddings must have at least one dimension")
    query = [_normalise(vector, dimensions) for vector in query_embeddings]
    corpus = [_normalise(vector, dimensions) for vector in corpus_embeddings]
    return [
        [sum(left * right for left, right in zip(a, b, strict=True)) for b in corpus] for a in query
    ]


def locate_contained_embeddings(
    query_embeddings: Sequence[Embedding],
    corpus_embeddings: Sequence[Embedding],
    config: ContainmentConfig | None = None,
) -> ContainmentMatch:
    """Locate query embeddings as an ordered subsequence of the corpus."""

    return locate_from_similarity(
        cosine_similarity_matrix(query_embeddings, corpus_embeddings), config
    )


def locate_from_similarity(
    similarities: SimilarityMatrix, config: ContainmentConfig | None = None
) -> ContainmentMatch:
    """Locate a query sequence using semi-global dynamic programming."""

    settings = config or ContainmentConfig()
    query_size, corpus_size = _matrix_shape(similarities)
    candidate = _best_candidate(similarities, settings)
    alternatives = []
    if candidate.corpus_start:
        alternatives.append(
            _best_candidate([row[: candidate.corpus_start] for row in similarities], settings)
        )
    if candidate.corpus_end < corpus_size:
        right = _best_candidate([row[candidate.corpus_end :] for row in similarities], settings)
        alternatives.append(
            _Candidate(
                right.corpus_start + candidate.corpus_end,
                right.corpus_end + candidate.corpus_end,
                tuple((query, corpus + candidate.corpus_end) for query, corpus in right.pairs),
                right.raw_score,
            )
        )

    ranked = tuple(
        sorted(
            (
                _summarise_candidate(item, similarities, query_size)
                for item in (candidate, *alternatives)
            ),
            key=lambda item: item.normalized_score,
            reverse=True,
        )
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    runner_up_score = runner_up.normalized_score if runner_up else None
    margin = best.normalized_score - runner_up_score if runner_up_score is not None else None
    reasons = []
    if best.coverage < settings.minimum_coverage:
        reasons.append(f"coverage {best.coverage:.3f} is below {settings.minimum_coverage:.3f}")
    if best.mean_similarity < settings.minimum_mean_similarity:
        reasons.append(
            "mean similarity "
            f"{best.mean_similarity:.3f} is below {settings.minimum_mean_similarity:.3f}"
        )
    if margin is not None and margin < settings.minimum_runner_up_margin:
        reasons.append(
            f"runner-up margin {margin:.3f} is below {settings.minimum_runner_up_margin:.3f}"
        )
    pair_similarities = tuple(similarities[i][j] for i, j in candidate.pairs)
    return ContainmentMatch(
        candidate.corpus_start,
        candidate.corpus_end,
        candidate.pairs,
        candidate.raw_score,
        best.normalized_score,
        best.coverage,
        best.mean_similarity,
        runner_up_score,
        margin,
        not reasons,
        tuple(reasons),
        pair_similarities,
        ranked,
    )


@dataclass(frozen=True)
class _Candidate:
    corpus_start: int
    corpus_end: int
    pairs: tuple[tuple[int, int], ...]
    raw_score: float


def _summarise_candidate(
    candidate: _Candidate, similarities: SimilarityMatrix, query_size: int
) -> ContainmentCandidate:
    values = [similarities[i][j] for i, j in candidate.pairs]
    return ContainmentCandidate(
        candidate.corpus_start,
        candidate.corpus_end,
        candidate.raw_score,
        candidate.raw_score / query_size,
        len({query for query, _corpus in candidate.pairs}) / query_size,
        sum(values) / len(values) if values else -1.0,
    )


def _best_candidate(similarities: SimilarityMatrix, settings: ContainmentConfig) -> _Candidate:
    query_size, corpus_size = _matrix_shape(similarities)
    scores = [[0.0] * (corpus_size + 1) for _ in range(query_size + 1)]
    moves = [[0] * (corpus_size + 1) for _ in range(query_size + 1)]
    for query_index in range(1, query_size + 1):
        scores[query_index][0] = scores[query_index - 1][0] - settings.query_gap_penalty
        moves[query_index][0] = 2
    for query_index in range(1, query_size + 1):
        for corpus_index in range(1, corpus_size + 1):
            choices = (
                scores[query_index - 1][corpus_index - 1]
                + similarities[query_index - 1][corpus_index - 1]
                - settings.similarity_floor,
                scores[query_index - 1][corpus_index] - settings.query_gap_penalty,
                scores[query_index][corpus_index - 1] - settings.corpus_gap_penalty,
            )
            move, score = max(enumerate(choices, 1), key=lambda item: item[1])
            scores[query_index][corpus_index] = score
            moves[query_index][corpus_index] = move
    corpus_end = max(range(corpus_size + 1), key=lambda index: scores[query_size][index])
    query_index, corpus_index = query_size, corpus_end
    pairs = []
    while query_index > 0:
        move = moves[query_index][corpus_index]
        if move == 1:
            pairs.append((query_index - 1, corpus_index - 1))
            query_index -= 1
            corpus_index -= 1
        elif move == 2:
            query_index -= 1
        elif move == 3:
            corpus_index -= 1
        else:
            raise RuntimeError("containment traceback reached an invalid state")
    pairs.reverse()
    corpus_start = pairs[0][1] if pairs else corpus_end
    return _Candidate(corpus_start, corpus_end, tuple(pairs), scores[query_size][corpus_end])


def _normalise(vector: Embedding, dimensions: int) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise ValueError("all embeddings must have the same dimensions")
    values = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("embeddings must contain only finite values")
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0:
        raise ValueError("embeddings must not be zero vectors")
    return tuple(value / magnitude for value in values)


def _matrix_shape(similarities: SimilarityMatrix) -> tuple[int, int]:
    if not similarities or not similarities[0]:
        raise ValueError("similarity matrix must be non-empty")
    corpus_size = len(similarities[0])
    for row in similarities:
        if len(row) != corpus_size:
            raise ValueError("similarity matrix must be rectangular")
        if any(not math.isfinite(float(value)) for value in row):
            raise ValueError("similarity matrix must contain only finite values")
    return len(similarities), corpus_size
