"""Public API for bilingual text alignment."""

from .application import AlignmentPhase, ProgressReporter, align_files
from .boundary_reconciliation import (
    BoundaryReconciliation,
    BoundaryReconciliationConfig,
    BoundarySelection,
    TextSimilarityScorer,
    reconcile_boundary_context,
)
from .containment import (
    ContainmentConfig,
    ContainmentLocator,
    ContainmentMatch,
    TextWindow,
    TextWindowConfig,
    conservative_text_range,
    locate_contained_embeddings,
    locate_from_similarity,
    text_windows,
)
from .model import AlignmentLink, TextUnit
from .plain_text import FootnoteSource, TextMapping, align_texts
from .protocol import AlignmentAlgorithm

__all__ = [
    "AlignmentAlgorithm",
    "AlignmentLink",
    "AlignmentPhase",
    "BoundaryReconciliation",
    "BoundaryReconciliationConfig",
    "BoundarySelection",
    "ContainmentConfig",
    "ContainmentLocator",
    "ContainmentMatch",
    "FootnoteSource",
    "TextMapping",
    "TextSimilarityScorer",
    "TextUnit",
    "TextWindow",
    "TextWindowConfig",
    "ProgressReporter",
    "align_texts",
    "align_files",
    "conservative_text_range",
    "locate_contained_embeddings",
    "locate_from_similarity",
    "reconcile_boundary_context",
    "text_windows",
]
