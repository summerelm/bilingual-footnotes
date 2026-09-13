"""Compose containment, fine alignment, and EPUB rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from bilingual_text_align import (
    AlignmentAlgorithm,
    AlignmentLink,
    BoundaryReconciliationConfig,
    ContainmentConfig,
    ContainmentLocator,
    ContainmentMatch,
    TextSimilarityScorer,
    TextUnit,
    TextWindowConfig,
    conservative_text_range,
    reconcile_boundary_context,
    text_windows,
)
from bilingual_text_align.service import align_units

from .model import Artifact, Book, ScopeSelection
from .protocol import BookReader, EpubRenderer, ProgressReporter


@dataclass(frozen=True)
class EpubFootnoteService:
    """Coordinate reading, alignment, and rendering through structural ports."""

    reader: BookReader
    renderer: EpubRenderer
    window_config: TextWindowConfig = TextWindowConfig()
    containment_config: ContainmentConfig = ContainmentConfig()
    boundary_config: BoundaryReconciliationConfig = BoundaryReconciliationConfig()

    def align(
        self,
        base_epub: str | Path,
        footnote_epub: str | Path,
        algorithm: AlignmentAlgorithm,
        /,
        *,
        progress: ProgressReporter | None = None,
    ) -> Artifact:
        _progress(progress, "reading")
        base = self.reader(base_epub)
        footnotes = self.reader(footnote_epub)
        if isinstance(algorithm, ContainmentLocator):
            _progress(progress, "containment")
            scope, links = self._align_scoped(base, footnotes, algorithm, algorithm, progress)
        else:
            scope = None
            _progress(progress, "fine-alignment")
            links = align_units(base.units, footnotes.units, algorithm)
        return Artifact(
            schema_version=3,
            base_sha256=base.sha256,
            footnote_sha256=footnotes.sha256,
            aligner=dict(algorithm.metadata),
            base_units=base.units,
            base_anchors=base.anchors,
            footnote_units=footnotes.units,
            links=links,
            scope=scope,
        )

    def _align_scoped(
        self,
        base: Book,
        footnotes: Book,
        algorithm: AlignmentAlgorithm,
        locator: ContainmentLocator,
        progress: ProgressReporter | None,
    ) -> tuple[ScopeSelection, tuple[AlignmentLink, ...]]:
        base_windows = text_windows([unit.text for unit in base.units], self.window_config)
        footnote_windows = text_windows([unit.text for unit in footnotes.units], self.window_config)
        context_side: Literal["first", "second"]
        if len(base.units) <= len(footnotes.units):
            match = locator.locate(
                [window.text for window in base_windows],
                [window.text for window in footnote_windows],
                self.containment_config,
            )
            _require_accepted(match)
            base_range = (0, len(base.units))
            context_side = "second"
            footnote_range = conservative_text_range(
                match,
                footnote_windows,
                len(footnotes.units),
                self.window_config,
            )
        else:
            match = locator.locate(
                [window.text for window in footnote_windows],
                [window.text for window in base_windows],
                self.containment_config,
            )
            _require_accepted(match)
            base_range = conservative_text_range(
                match,
                base_windows,
                len(base.units),
                self.window_config,
            )
            footnote_range = (0, len(footnotes.units))
            context_side = "first"

        base_start, base_end = base_range
        footnote_start, footnote_end = footnote_range
        classification = _classification(base_range, footnote_range, base, footnotes)
        evidence: dict[str, object] = {
            "unit": "fixed-text-window",
            "window_config": asdict(self.window_config),
            "boundary_context": self.window_config.size + self.window_config.stride,
            "match": asdict(match),
        }
        _progress(progress, "fine-alignment")
        local_base = _local_units(base.units[base_start:base_end])
        local_footnotes = _local_units(footnotes.units[footnote_start:footnote_end])
        local_links = align_units(
            local_base,
            local_footnotes,
            algorithm,
        )
        if isinstance(algorithm, TextSimilarityScorer):
            reconciliation = reconcile_boundary_context(
                local_links,
                local_base,
                local_footnotes,
                context_side,
                algorithm,
                self.boundary_config,
            )
            local_links = reconciliation.links
            evidence["boundary_reconciliation"] = {
                "config": asdict(self.boundary_config),
                "selections": [asdict(selection) for selection in reconciliation.selections],
            }
        scope = ScopeSelection(
            classification,
            base_start,
            base_end,
            footnote_start,
            footnote_end,
            evidence,
        )
        links = []
        if base_start:
            links.append(AlignmentLink(tuple(range(base_start)), (), None))
        if footnote_start:
            links.append(AlignmentLink((), tuple(range(footnote_start)), None))
        links.extend(
            AlignmentLink(
                tuple(base_start + index for index in link.first_indices),
                tuple(footnote_start + index for index in link.second_indices),
                link.score,
            )
            for link in local_links
        )
        if base_end < len(base.units):
            links.append(AlignmentLink(tuple(range(base_end, len(base.units))), (), None))
        if footnote_end < len(footnotes.units):
            links.append(AlignmentLink((), tuple(range(footnote_end, len(footnotes.units))), None))
        return scope, tuple(links)

    def build(
        self,
        base_epub: str | Path,
        footnote_epub: str | Path,
        output: str | Path,
        algorithm: AlignmentAlgorithm,
        /,
        *,
        progress: ProgressReporter | None = None,
    ) -> Artifact:
        artifact = self.align(base_epub, footnote_epub, algorithm, progress=progress)
        _progress(progress, "rendering")
        self.renderer(base_epub, artifact, output)
        return artifact


def _progress(reporter: ProgressReporter | None, phase: str) -> None:
    if reporter is not None:
        reporter(phase)


def _require_accepted(match: ContainmentMatch) -> None:
    if not match.accepted:
        reasons = "; ".join(match.rejection_reasons)
        raise ValueError(
            f"text containment is ambiguous or too weak: {reasons}. "
            "Check that the two EPUBs are corresponding, broadly complete editions"
        )


def _local_units(units: tuple[TextUnit, ...]) -> tuple[TextUnit, ...]:
    return tuple(replace(unit, index=index) for index, unit in enumerate(units))


def _classification(
    base_range: tuple[int, int],
    footnote_range: tuple[int, int],
    base: Book,
    footnotes: Book,
) -> str:
    if base_range == (0, len(base.units)) and footnote_range == (
        0,
        len(footnotes.units),
    ):
        return "complete-editions"
    if base_range == (0, len(base.units)):
        return "base-contained-in-footnotes"
    return "footnotes-contained-in-base"
