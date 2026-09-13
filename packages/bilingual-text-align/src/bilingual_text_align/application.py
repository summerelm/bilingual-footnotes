"""File-oriented application service shared by interactive frontends."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from .plain_text import FootnoteSource, TextMapping, align_texts
from .protocol import AlignmentAlgorithm


class AlignmentPhase(StrEnum):
    """Coarse, truthful phases shared by interactive frontends."""

    PREPARING = "preparing"
    READING_INPUTS = "reading-inputs"
    ALIGNING = "aligning"
    WRITING_OUTPUT = "writing-output"


ProgressReporter = Callable[[AlignmentPhase], None]


def align_files(
    first: Path,
    second: Path,
    output: Path,
    *,
    footnote_source: FootnoteSource,
    aligner: AlignmentAlgorithm,
    unit: str = "sentence",
    report_progress: ProgressReporter | None = None,
) -> TextMapping:
    """Align two UTF-8 files and atomically claim a new JSON output path.

    Frontends own presentation and aligner lifecycle. This service owns the
    shared input/output rules so command-line and graphical clients cannot
    disagree about artifact safety.
    """
    inputs = (first.resolve(), second.resolve())
    resolved_output = output.resolve()
    if resolved_output in inputs or output.exists():
        raise FileExistsError(f"output must be a new path: {output}")

    if report_progress is not None:
        report_progress(AlignmentPhase.READING_INPUTS)
    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    if report_progress is not None:
        report_progress(AlignmentPhase.ALIGNING)
    mapping = align_texts(
        first_text,
        second_text,
        footnote_source=footnote_source,
        aligner=aligner,
        unit=unit,
    )
    if report_progress is not None:
        report_progress(AlignmentPhase.WRITING_OUTPUT)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(mapping.to_json())
    return mapping
