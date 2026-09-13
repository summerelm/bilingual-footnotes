"""Default adapter composition for whole-book EPUB processing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from bilingual_text_align import AlignmentAlgorithm

from .extraction import read_book
from .model import Artifact
from .protocol import ProgressReporter
from .rendering import render_epub
from .service import EpubFootnoteService
from .verification import VerificationReport, verify_epub

_DEFAULT_SERVICE = EpubFootnoteService(read_book, render_epub)


class EpubVerifier(Protocol):
    def __call__(
        self, base: str | Path, notes: str | Path, output: str | Path, artifact: Artifact, /
    ) -> VerificationReport: ...


def align_epubs(
    base_epub: str | Path,
    footnote_epub: str | Path,
    algorithm: AlignmentAlgorithm,
    *,
    progress: ProgressReporter | None = None,
) -> Artifact:
    return _DEFAULT_SERVICE.align(base_epub, footnote_epub, algorithm, progress=progress)


def build_epub(
    base_epub: str | Path,
    footnote_epub: str | Path,
    output: str | Path,
    algorithm: AlignmentAlgorithm,
    *,
    progress: ProgressReporter | None = None,
    verifier: EpubVerifier = verify_epub,
) -> Artifact:
    artifact = _DEFAULT_SERVICE.align(base_epub, footnote_epub, algorithm, progress=progress)
    if progress is not None:
        progress("rendering")
    render_epub(base_epub, artifact, output)
    if progress is not None:
        progress("verification")
    try:
        report = verifier(base_epub, footnote_epub, output, artifact)
    except Exception:
        Path(output).unlink(missing_ok=True)
        raise
    return replace(artifact, verification=report.to_dict())
