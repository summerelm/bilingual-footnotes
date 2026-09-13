"""Headless application orchestration for the desktop frontend."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from bilingual_text_align import AlignmentAlgorithm
from bilingual_text_align.resources import (
    default_embedding_cache,
    default_model_storage,
    semantic_model_installed,
)
from bilingual_text_align.vecalign_labse import VecalignLabseAligner
from epub_bilingual_footnotes import Artifact, build_epub

from .configuration import bundled_worker_command, default_aligner_python

ProgressReporter = Callable[[str], None]


class EpubBuilder(Protocol):
    def __call__(
        self,
        reading: str | Path,
        translation: str | Path,
        output: str | Path,
        aligner: AlignmentAlgorithm,
        /,
        *,
        progress: ProgressReporter | None = None,
    ) -> Artifact: ...


@dataclass(frozen=True)
class EpubBuildRequest:
    reading_epub: Path
    translation_epub: Path
    output_epub: Path
    aligner_python: Path | None = None
    aligner_cache: Path | None = field(default_factory=default_embedding_cache)
    model_storage: Path = field(default_factory=default_model_storage)


@dataclass(frozen=True)
class EpubBuildResult:
    rendered_note_count: int
    link_count: int


def run_epub_build(
    request: EpubBuildRequest,
    aligner: AlignmentAlgorithm | None = None,
    report_progress: ProgressReporter | None = None,
    *,
    builder: EpubBuilder = build_epub,
    aligner_factory: Callable[..., AbstractContextManager[AlignmentAlgorithm]] = (
        VecalignLabseAligner
    ),
    writable: Callable[[Path], bool] | None = None,
    worker_command_factory: Callable[[], tuple[str, ...] | None] = bundled_worker_command,
) -> EpubBuildResult:
    """Build one verified reading EPUB with translation popup notes."""
    _validate_request(request, writable or _is_writable)
    if report_progress is not None:
        report_progress("preparing")
    context: AbstractContextManager[AlignmentAlgorithm]
    if aligner is None:
        if not semantic_model_installed(request.model_storage):
            raise RuntimeError(
                "The semantic model is not installed. Open Model & storage and install it before "
                "creating an EPUB."
            )
        options: dict[str, object] = {
            "cache_dir": request.aligner_cache,
            "model_storage": request.model_storage,
        }
        if command := worker_command_factory():
            options["worker_command"] = command
        context = aligner_factory(
            request.aligner_python or default_aligner_python(),
            **options,
        )
    else:
        context = nullcontext(aligner)
    with context as backend:
        artifact = builder(
            request.reading_epub,
            request.translation_epub,
            request.output_epub,
            backend,
            progress=report_progress,
        )
    rendered_notes = (artifact.verification or {}).get("rendered_note_count")
    if not isinstance(rendered_notes, int) or isinstance(rendered_notes, bool):
        rendered_notes = len(artifact.links)
    return EpubBuildResult(rendered_notes, len(artifact.links))


def _validate_request(request: EpubBuildRequest, writable: Callable[[Path], bool]) -> None:
    inputs = (request.reading_epub, request.translation_epub)
    for path in inputs:
        if path.suffix.lower() != ".epub":
            raise ValueError(f"input must be an EPUB: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"input EPUB not found: {path}")
    output = request.output_epub
    if output.suffix.lower() != ".epub":
        raise ValueError(f"output must be an EPUB: {output}")
    if output.resolve() in {path.resolve() for path in inputs} or output.exists():
        raise FileExistsError(f"output must be a new path: {output}")
    directory = output.resolve().parent
    if not directory.exists():
        raise FileNotFoundError(f"output directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {directory}")
    if not writable(directory):
        raise PermissionError(f"output directory is not writable: {directory}")


def _is_writable(path: Path) -> bool:
    return os.access(path, os.W_OK)
