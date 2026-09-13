"""Focused two-EPUB command line."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from bilingual_text_align import AlignmentAlgorithm
from bilingual_text_align.progress import ElapsedHeartbeat
from bilingual_text_align.resources import (
    default_embedding_cache,
    default_model_storage,
    default_worker_python,
)
from bilingual_text_align.vecalign_labse import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_WORKER_STARTUP_TIMEOUT,
    VecalignLabseAligner,
)

from .artifacts import write_artifact
from .pipeline import build_epub

_PHASE_MESSAGES = {
    "reading": "Reading both EPUBs",
    "containment": "Locating corresponding global text",
    "fine-alignment": "Aligning corresponding text",
    "rendering": "Rendering translation footnotes",
    "verification": "Verifying the rendered EPUB",
}


class _Progress:
    def __init__(self) -> None:
        self.started = perf_counter()
        self.phase_started = self.started
        self.current: str | None = None
        self.durations: dict[str, float] = {}

    def __call__(self, phase: str) -> None:
        now = perf_counter()
        self._finish_current(now)
        self.current = phase
        self.phase_started = now
        print(f"[{phase}] {_PHASE_MESSAGES[phase]}")

    def finish(self) -> tuple[dict[str, float], float]:
        now = perf_counter()
        self._finish_current(now)
        return (
            {name: round(value, 3) for name, value in self.durations.items()},
            now - self.started,
        )

    def _finish_current(self, now: float) -> None:
        if self.current is not None:
            self.durations[self.current] = self.durations.get(self.current, 0.0) + (
                now - self.phase_started
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilingual-footnotes",
        description=(
            "Create a new EPUB by adding linked translation notes to one of two\n"
            "corresponding editions.\n\n"
            "Choose the note source explicitly; the other EPUB becomes the reading book.\n"
            "Both source books remain unchanged."
        ),
        epilog=(
            "example — read French with English notes:\n"
            "  bilingual-footnotes french.epub english.epub --footnote-source second \\\n"
            "    --output french-with-english-notes.epub\n\n"
            "before the first run:\n"
            "  bilingual-align-setup --python python3.12 --venv .venv-bilingual\n\n"
            "Every output path must be new. Open the result in your EPUB reader and sample\n"
            "the alignment before relying on a new edition pair."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('epub-bilingual-footnotes')}"
    )
    parser.add_argument("first", metavar="FIRST_EPUB", type=Path, help="First source EPUB")
    parser.add_argument(
        "second", metavar="SECOND_EPUB", type=Path, help="Corresponding second source EPUB"
    )
    parser.add_argument(
        "--footnote-source",
        choices=("first", "second"),
        required=True,
        help=(
            "which EPUB supplies translation notes: 'second' makes FIRST_EPUB the reading "
            "book; 'first' makes SECOND_EPUB the reading book"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="EPUB",
        type=Path,
        required=True,
        help="write the completed reading book to this new EPUB file",
    )
    parser.add_argument(
        "--alignment-output",
        metavar="JSON",
        type=Path,
        help="also write alignment and verification details to this new JSON file",
    )
    runtime = parser.add_argument_group("semantic worker and storage")
    worker_python = default_worker_python()
    runtime.add_argument(
        "--aligner-python",
        default=str(worker_python),
        metavar="PYTHON",
        help=f"Python created by bilingual-align-setup (default: {worker_python})",
    )
    cache = runtime.add_mutually_exclusive_group()
    cache.add_argument(
        "--aligner-cache",
        type=Path,
        default=default_embedding_cache(),
        metavar="DIRECTORY",
        help="reuse embeddings from this directory (default: OS user cache)",
    )
    cache.add_argument(
        "--no-aligner-cache",
        action="store_const",
        const=None,
        dest="aligner_cache",
        help="do not read or write reusable book-derived embeddings",
    )
    runtime.add_argument(
        "--model-storage",
        type=Path,
        default=default_model_storage(),
        metavar="DIRECTORY",
        help="directory containing the model installed by setup (default: OS user cache)",
    )
    advanced = parser.add_argument_group("advanced alignment tuning")
    advanced.add_argument(
        "--aligner-model", default=DEFAULT_MODEL, metavar="MODEL", help="model identifier"
    )
    advanced.add_argument(
        "--aligner-model-revision",
        default=DEFAULT_MODEL_REVISION,
        metavar="REVISION",
        help="fixed model revision",
    )
    advanced.add_argument(
        "--alignment-max-size",
        type=int,
        default=8,
        metavar="UNITS",
        help="maximum units considered in one aligned group (default: 8)",
    )
    advanced.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        metavar="SIZE",
        help="embedding batch size; reduce when memory is limited (default: 16)",
    )
    advanced.add_argument(
        "--worker-startup-timeout",
        type=float,
        default=DEFAULT_WORKER_STARTUP_TIMEOUT,
        metavar="SECONDS",
        help="time allowed for worker startup and model loading (default: 600)",
    )
    advanced.add_argument(
        "--progress-interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="interval between elapsed-time updates; 0 disables them (default: 60)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    aligner: AlignmentAlgorithm | None = None,
    *,
    aligner_factory: Callable[..., AbstractContextManager[AlignmentAlgorithm]] = (
        VecalignLabseAligner
    ),
) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args, aligner, aligner_factory)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


def _run(
    args: argparse.Namespace,
    aligner: AlignmentAlgorithm | None,
    aligner_factory: Callable[..., AbstractContextManager[AlignmentAlgorithm]],
) -> None:
    inputs = {args.first.resolve(), args.second.resolve()}
    outputs = [args.output]
    if args.alignment_output is not None:
        outputs.append(args.alignment_output)
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise FileExistsError("output paths must be distinct")
    for path, resolved in zip(outputs, resolved_outputs, strict=True):
        if resolved in inputs or path.exists():
            raise FileExistsError(f"output must be a new path: {path}")
        _validate_output_directory(path)
    for path in (args.first, args.second):
        if not path.is_file():
            raise FileNotFoundError(f"input EPUB not found: {path}")
    context: AbstractContextManager[AlignmentAlgorithm]
    if aligner is None:
        context = aligner_factory(
            args.aligner_python,
            model=args.aligner_model,
            model_revision=args.aligner_model_revision,
            model_storage=args.model_storage,
            cache_dir=args.aligner_cache,
            alignment_max_size=args.alignment_max_size,
            batch_size=args.embedding_batch_size,
            startup_timeout=args.worker_startup_timeout,
        )
    else:
        context = nullcontext(aligner)
    progress = _Progress()
    with (
        ElapsedHeartbeat(lambda: progress.current or "starting", args.progress_interval),
        context as backend,
    ):
        base, notes = (
            (args.first, args.second)
            if args.footnote_source == "second"
            else (args.second, args.first)
        )
        artifact = build_epub(base, notes, args.output, backend, progress=progress)
    phase_seconds, pipeline_seconds = progress.finish()
    artifact = replace(
        artifact,
        run={
            "pipeline_seconds": round(pipeline_seconds, 3),
            "phase_seconds": phase_seconds,
            "aligner_cache": (
                str(args.aligner_cache.resolve()) if args.aligner_cache is not None else None
            ),
        },
    )
    if args.alignment_output:
        write_artifact(artifact, args.alignment_output)
    report = artifact.verification
    if report is None:
        raise RuntimeError("completed build has no verification report")
    scope = artifact.scope.classification if artifact.scope else "matching-inputs"
    model = artifact.aligner.get("model", "not-reported")
    revision = artifact.aligner.get("model_revision", "not-reported")
    cache = artifact.run["aligner_cache"] if artifact.run else None
    print("[complete] Build completed and verified")
    print(
        f"Created {args.output}: {report['base_unit_count']} base units, "
        f"{report['footnote_unit_count']} footnote units, {report['link_count']} links, "
        f"{report['rendered_note_count']} rendered notes, "
        f"{report['base_only_link_count']} base-only links, "
        f"{report['footnote_only_link_count']} footnote-only links, scope {scope}, "
        f"model {model}@{revision}, cache {cache or 'disabled'}, "
        f"verified {report['entry_count']} archive entries in {pipeline_seconds:.2f}s"
    )


def _validate_output_directory(path: Path) -> None:
    directory = path.resolve().parent
    if not directory.exists():
        raise FileNotFoundError(f"output directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"output directory is not writable: {directory}")
