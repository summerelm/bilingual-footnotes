"""Command-line frontend for plain-text alignment."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from importlib.metadata import version
from pathlib import Path

from .application import align_files
from .plain_text import FootnoteSource
from .progress import ElapsedHeartbeat
from .protocol import AlignmentAlgorithm
from .resources import default_embedding_cache, default_model_storage, default_worker_python
from .vecalign_labse import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_WORKER_STARTUP_TIMEOUT,
    VecalignLabseAligner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilingual-align",
        description=(
            "Align two corresponding UTF-8 text files and write an inspectable JSON mapping.\n\n"
            "The base is the text being read. The footnote source is its translation.\n"
            "Inputs may contain unmatched passages, but their shared content must stay in order."
        ),
        epilog=(
            "example:\n"
            "  bilingual-align french.txt english.txt --footnote-source second \\\n"
            "    --output french-english.json\n\n"
            "before the first run:\n"
            "  bilingual-align-setup --python python3.12 --venv .venv-bilingual\n\n"
            "The output path must not already exist. Source files are never changed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('bilingual-text-align')}"
    )
    parser.add_argument("first", metavar="FIRST_TEXT", type=Path, help="First UTF-8 text file")
    parser.add_argument(
        "second", metavar="SECOND_TEXT", type=Path, help="Corresponding second UTF-8 text file"
    )
    parser.add_argument(
        "--footnote-source",
        required=True,
        choices=("first", "second"),
        help=(
            "which input supplies translation text: 'second' makes FIRST_TEXT the base; "
            "'first' makes SECOND_TEXT the base"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="JSON",
        type=Path,
        required=True,
        help="write the alignment mapping to this new JSON file",
    )
    parser.add_argument(
        "--unit",
        choices=("sentence", "line"),
        default="sentence",
        help=(
            "how to split each text; 'line' expects one unit per non-empty line (default: sentence)"
        ),
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


def main(argv: list[str] | None = None, aligner: AlignmentAlgorithm | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args, aligner)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


def _run(args: argparse.Namespace, aligner: AlignmentAlgorithm | None) -> None:
    context: AbstractContextManager[AlignmentAlgorithm]
    if aligner is None:
        context = VecalignLabseAligner(
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
    print("[alignment] Aligning corresponding text")
    with ElapsedHeartbeat(lambda: "alignment", args.progress_interval), context as backend:
        mapping = align_files(
            args.first,
            args.second,
            args.output,
            footnote_source=FootnoteSource(args.footnote_source),
            aligner=backend,
            unit=args.unit,
        )
    print(f"Aligned {len(mapping.mappings)} links; created {args.output}")
