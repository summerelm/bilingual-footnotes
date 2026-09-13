"""Manage the semantic model separately from disposable book embeddings."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from .resources import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    clear_managed_resource,
    default_embedding_cache,
    default_model_storage,
    default_worker_python,
    format_size,
    inspect_resource,
    prepare_model,
    prune_embedding_cache,
    semantic_model_installed,
)
from .vecalign_labse import DEFAULT_MODEL, DEFAULT_MODEL_REVISION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilingual-align-resources",
        description=(
            "Inspect or manage the semantic model and processing cache used by the\n"
            "bilingual alignment commands.\n\n"
            "The model is required. The cache contains disposable book-derived embeddings;\n"
            "clearing it never removes the model or source books."
        ),
        epilog=(
            "examples:\n"
            "  bilingual-align-resources status\n"
            "  bilingual-align-resources model verify\n"
            "  bilingual-align-resources cache prune --max-size-gb 5\n\n"
            "Run 'bilingual-align-resources COMMAND --help' for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('bilingual-text-align')}"
    )
    commands = parser.add_subparsers(dest="resource", required=True, title="commands")
    status = commands.add_parser(
        "status",
        help="show whether the model is installed and how much storage is used",
        description="Show model readiness and processing-cache size and locations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _storage_arguments(status)

    model = commands.add_parser(
        "model",
        help="install, verify, or remove the required semantic model",
        description="Manage the required semantic model without changing the processing cache.",
        epilog="Run 'bilingual-align-resources model ACTION --help' for action-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_commands = model.add_subparsers(dest="action", required=True, title="actions")
    for action, help_text, description in (
        (
            "install",
            "download or repair the pinned model",
            "Download or repair the pinned semantic model. Network access is required.",
        ),
        (
            "verify",
            "verify the model without network access",
            "Verify that the pinned semantic model is complete without using the network.",
        ),
        (
            "remove",
            "remove the model while preserving cached book data",
            "Remove the managed semantic model. Cached embeddings and source books are preserved.",
        ),
    ):
        command = model_commands.add_parser(action, help=help_text, description=description)
        command.add_argument(
            "--model-storage",
            type=Path,
            default=default_model_storage(),
            metavar="DIRECTORY",
            help="model directory (default: OS user cache)",
        )
        if action != "remove":
            command.add_argument(
                "--aligner-python",
                type=Path,
                default=default_worker_python(),
                metavar="PYTHON",
                help=(
                    f"Python created by bilingual-align-setup (default: {default_worker_python()})"
                ),
            )

    cache = commands.add_parser(
        "cache",
        help="clear or prune disposable book-derived embeddings",
        description="Manage disposable embeddings without changing the semantic model or books.",
        epilog="Run 'bilingual-align-resources cache ACTION --help' for action-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cache_commands = cache.add_subparsers(dest="action", required=True, title="actions")
    clear = cache_commands.add_parser(
        "clear",
        help="remove all processing-cache entries",
        description="Remove all managed book-derived embeddings. They are recreated when needed.",
    )
    clear.add_argument(
        "--cache-dir",
        type=Path,
        default=default_embedding_cache(),
        metavar="DIRECTORY",
        help="processing-cache directory (default: OS user cache)",
    )
    prune = cache_commands.add_parser(
        "prune",
        help="remove oldest entries to fit a size limit",
        description="Remove least-recently-used embeddings until the cache fits the requested size.",
    )
    prune.add_argument(
        "--cache-dir",
        type=Path,
        default=default_embedding_cache(),
        metavar="DIRECTORY",
        help="processing-cache directory (default: OS user cache)",
    )
    prune.add_argument(
        "--max-size-gb",
        type=float,
        required=True,
        metavar="GB",
        help="target maximum cache size in gigabytes",
    )
    return parser


def _storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-storage",
        type=Path,
        default=default_model_storage(),
        metavar="DIRECTORY",
        help="model directory (default: OS user cache)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_embedding_cache(),
        metavar="DIRECTORY",
        help="processing-cache directory (default: OS user cache)",
    )


def main(
    argv: list[str] | None = None,
    *,
    model_preparer: Callable[..., None] = prepare_model,
) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args, model_preparer)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


def _run(args: argparse.Namespace, model_preparer: Callable[..., None]) -> None:
    if args.resource == "status":
        _print_status(args.model_storage, args.cache_dir)
        return
    if args.resource == "model":
        if args.action == "remove":
            removed = clear_managed_resource(args.model_storage, MODEL_STORAGE_MARKER)
            print(f"Removed semantic model storage ({format_size(removed.size_bytes)})")
            return
        worker = Path(__file__).with_name("worker.py")
        model_preparer(
            args.aligner_python,
            worker,
            DEFAULT_MODEL,
            DEFAULT_MODEL_REVISION,
            args.model_storage,
            offline=args.action == "verify",
        )
        verb = "Verified" if args.action == "verify" else "Installed"
        print(f"{verb} semantic model at {args.model_storage.resolve()}")
        return
    if args.action == "clear":
        removed = clear_managed_resource(args.cache_dir, EMBEDDING_CACHE_MARKER)
        print(f"Cleared processing cache ({format_size(removed.size_bytes)})")
        return
    maximum_bytes = round(args.max_size_gb * 1024**3)
    before, after = prune_embedding_cache(args.cache_dir, maximum_bytes)
    print(
        f"Pruned processing cache from {format_size(before.size_bytes)} "
        f"to {format_size(after.size_bytes)}"
    )


def _print_status(model_path: Path, cache_path: Path) -> None:
    model = inspect_resource(model_path)
    cache = inspect_resource(cache_path)
    model_state = "installed" if semantic_model_installed(model_path) else "not installed"
    print(f"Semantic model: {model_state}, {format_size(model.size_bytes)}, {model.path.resolve()}")
    print(
        f"Processing cache: {format_size(cache.size_bytes)}, "
        f"{cache.file_count} files, {cache.path.resolve()}"
    )
