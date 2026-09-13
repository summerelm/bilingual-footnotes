"""Headless model and processing-cache use cases for the desktop frontend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bilingual_text_align import worker as semantic_worker
from bilingual_text_align.resources import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    clear_managed_resource,
    default_embedding_cache,
    default_model_storage,
    format_size,
    inspect_resource,
    prepare_model,
    prepare_model_command,
    semantic_model_installed,
)
from bilingual_text_align.vecalign_labse import DEFAULT_MODEL, DEFAULT_MODEL_REVISION

from .configuration import bundled_worker_command, default_aligner_python


class ModelPreparer(Protocol):
    def __call__(
        self,
        python: Path,
        worker: Path,
        model: str,
        revision: str,
        destination: Path,
        *,
        offline: bool = False,
    ) -> None: ...


class CommandModelPreparer(Protocol):
    def __call__(
        self,
        worker_command: Sequence[str],
        model: str,
        revision: str,
        destination: Path,
        *,
        offline: bool = False,
    ) -> None: ...


@dataclass(frozen=True)
class ResourceOverview:
    """Display-ready snapshot of the two managed resources."""

    model_status: str
    model_path: Path
    cache_status: str
    cache_path: Path


def inspect_managed_resources(
    model_storage: Path | None = None, cache_dir: Path | None = None
) -> ResourceOverview:
    """Return display-ready model and processing-cache state."""

    model_path = model_storage or default_model_storage()
    cache_path = cache_dir or default_embedding_cache()
    model = inspect_resource(model_path)
    cache = inspect_resource(cache_path)
    model_state = "Installed" if semantic_model_installed(model_path) else "Not installed"
    return ResourceOverview(
        f"{model_state} · {format_size(model.size_bytes)}",
        model.path,
        f"{format_size(cache.size_bytes)} · {cache.file_count} files",
        cache.path,
    )


def install_semantic_model(
    *,
    worker_python: Path | None = None,
    model_storage: Path | None = None,
    verify_only: bool = False,
    prepare: ModelPreparer = prepare_model,
    command_locator: Callable[[], tuple[str, ...] | None] = bundled_worker_command,
    prepare_command: CommandModelPreparer = prepare_model_command,
) -> ResourceOverview:
    """Install, repair, or verify the pinned model used by the desktop app."""

    destination = model_storage or default_model_storage()
    if command := command_locator():
        executable = Path(command[0])
        if not executable.is_file():
            raise RuntimeError(f"bundled semantic alignment worker is missing: {executable}")
        prepare_command(
            command,
            DEFAULT_MODEL,
            DEFAULT_MODEL_REVISION,
            destination,
            offline=verify_only,
        )
        return inspect_managed_resources(destination)
    worker_file = Path(semantic_worker.__file__ or "")
    if not worker_file.is_file():
        raise RuntimeError("semantic alignment worker is not installed")
    prepare(
        worker_python or default_aligner_python(),
        worker_file,
        DEFAULT_MODEL,
        DEFAULT_MODEL_REVISION,
        destination,
        offline=verify_only,
    )
    return inspect_managed_resources(destination)


def remove_semantic_model(model_storage: Path | None = None) -> None:
    clear_managed_resource(model_storage or default_model_storage(), MODEL_STORAGE_MARKER)


def clear_processing_cache(cache_dir: Path | None = None) -> None:
    clear_managed_resource(cache_dir or default_embedding_cache(), EMBEDDING_CACHE_MARKER)
