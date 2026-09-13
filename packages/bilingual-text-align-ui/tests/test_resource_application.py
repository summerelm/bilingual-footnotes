from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from bilingual_text_align.resources import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    mark_resource,
)

from bilingual_text_align_ui import (
    clear_processing_cache,
    inspect_managed_resources,
    install_semantic_model,
    remove_semantic_model,
)


def test_desktop_resource_lifecycle_is_explicit_and_separate(tmp_path: Path) -> None:
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    worker = tmp_path / "python"
    calls: list[tuple[Path, Path, bool]] = []

    def prepare(
        python: Path,
        worker: Path,
        model: str,
        revision: str,
        destination: Path,
        *,
        offline: bool = False,
    ) -> None:
        del model, revision
        assert worker.is_file()
        calls.append((python, destination, offline))
        mark_resource(destination, MODEL_STORAGE_MARKER)
        (destination / "weights").write_bytes(b"model")

    install_semantic_model(worker_python=worker, model_storage=model, prepare=prepare)
    install_semantic_model(
        worker_python=worker, model_storage=model, verify_only=True, prepare=prepare
    )
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    (cache / "book.npy").write_bytes(b"cache")

    overview = inspect_managed_resources(model, cache)

    assert overview.model_status == "Installed · 5 B"
    assert overview.model_path == model
    assert overview.cache_status == "5 B · 2 files"
    assert overview.cache_path == cache
    assert calls == [(worker, model, False), (worker, model, True)]
    clear_processing_cache(cache)
    assert model.exists()
    remove_semantic_model(model)
    assert not model.exists()


def test_frozen_desktop_uses_embedded_worker_to_manage_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    executable = tmp_path / "bilingual-align-worker"
    executable.write_bytes(b"worker")
    calls: list[tuple[tuple[str, ...], Path, bool]] = []

    def prepare_command(
        worker_command: Sequence[str],
        model: str,
        revision: str,
        destination: Path,
        *,
        offline: bool = False,
    ) -> None:
        del model, revision
        calls.append((tuple(worker_command), destination, offline))
        mark_resource(destination, MODEL_STORAGE_MARKER)
        (destination / "weights").write_bytes(b"model")

    overview = install_semantic_model(
        model_storage=model,
        verify_only=True,
        command_locator=lambda: (str(executable),),
        prepare_command=prepare_command,
    )

    assert overview.model_status == "Installed · 5 B"
    assert calls == [((str(executable),), model, True)]


def test_frozen_desktop_reports_missing_embedded_worker(tmp_path: Path) -> None:
    missing = tmp_path / "missing-worker"
    with pytest.raises(RuntimeError, match="bundled semantic alignment worker is missing"):
        install_semantic_model(command_locator=lambda: (str(missing),))
