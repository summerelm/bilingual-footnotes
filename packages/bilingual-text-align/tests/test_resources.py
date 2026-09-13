from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bilingual_text_align.resources import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    clear_managed_resource,
    default_embedding_cache,
    default_model_storage,
    default_storage_root,
    format_size,
    inspect_resource,
    mark_resource,
    prepare_model,
    prune_embedding_cache,
    resource_activity_marker,
    semantic_model_installed,
)


def test_default_storage_uses_platform_conventions_and_override(tmp_path: Path) -> None:
    assert default_storage_root(platform="darwin", environment={}, home=tmp_path) == (
        tmp_path / "Library" / "Caches" / "Bilingual Footnotes"
    )
    assert default_storage_root(platform="linux", environment={}, home=tmp_path) == (
        tmp_path / ".cache" / "bilingual-footnotes"
    )
    assert default_storage_root(
        platform="linux", environment={"XDG_CACHE_HOME": "/cache"}, home=tmp_path
    ) == Path("/cache/bilingual-footnotes")
    assert default_storage_root(
        platform="win32", environment={"LOCALAPPDATA": "C:/Local"}, home=tmp_path
    ) == Path("C:/Local/Bilingual Footnotes")
    assert (
        default_storage_root(
            platform="darwin",
            environment={"BILINGUAL_FOOTNOTES_STORAGE": str(tmp_path / "chosen")},
            home=Path("/ignored"),
        )
        == tmp_path / "chosen"
    )
    assert default_model_storage(platform="darwin", environment={}, home=tmp_path).name == "model"
    assert default_embedding_cache(platform="darwin", environment={}, home=tmp_path).name == (
        "embeddings"
    )


def test_resource_status_clear_and_formatting(tmp_path: Path) -> None:
    resource = tmp_path / "resource"
    (resource / "nested").mkdir(parents=True)
    mark_resource(resource, EMBEDDING_CACHE_MARKER)
    (resource / "one").write_bytes(b"123")
    (resource / "nested" / "two").write_bytes(b"4567")

    status = inspect_resource(resource)

    assert status.present
    assert status.file_count == 3
    assert status.size_bytes == 7
    assert clear_managed_resource(resource, EMBEDDING_CACHE_MARKER) == status
    assert not resource.exists()
    assert inspect_resource(resource).size_bytes == 0
    assert format_size(0) == "0 B"
    assert format_size(1536) == "1.5 KB"


def test_resource_size_counts_linked_content_once(tmp_path: Path) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    mark_resource(resource, EMBEDDING_CACHE_MARKER)
    payload = resource / "payload"
    payload.write_bytes(b"1234")
    (resource / "symbolic-link").symlink_to(payload)
    os.link(payload, resource / "hard-link")

    status = inspect_resource(resource)

    assert status.file_count == 4
    assert status.size_bytes == 4


def test_semantic_model_requires_marker_and_model_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    assert not semantic_model_installed(model)
    mark_resource(model, MODEL_STORAGE_MARKER)
    assert not semantic_model_installed(model)
    (model / "weights").write_bytes(b"model")
    assert semantic_model_installed(model)


def test_clear_rejects_a_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("keep", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        clear_managed_resource(path, EMBEDDING_CACHE_MARKER)

    assert path.read_text(encoding="utf-8") == "keep"


def test_prune_removes_oldest_files_to_fit_limit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    oldest = cache / "old.npy"
    newest = cache / "new.npy"
    oldest.write_bytes(b"1234")
    newest.write_bytes(b"56789")
    os.utime(oldest, ns=(1, 1))
    os.utime(newest, ns=(2, 2))

    before, after = prune_embedding_cache(cache, 5)

    assert before.size_bytes == 9
    assert after.size_bytes == 5
    assert not oldest.exists()
    assert newest.exists()
    with pytest.raises(ValueError, match="must not be negative"):
        prune_embedding_cache(cache, -1)


def test_management_refuses_unmarked_directories(tmp_path: Path) -> None:
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "book.epub").write_bytes(b"keep")

    with pytest.raises(ValueError, match="refusing to manage unmarked"):
        clear_managed_resource(directory, EMBEDDING_CACHE_MARKER)
    with pytest.raises(ValueError, match="refusing to manage unmarked"):
        prune_embedding_cache(directory, 0)

    assert (directory / "book.epub").read_bytes() == b"keep"


def test_cache_management_refuses_active_work(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    resource_activity_marker(cache, os.getpid()).touch()

    with pytest.raises(RuntimeError, match="resource is in use"):
        clear_managed_resource(cache, EMBEDDING_CACHE_MARKER)

    assert cache.exists()


def test_cache_management_removes_stale_activity_marker(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    stale = resource_activity_marker(cache, 999_999_999)
    stale.touch()

    clear_managed_resource(cache, EMBEDDING_CACHE_MARKER)

    assert not cache.exists()


def test_model_management_refuses_active_work(tmp_path: Path) -> None:
    model = tmp_path / "model"
    mark_resource(model, MODEL_STORAGE_MARKER)
    resource_activity_marker(model, os.getpid()).touch()

    with pytest.raises(RuntimeError, match="resource is in use"):
        clear_managed_resource(model, MODEL_STORAGE_MARKER)

    assert model.exists()


def test_prepare_model_uses_worker_and_offline_verification(tmp_path: Path) -> None:
    calls: list[tuple[Sequence[str], Mapping[str, str]]] = []

    def runner(
        command: Sequence[str], *, check: bool, env: Mapping[str, str]
    ) -> subprocess.CompletedProcess[str]:
        assert check
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    arguments = (
        tmp_path / "python",
        tmp_path / "worker.py",
        "model",
        "revision",
        tmp_path / "model-storage",
    )
    prepare_model(*arguments, runner=runner)
    prepare_model(*arguments, offline=True, runner=runner)

    assert calls[0][0][-1] == "--prepare-model"
    assert "HF_HUB_OFFLINE" not in calls[0][1]
    assert calls[1][1]["HF_HUB_OFFLINE"] == "1"
    assert calls[1][1]["TRANSFORMERS_OFFLINE"] == "1"


def test_prepare_model_reports_process_failures(tmp_path: Path) -> None:
    def missing(_command: Sequence[str], **_kwargs: object) -> None:
        raise FileNotFoundError

    with pytest.raises(RuntimeError, match="Python not found"):
        prepare_model(tmp_path / "python", tmp_path / "worker", "m", "r", tmp_path, runner=missing)

    def failed(command: Sequence[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(RuntimeError, match="could not verify"):
        prepare_model(
            tmp_path / "python",
            tmp_path / "worker",
            "m",
            "r",
            tmp_path,
            offline=True,
            runner=failed,
        )
