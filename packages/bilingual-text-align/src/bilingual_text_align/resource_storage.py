"""Safe inspection and deletion of application-managed directories."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .resource_paths import EMBEDDING_CACHE_MARKER


@dataclass(frozen=True)
class StoredResource:
    path: Path
    size_bytes: int
    file_count: int

    @property
    def present(self) -> bool:
        return self.file_count > 0


def inspect_resource(path: Path) -> StoredResource:
    if not path.is_dir():
        return StoredResource(path, 0, 0)
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    seen: set[tuple[int, int]] = set()
    size_bytes = 0
    for candidate in files:
        status = candidate.stat()
        identity = (status.st_dev, status.st_ino)
        if identity not in seen:
            seen.add(identity)
            size_bytes += status.st_size
    return StoredResource(path, size_bytes, len(files))


def mark_resource(path: Path, marker: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / marker).touch(exist_ok=True)


def clear_managed_resource(path: Path, marker: str) -> StoredResource:
    status = inspect_resource(path)
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"managed resource is not a directory: {path}")
        _require_marker(path, marker)
        _require_resource_idle(path)
        shutil.rmtree(path)
    return status


def prune_embedding_cache(path: Path, maximum_bytes: int) -> tuple[StoredResource, StoredResource]:
    if maximum_bytes < 0:
        raise ValueError("cache size limit must not be negative")
    before = inspect_resource(path)
    if before.size_bytes <= maximum_bytes:
        return before, before
    _require_marker(path, EMBEDDING_CACHE_MARKER)
    _require_resource_idle(path)
    files = sorted(
        (candidate for candidate in path.rglob("*.npy") if candidate.is_file()),
        key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)),
    )
    remaining = before.size_bytes
    for candidate in files:
        if remaining <= maximum_bytes:
            break
        size = candidate.stat().st_size
        candidate.unlink()
        remaining -= size
    _remove_empty_directories(path)
    return before, inspect_resource(path)


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def resource_activity_marker(path: Path, pid: int | None = None) -> Path:
    return path / f".active-{os.getpid() if pid is None else pid}"


def _remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    directories: Sequence[Path] = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    )
    for directory in directories:
        with suppress(OSError):
            directory.rmdir()
    with suppress(OSError):
        root.rmdir()


def _require_marker(path: Path, marker: str) -> None:
    if not (path / marker).is_file():
        raise ValueError(f"refusing to manage unmarked directory: {path}")


def _require_resource_idle(path: Path) -> None:
    active: list[int] = []
    for marker in path.glob(".active-*"):
        try:
            pid = int(marker.name.removeprefix(".active-"))
        except ValueError:
            active.append(-1)
            continue
        if _process_running(pid):
            active.append(pid)
        else:
            marker.unlink(missing_ok=True)
    if active:
        raise RuntimeError("resource is in use; wait for active alignment work to finish")


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
