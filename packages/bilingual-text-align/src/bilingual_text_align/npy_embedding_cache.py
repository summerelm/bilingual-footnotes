"""NumPy filesystem adapter for the embedding cache boundary."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

EMBEDDING_CACHE_MARKER = ".bilingual-footnotes-embeddings"


class NpyEmbeddingCache:
    """Content-addressed embedding storage backed by atomic NPY files."""

    def __init__(self, directory: Path, numpy: Any) -> None:
        self.directory = directory
        self.numpy = numpy
        directory.mkdir(parents=True, exist_ok=True)
        (directory / EMBEDDING_CACHE_MARKER).touch(exist_ok=True)

    def load(self, key: str, expected_prefix: tuple[int, ...]) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            value = self.numpy.load(path, allow_pickle=False)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if (shape is not None and tuple(shape[: len(expected_prefix)]) != expected_prefix) or (
            dtype is not None and str(dtype) != "float32"
        ):
            path.unlink(missing_ok=True)
            return None
        path.touch()
        return value

    def store(self, key: str, embeddings: Any) -> None:
        path = self._path(key)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            self.numpy.save(temporary_path, embeddings, allow_pickle=False)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.npy"
