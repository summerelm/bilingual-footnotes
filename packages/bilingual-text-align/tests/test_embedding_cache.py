from __future__ import annotations

from pathlib import Path

import pytest

from bilingual_text_align.embedding_cache import DisabledEmbeddingCache
from bilingual_text_align.npy_embedding_cache import EMBEDDING_CACHE_MARKER, NpyEmbeddingCache


class CachedArray:
    shape = (2, 3, 4)
    dtype = "float32"


class FakeNumpy:
    def __init__(self) -> None:
        self.load_result: object = CachedArray()
        self.load_failure: OSError | ValueError | None = None
        self.save_failure: OSError | None = None
        self.loaded: list[tuple[Path, bool]] = []
        self.saved: list[tuple[Path, object, bool]] = []

    def load(self, path: Path, *, allow_pickle: bool) -> object:
        self.loaded.append((path, allow_pickle))
        if self.load_failure is not None:
            raise self.load_failure
        return self.load_result

    def save(self, path: Path, value: object, *, allow_pickle: bool) -> None:
        self.saved.append((path, value, allow_pickle))
        path.write_bytes(b"cache")
        if self.save_failure is not None:
            raise self.save_failure


def test_npy_cache_stores_and_loads_valid_embeddings(tmp_path: Path) -> None:
    numpy = FakeNumpy()
    cache = NpyEmbeddingCache(tmp_path / "nested" / "cache", numpy)
    embeddings = CachedArray()

    assert cache.load("missing", (2, 3)) is None
    cache.store("identity", embeddings)
    loaded = cache.load("identity", (2, 3))

    path = cache.directory / "identity.npy"
    assert loaded is numpy.load_result
    assert path.read_bytes() == b"cache"
    assert numpy.loaded == [(path, False)]
    assert len(numpy.saved) == 1
    assert numpy.saved[0][1:] == (embeddings, False)
    assert not numpy.saved[0][0].exists()
    assert (cache.directory / EMBEDDING_CACHE_MARKER).is_file()


@pytest.mark.parametrize("failure", [OSError("damaged"), ValueError("invalid")])
def test_npy_cache_discards_unreadable_entries(
    tmp_path: Path, failure: OSError | ValueError
) -> None:
    numpy = FakeNumpy()
    cache = NpyEmbeddingCache(tmp_path, numpy)
    cache.store("identity", CachedArray())
    numpy.load_failure = failure

    assert cache.load("identity", (2, 3)) is None
    assert not (tmp_path / "identity.npy").exists()


@pytest.mark.parametrize(
    "value",
    [
        type("WrongShape", (), {"shape": (9, 3), "dtype": "float32"})(),
        type("WrongDtype", (), {"shape": (2, 3), "dtype": "float64"})(),
    ],
)
def test_npy_cache_discards_entries_with_wrong_metadata(tmp_path: Path, value: object) -> None:
    numpy = FakeNumpy()
    cache = NpyEmbeddingCache(tmp_path, numpy)
    cache.store("identity", CachedArray())
    numpy.load_result = value

    assert cache.load("identity", (2, 3)) is None
    assert not (tmp_path / "identity.npy").exists()


def test_npy_cache_removes_partial_file_when_write_fails(tmp_path: Path) -> None:
    numpy = FakeNumpy()
    numpy.save_failure = OSError("interrupted write")
    cache = NpyEmbeddingCache(tmp_path, numpy)

    with pytest.raises(OSError, match="interrupted write"):
        cache.store("identity", CachedArray())

    assert not list(tmp_path.glob("*.npy"))


def test_disabled_cache_never_persists_values() -> None:
    cache = DisabledEmbeddingCache()

    assert cache.load("identity", (2,)) is None
    cache.store("identity", object())
