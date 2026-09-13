from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import pytest

from epub_bilingual_footnotes.archive_safety import read_member, validate_archive


def _archive(path: Path, entries: list[tuple[str, bytes]]) -> zipfile.ZipFile:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return zipfile.ZipFile(path)


def test_archive_limits_entry_count_and_expanded_sizes(tmp_path: Path) -> None:
    with _archive(tmp_path / "book.epub", [("one", b"1234"), ("two", b"56")]) as archive:
        with pytest.raises(ValueError, match="too many archive entries"):
            validate_archive(archive, max_entries=1)
        with pytest.raises(ValueError, match="entry that is too large"):
            validate_archive(archive, max_entry_bytes=3)
        with pytest.raises(ValueError, match="supported archive size"):
            validate_archive(archive, max_archive_bytes=5)
        with pytest.raises(ValueError, match="archive entry is too large"):
            read_member(archive, "one", maximum_bytes=3)


def test_archive_rejects_duplicate_names(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        archive = _archive(tmp_path / "duplicate.epub", [("same", b"1"), ("same", b"2")])
    with archive, pytest.raises(ValueError, match="duplicate archive entry names"):
        validate_archive(archive)
