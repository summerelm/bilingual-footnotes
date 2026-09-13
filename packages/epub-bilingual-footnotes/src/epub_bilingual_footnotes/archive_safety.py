"""Resource limits for EPUB archives supplied by users."""

from __future__ import annotations

import zipfile

MAX_ENTRY_COUNT = 10_000
MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024


def validate_archive(
    archive: zipfile.ZipFile,
    *,
    max_entries: int = MAX_ENTRY_COUNT,
    max_entry_bytes: int = MAX_ENTRY_BYTES,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
) -> None:
    """Reject ambiguous, encrypted, or excessively expanded ZIP content."""
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) > max_entries:
        raise ValueError(f"EPUB contains too many archive entries (maximum {max_entries})")
    if len(names) != len(set(names)):
        raise ValueError("EPUB contains duplicate archive entry names")
    if any(entry.flag_bits & 0x1 for entry in entries):
        raise ValueError("EPUB contains encrypted archive entries")
    if any(entry.file_size > max_entry_bytes for entry in entries):
        raise ValueError("EPUB contains an archive entry that is too large")
    if sum(entry.file_size for entry in entries) > max_archive_bytes:
        raise ValueError("EPUB expands beyond the supported archive size")


def read_member(
    archive: zipfile.ZipFile, member: str | zipfile.ZipInfo, *, maximum_bytes: int
) -> bytes:
    """Read one bounded member without trusting only its declared ZIP size."""
    info = archive.getinfo(member) if isinstance(member, str) else member
    if info.file_size > maximum_bytes:
        raise ValueError(f"EPUB archive entry is too large: {info.filename}")
    with archive.open(info) as source:
        data = source.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(f"EPUB archive entry is too large: {info.filename}")
    return data
