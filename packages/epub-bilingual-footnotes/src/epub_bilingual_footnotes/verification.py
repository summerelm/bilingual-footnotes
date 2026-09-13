"""Verify a rendered EPUB against its inputs and alignment artifact."""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .archive_safety import MAX_DOCUMENT_BYTES, read_member, validate_archive
from .model import Artifact

_CONTENT_SUFFIXES = (".html", ".htm", ".xhtml")
_MARKER_ID = re.compile(r"bf-ref-(\d+)\Z")
_NOTE_ID = re.compile(r"bf-note-(\d+)\Z")


@dataclass(frozen=True)
class VerificationReport:
    """Retainable structural evidence for one completed render."""

    base_sha256: str
    footnote_sha256: str
    output_sha256: str
    entry_count: int
    unchanged_entry_count: int
    modified_document_count: int
    base_unit_count: int
    footnote_unit_count: int
    link_count: int
    rendered_note_count: int
    base_only_link_count: int
    footnote_only_link_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_epub(
    base_epub: str | Path,
    footnote_epub: str | Path,
    output_epub: str | Path,
    artifact: Artifact,
) -> VerificationReport:
    """Validate source identity, coverage, archive preservation, and note links."""
    base, footnotes, output = Path(base_epub), Path(footnote_epub), Path(output_epub)
    base_hash = _sha256(base)
    footnote_hash = _sha256(footnotes)
    if base_hash != artifact.base_sha256:
        raise ValueError("EPUB verification failed: base input hash does not match artifact")
    if footnote_hash != artifact.footnote_sha256:
        raise ValueError("EPUB verification failed: footnote input hash does not match artifact")

    _validate_units_and_links(artifact)
    modified_documents = {
        artifact.base_anchors[link.first_indices[-1]].document
        for link in artifact.links
        if link.first_indices and link.second_indices
    }
    rendered_notes = sum(
        bool(link.first_indices and link.second_indices) for link in artifact.links
    )
    entry_count = _validate_archive(base, output, modified_documents, rendered_notes)
    return VerificationReport(
        base_sha256=base_hash,
        footnote_sha256=footnote_hash,
        output_sha256=_sha256(output),
        entry_count=entry_count,
        unchanged_entry_count=entry_count - len(modified_documents),
        modified_document_count=len(modified_documents),
        base_unit_count=len(artifact.base_units),
        footnote_unit_count=len(artifact.footnote_units),
        link_count=len(artifact.links),
        rendered_note_count=rendered_notes,
        base_only_link_count=sum(
            bool(link.first_indices and not link.second_indices) for link in artifact.links
        ),
        footnote_only_link_count=sum(
            bool(link.second_indices and not link.first_indices) for link in artifact.links
        ),
    )


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError(f"EPUB verification failed: cannot read {path}: {exc}") from exc


def _validate_units_and_links(artifact: Artifact) -> None:
    if len(artifact.base_units) != len(artifact.base_anchors):
        raise ValueError("EPUB verification failed: base units and anchors differ in length")
    if [unit.index for unit in artifact.base_units] != list(range(len(artifact.base_units))):
        raise ValueError("EPUB verification failed: base unit indices are not ordered")
    if [unit.index for unit in artifact.footnote_units] != list(
        range(len(artifact.footnote_units))
    ):
        raise ValueError("EPUB verification failed: footnote unit indices are not ordered")
    if any(not link.first_indices and not link.second_indices for link in artifact.links):
        raise ValueError("EPUB verification failed: an alignment link is empty on both sides")
    first = [index for link in artifact.links for index in link.first_indices]
    second = [index for link in artifact.links for index in link.second_indices]
    if first != list(range(len(artifact.base_units))):
        raise ValueError("EPUB verification failed: base alignment coverage is incomplete")
    if second != list(range(len(artifact.footnote_units))):
        raise ValueError("EPUB verification failed: footnote alignment coverage is incomplete")


def _validate_archive(
    base: Path, output: Path, modified_documents: set[str], expected_notes: int
) -> int:
    try:
        with zipfile.ZipFile(base) as source, zipfile.ZipFile(output) as rendered:
            validate_archive(source)
            validate_archive(rendered)
            source_entries = source.infolist()
            rendered_entries = rendered.infolist()
            source_names = [entry.filename for entry in source_entries]
            rendered_names = [entry.filename for entry in rendered_entries]
            if source_names != rendered_names:
                raise ValueError("EPUB verification failed: archive entry order or names changed")
            if not rendered_entries:
                raise ValueError("EPUB verification failed: output archive is empty")
            mimetype = rendered_entries[0]
            if (
                mimetype.filename != "mimetype"
                or mimetype.compress_type != zipfile.ZIP_STORED
                or read_member(rendered, mimetype, maximum_bytes=MAX_DOCUMENT_BYTES)
                != b"application/epub+zip"
            ):
                raise ValueError("EPUB verification failed: first stored mimetype is invalid")
            bad_entry = rendered.testzip()
            if bad_entry is not None:
                raise ValueError(f"EPUB verification failed: corrupt archive entry {bad_entry}")
            for source_info, rendered_info in zip(source_entries, rendered_entries, strict=True):
                if _stable_zip_metadata(source_info) != _stable_zip_metadata(rendered_info):
                    raise ValueError(
                        f"EPUB verification failed: ZIP metadata changed for {source_info.filename}"
                    )
                if source_info.filename not in modified_documents and read_member(
                    source, source_info, maximum_bytes=source_info.file_size
                ) != read_member(rendered, rendered_info, maximum_bytes=rendered_info.file_size):
                    raise ValueError(
                        f"EPUB verification failed: unchanged entry differs: {source_info.filename}"
                    )
            _validate_notes(rendered, expected_notes)
            return len(rendered_entries)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"EPUB verification failed: invalid archive: {exc}") from exc


def _stable_zip_metadata(info: zipfile.ZipInfo) -> tuple[object, ...]:
    return (
        info.filename,
        info.date_time,
        info.compress_type,
        info.comment,
        info.extra,
        info.internal_attr,
        info.external_attr,
        info.create_system,
    )


def _validate_notes(archive: zipfile.ZipFile, expected: int) -> None:
    markers, notes = _generated_elements(archive)
    expected_numbers = set(range(1, expected + 1))
    if set(markers) != expected_numbers or set(notes) != expected_numbers:
        raise ValueError("EPUB verification failed: marker and note identifiers are incomplete")
    for number in sorted(expected_numbers):
        _validate_note_pair(number, markers[number], notes[number])


def _generated_elements(
    archive: zipfile.ZipFile,
) -> tuple[dict[int, tuple[str, Tag]], dict[int, tuple[str, Tag]]]:
    markers: dict[int, tuple[str, Tag]] = {}
    notes: dict[int, tuple[str, Tag]] = {}
    for name in archive.namelist():
        if not name.lower().endswith(_CONTENT_SUFFIXES):
            continue
        soup = BeautifulSoup(
            read_member(archive, name, maximum_bytes=MAX_DOCUMENT_BYTES), "html.parser"
        )
        for tag in soup.find_all(id=_MARKER_ID):
            number = _identifier_number(tag)
            if number in markers:
                raise ValueError(f"EPUB verification failed: duplicate marker {number}")
            markers[number] = (name, tag)
        for tag in soup.find_all(id=_NOTE_ID):
            number = _identifier_number(tag)
            if number in notes:
                raise ValueError(f"EPUB verification failed: duplicate note {number}")
            notes[number] = (name, tag)
    return markers, notes


def _validate_note_pair(
    number: int, marker_item: tuple[str, Tag], note_item: tuple[str, Tag]
) -> None:
    marker_document, marker = marker_item
    note_document, note = note_item
    marker_classes = marker.get("class")
    has_marker_class = isinstance(marker_classes, list) and "bf-noteref" in marker_classes
    if (
        marker_document != note_document
        or marker.name != "a"
        or not has_marker_class
        or marker.get("href") != f"#bf-note-{number}"
        or marker.get("epub:type") != "noteref"
        or note.name != "aside"
        or note.get("epub:type") != "footnote"
    ):
        raise ValueError(f"EPUB verification failed: marker/note pair {number} is invalid")
    backlinks = note.find_all("a", href=f"#bf-ref-{number}")
    if len(backlinks) != 1:
        raise ValueError(f"EPUB verification failed: note {number} needs one backlink")


def _identifier_number(tag: Tag) -> int:
    return int(str(tag["id"]).rsplit("-", 1)[1])
