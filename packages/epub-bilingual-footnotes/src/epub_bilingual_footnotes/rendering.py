"""Render global alignment links through their per-document EPUB anchors."""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from .archive_safety import MAX_ENTRY_BYTES, read_member, validate_archive
from .extraction import content_blocks
from .model import Anchor, Artifact
from .output_safety import publish_new_file


@dataclass(frozen=True)
class _Insertion:
    anchor: Anchor
    base_text: str
    note_text: str
    number: int


def _marker(soup: BeautifulSoup, block: Tag, offset: int, number: int) -> None:
    cursor = 0
    for node in list(block.find_all(string=True)):
        text = str(node)
        end = cursor + len(text)
        if cursor < offset <= end:
            local = offset - cursor
            link = soup.new_tag("a", href=f"#bf-note-{number}", id=f"bf-ref-{number}")
            link["class"], link["epub:type"], link["role"] = (
                "bf-noteref",
                "noteref",
                "doc-noteref",
            )
            sup = soup.new_tag("sup")
            sup.string = str(number)
            link.append(sup)
            node.replace_with(NavigableString(text[:local]), link, NavigableString(text[local:]))
            return
        cursor = end
    raise ValueError(f"Could not locate source offset {offset}")


def _document(raw: bytes, insertions: list[_Insertion]) -> bytes:
    soup = BeautifulSoup(raw, "html.parser")
    if soup.body is None or soup.html is None:
        raise ValueError("EPUB content document has no html/body")
    soup.html["xmlns:epub"] = "http://www.idpf.org/2007/ops"
    blocks = content_blocks(soup)
    for insertion in insertions:
        block = blocks[insertion.anchor.block]
        text = "".join(block.strings)
        if text[insertion.anchor.start : insertion.anchor.end] != insertion.base_text:
            raise ValueError("Base EPUB text changed after alignment")
    for insertion in sorted(
        insertions, key=lambda item: (item.anchor.block, item.anchor.end), reverse=True
    ):
        _marker(soup, blocks[insertion.anchor.block], insertion.anchor.end, insertion.number)
    if insertions:
        section = soup.new_tag("section")
        section["class"], section["epub:type"] = "bf-footnotes", "footnotes"
        for insertion in sorted(insertions, key=lambda item: item.number):
            aside = soup.new_tag("aside", id=f"bf-note-{insertion.number}")
            aside["epub:type"] = "footnote"
            paragraph = soup.new_tag("p")
            paragraph.append(NavigableString(insertion.note_text + " "))
            back = soup.new_tag("a", href=f"#bf-ref-{insertion.number}")
            back.string = "↩"
            paragraph.append(back)
            aside.append(paragraph)
            section.append(aside)
        soup.body.append(section)
    return str(soup).encode()


def _insertions(artifact: Artifact) -> dict[str, list[_Insertion]]:
    by_document: dict[str, list[_Insertion]] = defaultdict(list)
    number = 1
    for link in artifact.links:
        if not link.first_indices or not link.second_indices:
            continue
        base_index = link.first_indices[-1]
        anchor = artifact.base_anchors[base_index]
        note_text = " ".join(artifact.footnote_units[index].text for index in link.second_indices)
        by_document[anchor.document].append(
            _Insertion(anchor, artifact.base_units[base_index].text, note_text, number)
        )
        number += 1
    return by_document


def render_epub(base_epub: str | Path, artifact: Artifact, output_epub: str | Path) -> None:
    base, output = Path(base_epub), Path(output_epub)
    if base.resolve() == output.resolve() or output.exists():
        raise FileExistsError(f"Output must be a new path: {output}")
    if hashlib.sha256(base.read_bytes()).hexdigest() != artifact.base_sha256:
        raise ValueError("Base EPUB does not match alignment artifact")
    by_document = _insertions(artifact)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(base) as source, zipfile.ZipFile(temporary, "w") as target:
            validate_archive(source)
            for info in source.infolist():
                data = read_member(source, info, maximum_bytes=MAX_ENTRY_BYTES)
                if info.filename in by_document:
                    data = _document(data, by_document[info.filename])
                target.writestr(info, data)
        publish_new_file(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
