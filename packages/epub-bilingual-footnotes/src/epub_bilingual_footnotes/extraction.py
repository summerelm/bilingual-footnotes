"""Read one spine-ordered text stream and durable anchors from an EPUB."""

from __future__ import annotations

import hashlib
import posixpath
import zipfile
from pathlib import Path

from bilingual_text_align import TextUnit
from bilingual_text_align.segmentation import sentence_spans
from bs4 import BeautifulSoup, Tag
from defusedxml import ElementTree

from .archive_safety import MAX_DOCUMENT_BYTES, read_member, validate_archive
from .model import Anchor, Book

_BLOCKS = {"p", "li", "blockquote", "div", "pre"}
_POETRY_CLASSES = {"poem", "poetry", "stanza", "verse"}


def _navigation_only(tag: Tag) -> bool:
    """Return whether all visible text belongs to links rather than prose."""
    text_parts = list(tag.stripped_strings)
    linked_parts = [part for link in tag.find_all("a") for part in link.stripped_strings]
    return bool(text_parts) and text_parts == linked_parts


def _poetry_container(tag: Tag) -> bool:
    """Return whether a block's internal layout belongs to one poem."""
    class_values = tag.get("class")
    classes = (
        {str(value).lower() for value in class_values} if isinstance(class_values, list) else set()
    )
    epub_types = {value.lower() for value in str(tag.get("epub:type", "")).split()}
    return (
        tag.name == "pre" or bool(classes & _POETRY_CLASSES) or bool(epub_types & _POETRY_CLASSES)
    )


def content_blocks(soup: BeautifulSoup) -> list[Tag]:
    """Return non-overlapping prose blocks in document reading order."""
    return [
        tag
        for tag in soup.find_all(_BLOCKS)
        if tag.get_text("", strip=True)
        and not any(_poetry_container(parent) for parent in tag.parents if isinstance(parent, Tag))
        and not (tag.name == "div" and tag.find(_BLOCKS) and not _poetry_container(tag))
        and not any(parent.name in {"p", "li", "blockquote"} for parent in tag.parents)
        and not tag.find_parent(["nav", "aside", "header", "footer"])
        and not _navigation_only(tag)
    ]


def _documents(zf: zipfile.ZipFile) -> list[str]:
    container = ElementTree.fromstring(
        read_member(zf, "META-INF/container.xml", maximum_bytes=MAX_DOCUMENT_BYTES)
    )
    opf = next(
        node.attrib["full-path"] for node in container.iter() if node.tag.endswith("rootfile")
    )
    root = ElementTree.fromstring(read_member(zf, opf, maximum_bytes=MAX_DOCUMENT_BYTES))
    manifest = {
        node.attrib["id"]: node.attrib["href"]
        for node in root.iter()
        if node.tag.endswith("item")
        and node.attrib.get("media-type") in {"application/xhtml+xml", "text/html"}
    }
    base = posixpath.dirname(opf)
    return [
        posixpath.normpath(posixpath.join(base, manifest[node.attrib["idref"]]))
        for node in root.iter()
        if node.tag.endswith("itemref") and node.attrib.get("idref") in manifest
    ]


def read_book(path: str | Path) -> Book:
    """Extract all prose as one global sequence, independent of section layout."""
    epub = Path(path)
    units: list[TextUnit] = []
    anchors: list[Anchor] = []
    group = 0
    try:
        with zipfile.ZipFile(epub) as archive:
            validate_archive(archive)
            for document in _documents(archive):
                soup = BeautifulSoup(
                    read_member(archive, document, maximum_bytes=MAX_DOCUMENT_BYTES),
                    "html.parser",
                )
                for block_index, block in enumerate(content_blocks(soup)):
                    text = "".join(block.strings)
                    spans = sentence_spans(text)
                    for start, end in spans:
                        units.append(TextUnit(len(units), text[start:end], group))
                        anchors.append(Anchor(document, block_index, start, end))
                    if spans:
                        group += 1
    except (KeyError, StopIteration, ElementTree.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid EPUB {epub}: {exc}") from exc
    if not units:
        raise ValueError(f"No prose text found in {epub}")
    return Book(
        str(epub), hashlib.sha256(epub.read_bytes()).hexdigest(), tuple(units), tuple(anchors)
    )
