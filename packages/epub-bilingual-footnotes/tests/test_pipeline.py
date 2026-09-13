from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from bilingual_text_align import (
    AlignmentLink,
    ContainmentMatch,
    TextUnit,
    TextWindowConfig,
)
from bs4 import BeautifulSoup

from epub_bilingual_footnotes.artifacts import write_artifact
from epub_bilingual_footnotes.extraction import content_blocks, read_book
from epub_bilingual_footnotes.model import Anchor, Artifact, Book
from epub_bilingual_footnotes.pipeline import align_epubs, build_epub
from epub_bilingual_footnotes.rendering import _document, _Insertion, _marker, render_epub
from epub_bilingual_footnotes.service import EpubFootnoteService


class ScriptedAligner:
    @property
    def metadata(self) -> Mapping[str, object]:
        return {"name": "scripted"}

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        assert len(first) == len(second)
        return tuple(AlignmentLink((index,), (index,), 1.0) for index in range(len(first)))


class ScriptedContainmentAligner(ScriptedAligner):
    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        if len(first) == 2 and len(second) == 6:
            return (
                AlignmentLink((), (0, 1), None),
                AlignmentLink((0,), (2,), 1.0),
                AlignmentLink((1,), (3,), 1.0),
                AlignmentLink((), (4, 5), None),
            )
        if len(first) == 6 and len(second) == 2:
            return (
                AlignmentLink((0, 1), (), None),
                AlignmentLink((2,), (0,), 1.0),
                AlignmentLink((3,), (1,), 1.0),
                AlignmentLink((4, 5), (), None),
            )
        return super().align(first, second)

    def locate(
        self, query_texts: Sequence[str], corpus_texts: Sequence[str], config: object = None
    ) -> ContainmentMatch:
        del query_texts, config
        assert len(corpus_texts) == 3
        return ContainmentMatch(
            corpus_start=1,
            corpus_end=2,
            pairs=((0, 1),),
            raw_score=0.6,
            normalized_score=0.6,
            coverage=1.0,
            mean_similarity=0.8,
            runner_up_normalized_score=0.2,
            runner_up_margin=0.4,
            accepted=True,
            rejection_reasons=(),
        )


def make_epub(path: Path, documents: list[tuple[int, str]], language: str = "fr") -> None:
    """Create an EPUB whose physical document split has no semantic meaning."""
    manifest, spine, contents = [], [], {}
    for identifier, text in documents:
        name, item_id = f"document-{identifier}.xhtml", f"document-{identifier}"
        manifest.append(f'<item id="{item_id}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{item_id}"/>')
        contents[name] = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title/></head><body>'
            f"<h1>Arbitrary section {identifier}</h1><p>{text}</p></body></html>"
        )
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata>'
        f'<language xmlns="http://purl.org/dc/elements/1.1/">{language}</language></metadata>'
        f"<manifest>{''.join(manifest)}</manifest><spine>{''.join(spine)}</spine></package>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
        '<rootfiles><rootfile full-path="content.opf"/></rootfiles></container>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr("style.css", "body { color: black; }")
        for name, document in contents.items():
            archive.writestr(name, document)


def test_build_aligns_global_streams_across_different_document_splits(tmp_path: Path) -> None:
    base = tmp_path / "base.epub"
    notes = tmp_path / "notes.epub"
    output = tmp_path / "out.epub"
    make_epub(base, [(10, "Il entra."), (90, "Il sourit.")])
    make_epub(notes, [(7, "He entered. He smiled.")], "en")

    artifact = build_epub(base, notes, output, ScriptedAligner())

    assert artifact.schema_version == 3
    assert artifact.base_sha256 == hashlib.sha256(base.read_bytes()).hexdigest()
    assert [anchor.document for anchor in artifact.base_anchors] == [
        "document-10.xhtml",
        "document-90.xhtml",
    ]
    with zipfile.ZipFile(output) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("style.css") == b"body { color: black; }"
        first = BeautifulSoup(archive.read("document-10.xhtml"), "html.parser")
        second = BeautifulSoup(archive.read("document-90.xhtml"), "html.parser")
        first_marker = first.select_one("a.bf-noteref")
        second_marker = second.select_one("a.bf-noteref")
        first_note = first.select_one('[epub\\:type="footnote"]')
        second_note = second.select_one('[epub\\:type="footnote"]')
        assert first_marker is not None
        assert second_marker is not None
        assert first_note is not None
        assert second_note is not None
        assert first_marker.get_text(strip=True) == "1"
        assert second_marker.get_text(strip=True) == "2"
        assert first_note.get_text(" ", strip=True) == "He entered. ↩"
        assert second_note.get_text(" ", strip=True) == "He smiled. ↩"


class CrossDocumentAligner(ScriptedAligner):
    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        assert len(first) == len(second) == 2
        return (AlignmentLink((0, 1), (0, 1), 0.8),)


def test_many_to_many_link_uses_last_base_anchor_and_joins_note_text(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(10, "Il entre."), (20, "Il sourit.")])
    make_epub(notes, [(30, "He enters. He smiles.")], "en")

    artifact = build_epub(base, notes, output, CrossDocumentAligner())

    assert artifact.links[0].first_indices == (0, 1)
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(output) as rendered:
        assert rendered.read("document-10.xhtml") == source.read("document-10.xhtml")
        final_document = BeautifulSoup(rendered.read("document-20.xhtml"), "html.parser")
    marker = final_document.select_one("a.bf-noteref")
    note = final_document.select_one('[epub\\:type="footnote"]')
    assert marker is not None
    assert note is not None
    assert marker.get_text(strip=True) == "1"
    assert note.get_text(" ", strip=True) == "He enters. He smiles. ↩"


def test_application_service_uses_structural_reader_and_renderer_ports() -> None:
    base = Book(
        "base.epub",
        "base-hash",
        (TextUnit(0, "Bonjour.", 0),),
        (Anchor("text.xhtml", 0, 0, 8),),
    )
    footnotes = Book(
        "notes.epub",
        "notes-hash",
        (TextUnit(0, "Hello.", 0),),
        (Anchor("notes.xhtml", 0, 0, 6),),
    )
    books = {"base.epub": base, "notes.epub": footnotes}
    rendered: list[tuple[str | Path, Artifact, str | Path]] = []

    def reader(path: str | Path) -> Book:
        return books[str(path)]

    def renderer(base_epub: str | Path, artifact: Artifact, output: str | Path) -> None:
        rendered.append((base_epub, artifact, output))

    service = EpubFootnoteService(reader, renderer)
    artifact = service.build("base.epub", "notes.epub", "output.epub", ScriptedAligner())

    assert artifact.base_sha256 == "base-hash"
    assert artifact.footnote_sha256 == "notes-hash"
    assert rendered == [("base.epub", artifact, "output.epub")]


def test_application_composes_containment_then_alignment_and_remaps_indices(
    tmp_path: Path,
) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "omnibus.epub", tmp_path / "out.epub"
    make_epub(base, [(8, "Bonjour."), (9, "Au revoir.")])
    make_epub(
        notes,
        [
            (80, "Before one. Before two. Hello."),
            (90, "Goodbye. After one. After two."),
        ],
        "en",
    )
    service = EpubFootnoteService(
        read_book,
        render_epub,
        TextWindowConfig(size=2, stride=2, sampled_texts=2),
    )

    artifact = service.build(base, notes, output, ScriptedContainmentAligner())

    assert artifact.scope is not None
    assert artifact.scope.classification == "complete-editions"
    assert (artifact.scope.footnote_start, artifact.scope.footnote_end) == (0, 6)
    assert artifact.scope.evidence["boundary_context"] == 4
    assert [link.second_indices for link in artifact.links] == [
        (0, 1),
        (2,),
        (3,),
        (4, 5),
    ]
    with zipfile.ZipFile(output) as archive:
        rendered = BeautifulSoup(archive.read("document-8.xhtml"), "html.parser")
    assert [
        note.get_text(" ", strip=True) for note in rendered.select('[epub\\:type="footnote"]')
    ] == [
        "Hello. ↩",
    ]
    with zipfile.ZipFile(output) as archive:
        next_document = BeautifulSoup(archive.read("document-9.xhtml"), "html.parser")
    assert [
        note.get_text(" ", strip=True) for note in next_document.select('[epub\\:type="footnote"]')
    ] == ["Goodbye. ↩"]


def test_containment_fine_alignment_can_cross_epub_document_boundaries(
    tmp_path: Path,
) -> None:
    class BoundarySpanningAligner(ScriptedContainmentAligner):
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def align(
            self, first: Sequence[TextUnit], second: Sequence[TextUnit]
        ) -> Sequence[AlignmentLink]:
            self.calls.append(
                (tuple(unit.text for unit in first), tuple(unit.text for unit in second))
            )
            return (
                AlignmentLink((), (0, 1), None),
                AlignmentLink((0, 1), (2, 3), 0.8),
                AlignmentLink((), (4, 5), None),
            )

    base = tmp_path / "base.epub"
    notes = tmp_path / "omnibus.epub"
    output = tmp_path / "out.epub"
    make_epub(base, [(8, "Bonjour."), (9, "Au revoir.")])
    make_epub(
        notes,
        [
            (80, "Before one. Before two. Hello."),
            (90, "Goodbye. After one. After two."),
        ],
        "en",
    )
    algorithm = BoundarySpanningAligner()
    service = EpubFootnoteService(
        read_book,
        render_epub,
        TextWindowConfig(size=2, stride=2, sampled_texts=2),
    )

    artifact = service.build(base, notes, output, algorithm)

    assert algorithm.calls == [
        (
            ("Bonjour.", "Au revoir."),
            (
                "Before one.",
                "Before two.",
                "Hello.",
                "Goodbye.",
                "After one.",
                "After two.",
            ),
        )
    ]
    assert artifact.links[1] == AlignmentLink((0, 1), (2, 3), 0.8)
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(output) as rendered:
        assert rendered.read("document-8.xhtml") == source.read("document-8.xhtml")
        final_document = BeautifulSoup(rendered.read("document-9.xhtml"), "html.parser")
    note = final_document.select_one('[epub\\:type="footnote"]')
    assert note is not None
    assert note.get_text(" ", strip=True) == "Hello. Goodbye. ↩"


def test_application_rejects_unaccepted_containment() -> None:
    class RejectingLocator(ScriptedContainmentAligner):
        def locate(
            self, query_texts: Sequence[str], corpus_texts: Sequence[str], config: object = None
        ) -> ContainmentMatch:
            del query_texts, corpus_texts, config
            return ContainmentMatch(
                0, 1, (), 0.0, 0.0, 0.2, 0.1, None, None, False, ("weak match",)
            )

    book = Book(
        "book.epub",
        "hash",
        (TextUnit(0, "Text.", 0),),
        (Anchor("text.xhtml", 0, 0, 5),),
    )
    service = EpubFootnoteService(lambda _path: book, lambda *_args: None)
    with pytest.raises(ValueError, match="ambiguous or too weak"):
        service.align("one.epub", "two.epub", RejectingLocator())


def test_application_can_locate_footnote_volume_inside_larger_base(tmp_path: Path) -> None:
    base, notes = tmp_path / "base-omnibus.epub", tmp_path / "notes.epub"
    make_epub(base, [(1, "Before one. Before two. Bonjour. Au revoir. After one. After two.")])
    make_epub(notes, [(2, "Hello. Goodbye.")], "en")
    service = EpubFootnoteService(
        read_book,
        render_epub,
        TextWindowConfig(size=2, stride=2, sampled_texts=2),
    )

    artifact = service.align(base, notes, ScriptedContainmentAligner())

    assert artifact.scope is not None
    assert artifact.scope.classification == "complete-editions"
    assert (artifact.scope.base_start, artifact.scope.base_end) == (0, 6)
    assert [link.first_indices for link in artifact.links] == [
        (0, 1),
        (2,),
        (3,),
        (4, 5),
    ]
    assert [link.second_indices for link in artifact.links] == [(), (0,), (1,), ()]


def test_application_renders_last_translation_on_story_before_colophon(tmp_path: Path) -> None:
    class BoundaryAwareAligner:
        @property
        def metadata(self) -> Mapping[str, object]:
            return {"name": "boundary-aware"}

        def locate(
            self,
            query_texts: Sequence[str],
            corpus_texts: Sequence[str],
            config: object = None,
        ) -> ContainmentMatch:
            del query_texts, config
            assert len(corpus_texts) == 3
            return ContainmentMatch(0, 3, ((0, 0),), 1.0, 1.0, 1.0, 1.0, None, None, True, ())

        def align(
            self, first: Sequence[TextUnit], second: Sequence[TextUnit]
        ) -> Sequence[AlignmentLink]:
            assert len(first) == 6
            assert len(second) == 3
            return (
                AlignmentLink((0, 1), (0,), 0.7),
                AlignmentLink((2,), (1,), 0.9),
                AlignmentLink((3, 4, 5), (2,), 0.7),
            )

        def similarities(
            self, query_texts: Sequence[str], corpus_texts: Sequence[str]
        ) -> Sequence[Sequence[float]]:
            if tuple(query_texts) == ("The opening.",):
                assert tuple(corpus_texts) == ("Ouverture.", "Préface. Ouverture.")
                return [[0.91, 0.89]]
            assert tuple(query_texts) == ("The ending.",)
            assert tuple(corpus_texts) == (
                "La fin du récit.",
                "La fin du récit. FIN.",
                "La fin du récit. FIN. Imprimerie.",
            )
            return [[0.88, 0.90, 0.71]]

    base = tmp_path / "alice-fr.epub"
    footnotes = tmp_path / "alice-en.epub"
    output = tmp_path / "alice-bilingual.epub"
    make_epub(
        base,
        [
            (0, "Préface."),
            (1, "Ouverture."),
            (2, "Le milieu."),
            (3, "La fin du récit."),
            (4, "FIN."),
            (5, "Imprimerie."),
        ],
    )
    make_epub(
        footnotes,
        [(0, "The opening."), (1, "The middle."), (2, "The ending.")],
        "en",
    )
    service = EpubFootnoteService(
        read_book,
        render_epub,
        TextWindowConfig(size=2, stride=2, sampled_texts=2),
    )

    artifact = service.build(base, footnotes, output, BoundaryAwareAligner())

    assert artifact.links == (
        AlignmentLink((0,), (), None),
        AlignmentLink((1,), (0,), 0.7),
        AlignmentLink((2,), (1,), 0.9),
        AlignmentLink((3,), (2,), 0.7),
        AlignmentLink((4, 5), (), None),
    )
    assert artifact.scope is not None
    decisions = artifact.scope.evidence["boundary_reconciliation"]
    assert isinstance(decisions, dict)
    assert [item["selected_indices"] for item in decisions["selections"]] == [(1,), (3,)]

    with zipfile.ZipFile(base) as source, zipfile.ZipFile(output) as rendered:
        story = BeautifulSoup(rendered.read("document-3.xhtml"), "html.parser")
        assert rendered.read("document-4.xhtml") == source.read("document-4.xhtml")
        assert rendered.read("document-5.xhtml") == source.read("document-5.xhtml")
    marker = story.select_one("p > a.bf-noteref")
    note = story.select_one('[epub\\:type="footnote"]')
    assert marker is not None
    assert note is not None
    assert note.get_text(" ", strip=True) == "The ending. ↩"


def test_artifact_json(tmp_path: Path) -> None:
    base, notes = tmp_path / "base.epub", tmp_path / "notes.epub"
    make_epub(base, [(1, "One.")])
    make_epub(notes, [(99, "Un.")])
    artifact = align_epubs(base, notes, ScriptedAligner())
    path = tmp_path / "artifact.json"
    write_artifact(artifact, path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == 3
    assert data["links"][0]["first_indices"] == [0]


def test_extraction_is_independent_of_headings_and_file_boundaries(tmp_path: Path) -> None:
    epub = tmp_path / "book.epub"
    make_epub(epub, [(40, "Un. Deux."), (3, "Trois.")])
    book = read_book(epub)
    assert [unit.text for unit in book.units] == ["Un.", "Deux.", "Trois."]
    assert [unit.index for unit in book.units] == [0, 1, 2]
    assert [unit.group for unit in book.units] == [0, 0, 1]
    assert [anchor.document for anchor in book.anchors] == [
        "document-40.xhtml",
        "document-40.xhtml",
        "document-3.xhtml",
    ]


def test_extraction_uses_semantic_prose_not_page_furniture_or_navigation() -> None:
    soup = BeautifulSoup(
        """<html><body>
        <header><p>Publisher header.</p></header>
        <nav><p><a href='#start'>Contents</a></p></nav>
        <p>Actual story.</p>
        <div><img src='plate.jpg'/><br/><a href='#story'>Return to text</a></div>
        <footer><p>Publisher footer.</p></footer>
        </body></html>""",
        "html.parser",
    )

    assert [block.get_text(" ", strip=True) for block in content_blocks(soup)] == ["Actual story."]


def test_extraction_keeps_preformatted_and_nested_poetry_as_complete_blocks() -> None:
    soup = BeautifulSoup(
        """<html><body>
        <p>The tale was something like this:</p>
        <pre>Fury said to a mouse.\nI will prosecute you.</pre>
        <div class="poem margin240">
          <span>Canichon dit à la Souris.</span><br/>
          <div class="small1"><span>Je vais te condamner à mort.</span></div>
        </div>
        </body></html>""",
        "html.parser",
    )

    blocks = content_blocks(soup)

    assert [block.name for block in blocks] == ["p", "pre", "div"]
    assert " ".join(blocks[1].get_text(" ", strip=True).split()) == (
        "Fury said to a mouse. I will prosecute you."
    )
    assert blocks[2]["class"] == ["poem", "margin240"]
    assert " ".join(blocks[2].get_text(" ", strip=True).split()) == (
        "Canichon dit à la Souris. Je vais te condamner à mort."
    )


def test_extraction_rejects_invalid_and_empty_epubs(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.epub"
    invalid.write_bytes(b"bad")
    try:
        read_book(invalid)
    except ValueError as exc:
        assert "Invalid EPUB" in str(exc)
    else:
        raise AssertionError("expected invalid EPUB")
    empty = tmp_path / "empty.epub"
    make_epub(empty, [])
    try:
        read_book(empty)
    except ValueError as exc:
        assert "No prose" in str(exc)
    else:
        raise AssertionError("expected empty EPUB")


def test_render_rejects_hash_existing_output_and_changed_anchor(tmp_path: Path) -> None:
    base, notes = tmp_path / "base.epub", tmp_path / "notes.epub"
    make_epub(base, [(1, "One.")])
    make_epub(notes, [(2, "Un.")])
    artifact = align_epubs(base, notes, ScriptedAligner())
    replacement = tmp_path / "replacement.epub"
    make_epub(replacement, [(1, "Changed.")])
    try:
        render_epub(replacement, artifact, tmp_path / "out.epub")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected hash rejection")
    try:
        render_epub(base, artifact, base)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected output rejection")
    insertion = _Insertion(Anchor("document-1.xhtml", 0, 0, 4), "Wrong", "Note", 1)
    raw = zipfile.ZipFile(base).read("document-1.xhtml")
    try:
        _document(raw, [insertion])
    except ValueError as exc:
        assert "text changed" in str(exc)
    else:
        raise AssertionError("expected anchor rejection")

    failed_output = tmp_path / "failed.epub"
    changed = replace(artifact, base_units=(TextUnit(0, "Wrong", 0),))
    try:
        render_epub(base, changed, failed_output)
    except ValueError as exc:
        assert "text changed" in str(exc)
    else:
        raise AssertionError("expected render failure")
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed.epub.*.tmp"))


def test_rendering_rejects_missing_html_and_offset() -> None:
    try:
        _document(b"<p>Text</p>", [])
    except ValueError as exc:
        assert "html/body" in str(exc)
    else:
        raise AssertionError("expected malformed document")
    soup = BeautifulSoup("<html><body><p>Text</p></body></html>", "html.parser")
    assert soup.p is not None
    try:
        _marker(soup, soup.p, 99, 1)
    except ValueError as exc:
        assert "offset" in str(exc)
    else:
        raise AssertionError("expected missing offset")


class SkipAligner(ScriptedAligner):
    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        del first, second
        return (AlignmentLink((0,), (), None), AlignmentLink((), (0,), None))


def test_insertions_and_deletions_do_not_create_empty_notes(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "One.")])
    make_epub(notes, [(2, "Un.")])
    build_epub(base, notes, output, SkipAligner())
    with zipfile.ZipFile(output) as archive:
        soup = BeautifulSoup(archive.read("document-1.xhtml"), "html.parser")
        assert not soup.select("a.bf-noteref")
