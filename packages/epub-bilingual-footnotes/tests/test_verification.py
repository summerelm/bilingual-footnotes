from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from bilingual_text_align import AlignmentLink
from test_pipeline import ScriptedAligner, make_epub

from epub_bilingual_footnotes.model import Artifact
from epub_bilingual_footnotes.pipeline import align_epubs, build_epub
from epub_bilingual_footnotes.rendering import render_epub
from epub_bilingual_footnotes.verification import VerificationReport, verify_epub


def rewrite_archive(
    source: Path,
    destination: Path,
    transform: Callable[[zipfile.ZipInfo, bytes], tuple[zipfile.ZipInfo | None, bytes]],
) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as rewritten:
        for info in original.infolist():
            data = original.read(info)
            changed_info, changed_data = transform(info, data)
            if changed_info is not None:
                rewritten.writestr(changed_info, changed_data)


def test_verifier_reports_hashes_coverage_preservation_and_notes(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un. Deux.")])
    make_epub(notes, [(2, "One. Two.")], "en")

    artifact = build_epub(base, notes, output, ScriptedAligner())

    assert artifact.verification is not None
    assert artifact.verification["base_sha256"] == hashlib.sha256(base.read_bytes()).hexdigest()
    assert (
        artifact.verification["footnote_sha256"] == hashlib.sha256(notes.read_bytes()).hexdigest()
    )
    assert artifact.verification["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert artifact.verification["base_unit_count"] == 2
    assert artifact.verification["footnote_unit_count"] == 2
    assert artifact.verification["link_count"] == 2
    assert artifact.verification["rendered_note_count"] == 2
    assert artifact.verification["base_only_link_count"] == 0
    assert artifact.verification["footnote_only_link_count"] == 0
    assert artifact.verification["modified_document_count"] == 1
    assert artifact.verification["unchanged_entry_count"] == 4


def test_verifier_rejects_source_hash_and_alignment_coverage(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un.")])
    make_epub(notes, [(2, "One.")], "en")
    artifact = align_epubs(base, notes, ScriptedAligner())
    render_epub(base, artifact, output)

    with pytest.raises(ValueError, match="base input hash"):
        verify_epub(base, notes, output, replace(artifact, base_sha256="wrong"))
    with pytest.raises(ValueError, match="footnote input hash"):
        verify_epub(base, notes, output, replace(artifact, footnote_sha256="wrong"))
    invalid = replace(artifact, links=(AlignmentLink((0,), (), None),))
    with pytest.raises(ValueError, match="footnote alignment coverage"):
        verify_epub(base, notes, output, invalid)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda artifact: replace(artifact, base_anchors=()), "units and anchors"),
        (
            lambda artifact: replace(
                artifact, base_units=(replace(artifact.base_units[0], index=2),)
            ),
            "base unit indices",
        ),
        (
            lambda artifact: replace(
                artifact, footnote_units=(replace(artifact.footnote_units[0], index=2),)
            ),
            "footnote unit indices",
        ),
        (lambda artifact: replace(artifact, links=(AlignmentLink((), (), None),)), "empty"),
        (
            lambda artifact: replace(artifact, links=(AlignmentLink((), (0,), None),)),
            "base alignment coverage",
        ),
    ],
)
def test_verifier_rejects_invalid_artifact_structure(
    tmp_path: Path, change: Callable[[Artifact], Artifact], message: str
) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un.")])
    make_epub(notes, [(2, "One.")], "en")
    artifact = align_epubs(base, notes, ScriptedAligner())
    render_epub(base, artifact, output)

    invalid = change(artifact)
    with pytest.raises(ValueError, match=message):
        verify_epub(base, notes, output, invalid)


def test_verifier_rejects_archive_and_note_corruption(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un.")])
    make_epub(notes, [(2, "One.")], "en")
    artifact = align_epubs(base, notes, ScriptedAligner())
    render_epub(base, artifact, output)
    replacement = tmp_path / "replacement.epub"
    rewrite_archive(
        output,
        replacement,
        lambda info, data: (
            info,
            data.replace(b'href="#bf-ref-1"', b'href="#missing"')
            if info.filename == "document-1.xhtml"
            else data,
        ),
    )

    with pytest.raises(ValueError, match="needs one backlink"):
        verify_epub(base, notes, replacement, artifact)

    invalid = tmp_path / "invalid.epub"
    invalid.write_bytes(b"not an EPUB")
    with pytest.raises(ValueError, match="invalid archive"):
        verify_epub(base, notes, invalid, artifact)

    missing = tmp_path / "missing-base.epub"
    with pytest.raises(ValueError, match="cannot read"):
        verify_epub(missing, notes, output, artifact)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("names", "entry order or names"),
        ("mimetype", "first stored mimetype"),
        ("metadata", "ZIP metadata changed"),
        ("unchanged", "unchanged entry differs"),
        ("marker", "marker and note identifiers"),
        ("pair", "marker/note pair"),
    ],
)
def test_verifier_rejects_specific_archive_contract_breaks(
    tmp_path: Path, kind: str, message: str
) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un.")])
    make_epub(notes, [(2, "One.")], "en")
    artifact = align_epubs(base, notes, ScriptedAligner())
    render_epub(base, artifact, output)
    broken = tmp_path / "broken.epub"

    def transform(info: zipfile.ZipInfo, data: bytes) -> tuple[zipfile.ZipInfo | None, bytes]:
        if kind == "names" and info.filename == "style.css":
            return None, data
        if kind == "mimetype" and info.filename == "mimetype":
            info.compress_type = zipfile.ZIP_DEFLATED
        if kind == "metadata" and info.filename == "style.css":
            info.external_attr += 1
        if kind == "unchanged" and info.filename == "style.css":
            data += b" changed"
        if info.filename == "document-1.xhtml":
            if kind == "marker":
                data = data.replace(b'id="bf-ref-1"', b'id="missing"')
            if kind == "pair":
                data = data.replace(b'href="#bf-note-1"', b'href="#missing"')
        return info, data

    rewrite_archive(output, broken, transform)
    with pytest.raises(ValueError, match=message):
        verify_epub(base, notes, broken, artifact)


def test_build_removes_new_output_when_verification_fails(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un.")])
    make_epub(notes, [(2, "One.")], "en")

    def fail(*_args: object) -> VerificationReport:
        raise ValueError("verification failed deliberately")

    with pytest.raises(ValueError, match="deliberately"):
        build_epub(base, notes, output, ScriptedAligner(), verifier=fail)
    assert not output.exists()


def test_verifier_reports_explicit_gaps(tmp_path: Path) -> None:
    base, notes, output = tmp_path / "base.epub", tmp_path / "notes.epub", tmp_path / "out.epub"
    make_epub(base, [(1, "Un. Deux.")])
    make_epub(notes, [(2, "One. Two.")], "en")
    artifact = align_epubs(base, notes, ScriptedAligner())
    gap_artifact = replace(
        artifact,
        links=(
            AlignmentLink((0,), (0,), 1.0),
            AlignmentLink((1,), (), None),
            AlignmentLink((), (1,), None),
        ),
    )
    render_epub(base, gap_artifact, output)

    report = verify_epub(base, notes, output, gap_artifact)

    assert report.rendered_note_count == 1
    assert report.base_only_link_count == 1
    assert report.footnote_only_link_count == 1
