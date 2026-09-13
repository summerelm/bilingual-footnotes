"""Representative ten-sentence journeys through both public application paths."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest
from bilingual_text_align.plain_text import FootnoteSource, align_texts
from bilingual_text_align.resource_paths import default_model_storage, default_worker_python
from bilingual_text_align.semantic_runtime import require_semantic_runtime
from bilingual_text_align.vecalign_labse import VecalignLabseAligner
from bs4 import BeautifulSoup

from epub_bilingual_footnotes.cli import main

FIXTURES = Path(__file__).with_name("fixtures")
CACHE_FIXTURE = FIXTURES / "alice-embedding-cache"
REPOSITORY = Path(__file__).resolve().parents[3]
FRENCH_ALICE = tuple((FIXTURES / "alice-fr.txt").read_text(encoding="utf-8").splitlines())
ENGLISH_ALICE = tuple((FIXTURES / "alice-en.txt").read_text(encoding="utf-8").splitlines())


@pytest.fixture(scope="module")
def alice_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path, frozenset[str]]:
    worker_python = Path(
        os.environ.get(
            "BILINGUAL_FOOTNOTES_TEST_WORKER_PYTHON",
            REPOSITORY / default_worker_python(),
        )
    )
    model_storage = Path(
        os.environ.get("BILINGUAL_FOOTNOTES_TEST_MODEL_STORAGE", default_model_storage())
    )
    require_semantic_runtime(worker_python, model_storage)
    cached_files = _validate_cache_fixture()
    cache = tmp_path_factory.mktemp("alice-semantic") / "cache"
    shutil.copytree(CACHE_FIXTURE, cache)
    return worker_python, model_storage, cache, cached_files


def _validate_cache_fixture() -> frozenset[str]:
    manifest = json.loads((CACHE_FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "sentence-transformers/LaBSE"
    assert manifest["model_revision"] == "836121a0533e5664b21c7aacc5d22951f2b8b25b"
    assert manifest["alignment_max_size"] == 2
    assert (
        manifest["first_sha256"]
        == hashlib.sha256((FIXTURES / "alice-fr.txt").read_bytes()).hexdigest()
    )
    assert (
        manifest["second_sha256"]
        == hashlib.sha256((FIXTURES / "alice-en.txt").read_bytes()).hexdigest()
    )
    files = manifest["files"]
    assert isinstance(files, dict) and files
    for name, digest in files.items():
        assert isinstance(name, str) and isinstance(digest, str)
        assert hashlib.sha256((CACHE_FIXTURE / name).read_bytes()).hexdigest() == digest
    return frozenset(files)


def _assert_cache_hit(cache: Path, expected: frozenset[str]) -> None:
    assert frozenset(path.name for path in cache.glob("*.npy")) == expected


def _write_epub(path: Path, sentences: tuple[str, ...], language: str) -> None:
    paragraphs = "".join(f"<p>{sentence}</p>" for sentence in sentences)
    document = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Alice extract</title></head>'
        f"<body>{paragraphs}</body></html>"
    )
    package = (
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata>'
        f'<language xmlns="http://purl.org/dc/elements/1.1/">{language}</language>'
        '</metadata><manifest><item id="alice" href="alice.xhtml" '
        'media-type="application/xhtml+xml"/></manifest><spine><itemref idref="alice"/>'
        "</spine></package>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
        '<rootfiles><rootfile full-path="content.opf"/></rootfiles></container>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", package)
        archive.writestr("alice.xhtml", document)
        archive.writestr("book.css", "p { margin: 0; }")


@pytest.mark.journey
def test_alice_parallel_through_plain_text_module(
    alice_runtime: tuple[Path, Path, Path, frozenset[str]],
) -> None:
    worker_python, model_storage, cache, cached_files = alice_runtime
    with VecalignLabseAligner(
        worker_python,
        model_storage=model_storage,
        cache_dir=cache,
        alignment_max_size=2,
    ) as aligner:
        mapping = align_texts(
            "\n".join(FRENCH_ALICE),
            "\n".join(ENGLISH_ALICE),
            footnote_source=FootnoteSource.SECOND,
            aligner=aligner,
            unit="line",
        )

    assert mapping.schema_version == 1
    assert mapping.footnote_source == "second"
    assert mapping.aligner["name"] == "vecalign-labse"
    assert mapping.aligner["model"] == "sentence-transformers/LaBSE"
    assert mapping.aligner["model_revision"] == "836121a0533e5664b21c7aacc5d22951f2b8b25b"
    assert len(mapping.mappings) == 10
    assert tuple(entry.base_text[0] for entry in mapping.mappings) == FRENCH_ALICE
    assert tuple(entry.footnote_text[0] for entry in mapping.mappings) == ENGLISH_ALICE
    assert [entry.base_indices for entry in mapping.mappings] == [(index,) for index in range(10)]
    assert [entry.footnote_indices for entry in mapping.mappings] == [
        (index,) for index in range(10)
    ]
    assert tuple(entry.score for entry in mapping.mappings) == (
        0.7867,
        0.7255,
        0.7429,
        0.548,
        0.8692,
        0.7396,
        0.5852,
        0.6549,
        0.8606,
        0.7429,
    )
    _assert_cache_hit(cache, cached_files)


@pytest.mark.journey
def test_alice_parallel_through_epub_cli(
    tmp_path: Path,
    alice_runtime: tuple[Path, Path, Path, frozenset[str]],
) -> None:
    worker_python, model_storage, cache, cached_files = alice_runtime
    french = tmp_path / "alice-fr.epub"
    english = tmp_path / "alice-en.epub"
    output = tmp_path / "alice-bilingual.epub"
    evidence = tmp_path / "alice-alignment.json"
    _write_epub(french, FRENCH_ALICE, "fr")
    _write_epub(english, ENGLISH_ALICE, "en")
    french_before = french.read_bytes()
    english_before = english.read_bytes()

    main(
        [
            str(french),
            str(english),
            "--footnote-source",
            "second",
            "--output",
            str(output),
            "--alignment-output",
            str(evidence),
            "--progress-interval",
            "0",
            "--aligner-cache",
            str(cache),
            "--aligner-python",
            str(worker_python),
            "--model-storage",
            str(model_storage),
            "--alignment-max-size",
            "2",
        ],
    )

    assert french.read_bytes() == french_before
    assert english.read_bytes() == english_before
    artifact = json.loads(evidence.read_text())
    assert hashlib.sha256(french_before).hexdigest() == artifact["base_sha256"]
    assert artifact["schema_version"] == 3
    verification = artifact["verification"]
    expected_counts = {
        "base_only_link_count": 0,
        "base_unit_count": 10,
        "entry_count": 5,
        "footnote_only_link_count": 0,
        "footnote_unit_count": 10,
        "link_count": 10,
        "rendered_note_count": 10,
    }
    assert {key: verification[key] for key in expected_counts} == expected_counts
    assert verification["base_sha256"] == hashlib.sha256(french_before).hexdigest()
    assert verification["footnote_sha256"] == hashlib.sha256(english_before).hexdigest()
    assert verification["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert verification["modified_document_count"] == 1
    assert verification["unchanged_entry_count"] == 4

    with zipfile.ZipFile(output) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("book.css") == b"p { margin: 0; }"
        document = BeautifulSoup(archive.read("alice.xhtml"), "html.parser")

    markers = document.select("a.bf-noteref")
    notes = document.select('[epub\\:type="footnote"]')
    assert [marker.get_text(strip=True) for marker in markers] == [
        str(index) for index in range(1, 11)
    ]
    assert [note.get_text(" ", strip=True) for note in notes] == [
        f"{sentence} ↩" for sentence in ENGLISH_ALICE
    ]
    assert [marker.get("href") for marker in markers] == [
        f"#bf-note-{index}" for index in range(1, 11)
    ]
    backlinks = [note.select_one("a") for note in notes]
    assert all(backlink is not None for backlink in backlinks)
    assert [backlink.get("href") for backlink in backlinks if backlink is not None] == [
        f"#bf-ref-{index}" for index in range(1, 11)
    ]
    _assert_cache_hit(cache, cached_files)
