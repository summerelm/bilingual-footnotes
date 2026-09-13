from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from bilingual_text_align import AlignmentAlgorithm, AlignmentLink, TextUnit
from bilingual_text_align.resources import (
    MODEL_STORAGE_MARKER,
    mark_resource,
)
from epub_bilingual_footnotes import Artifact

from bilingual_text_align_ui import EpubBuildRequest, default_aligner_python, run_epub_build
from bilingual_text_align_ui.configuration import bundled_worker_command
from bilingual_text_align_ui.epub_window import suggested_output


class ScriptedAligner:
    @property
    def metadata(self) -> Mapping[str, object]:
        return {"name": "scripted"}

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        del first, second
        return (AlignmentLink((0,), (0,), 0.9),)


class ContextAligner(ScriptedAligner):
    def __enter__(self) -> ContextAligner:
        return self

    def __exit__(self, *args: object) -> None:
        del args


def request(tmp_path: Path, *, worker: Path | None = None) -> EpubBuildRequest:
    reading = tmp_path / "reading.epub"
    translation = tmp_path / "translation.epub"
    reading.write_bytes(b"reading")
    translation.write_bytes(b"translation")
    return EpubBuildRequest(
        reading,
        translation,
        tmp_path / "bilingual.epub",
        aligner_python=worker,
    )


def artifact() -> Artifact:
    return Artifact(3, "base", "translation", {"name": "scripted"}, (), (), (), ())


def fake_builder(
    reading: str | Path,
    translation: str | Path,
    output: str | Path,
    aligner: AlignmentAlgorithm,
    *,
    progress: Callable[[str], None] | None = None,
) -> Artifact:
    del reading, translation, output, aligner
    if progress is not None:
        progress("reading")
        progress("fine-alignment")
        progress("rendering")
        progress("verification")
    return artifact()


def test_run_epub_build_fixes_reading_and_translation_roles(tmp_path: Path) -> None:
    paths: list[Path] = []

    def capture_builder(
        reading: str | Path,
        translation: str | Path,
        output: str | Path,
        aligner: AlignmentAlgorithm,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> Artifact:
        del aligner
        paths.extend((Path(reading), Path(translation), Path(output)))
        return fake_builder(reading, translation, output, ScriptedAligner(), progress=progress)

    build_request = request(tmp_path)
    phases: list[str] = []
    result = run_epub_build(
        build_request, ScriptedAligner(), phases.append, builder=capture_builder
    )

    assert result.rendered_note_count == 0
    assert result.link_count == 0
    assert paths == [
        build_request.reading_epub,
        build_request.translation_epub,
        build_request.output_epub,
    ]
    assert phases == ["preparing", "reading", "fine-alignment", "rendering", "verification"]


def test_run_epub_build_constructs_worker_from_request(tmp_path: Path) -> None:
    created: list[tuple[Path, Path | None, Path | None]] = []

    def fake_backend(
        worker: Path,
        *,
        cache_dir: Path | None = None,
        model_storage: Path | None = None,
    ) -> ContextAligner:
        created.append((worker, cache_dir, model_storage))
        return ContextAligner()

    worker = tmp_path / "worker-python"
    cache = tmp_path / "cache"
    model = tmp_path / "model"
    mark_resource(model, MODEL_STORAGE_MARKER)
    (model / "weights").write_bytes(b"model")
    build_request = request(tmp_path, worker=worker)
    build_request = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        build_request.output_epub,
        aligner_python=worker,
        aligner_cache=cache,
        model_storage=model,
    )
    run_epub_build(build_request, builder=fake_builder, aligner_factory=fake_backend)
    assert created == [(worker, cache, model)]


def test_run_epub_build_uses_worker_embedded_in_frozen_app(tmp_path: Path) -> None:
    created: list[tuple[str, ...]] = []

    def fake_backend(
        _python: Path,
        *,
        cache_dir: Path | None = None,
        model_storage: Path | None = None,
        worker_command: tuple[str, ...],
    ) -> ContextAligner:
        del cache_dir, model_storage
        created.append(worker_command)
        return ContextAligner()

    build_request = request(tmp_path)
    model = tmp_path / "model"
    mark_resource(model, MODEL_STORAGE_MARKER)
    (model / "weights").write_bytes(b"model")
    build_request = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        build_request.output_epub,
        model_storage=model,
    )

    run_epub_build(
        build_request,
        builder=fake_builder,
        aligner_factory=fake_backend,
        worker_command_factory=lambda: ("/App/Contents/Resources/worker",),
    )

    assert created == [("/App/Contents/Resources/worker",)]


def test_run_epub_build_requires_installed_model_for_real_backend(tmp_path: Path) -> None:
    build_request = request(tmp_path)
    build_request = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        build_request.output_epub,
        model_storage=tmp_path / "missing-model",
    )

    with pytest.raises(RuntimeError, match="semantic model is not installed"):
        run_epub_build(build_request)


def test_run_epub_build_rejects_non_epub_and_existing_output(tmp_path: Path) -> None:
    invalid = request(tmp_path)
    invalid.reading_epub.rename(tmp_path / "reading.txt")
    invalid = EpubBuildRequest(
        tmp_path / "reading.txt", invalid.translation_epub, invalid.output_epub
    )
    with pytest.raises(ValueError, match="input must be an EPUB"):
        run_epub_build(invalid, ScriptedAligner())

    existing = tmp_path / "existing"
    existing.mkdir()
    build_request = request(existing)
    build_request.output_epub.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="output must be a new path"):
        run_epub_build(build_request, ScriptedAligner())
    assert build_request.output_epub.read_bytes() == b"keep"


def test_run_epub_build_validates_all_paths(tmp_path: Path) -> None:
    build_request = request(tmp_path)
    missing = EpubBuildRequest(
        tmp_path / "missing.epub",
        build_request.translation_epub,
        build_request.output_epub,
    )
    with pytest.raises(FileNotFoundError, match="input EPUB not found"):
        run_epub_build(missing, ScriptedAligner())

    wrong_output = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        tmp_path / "output.json",
    )
    with pytest.raises(ValueError, match="output must be an EPUB"):
        run_epub_build(wrong_output, ScriptedAligner())

    missing_parent = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        tmp_path / "missing" / "output.epub",
    )
    with pytest.raises(FileNotFoundError, match="output directory does not exist"):
        run_epub_build(missing_parent, ScriptedAligner())

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    invalid_parent = EpubBuildRequest(
        build_request.reading_epub,
        build_request.translation_epub,
        blocker / "output.epub",
    )
    with pytest.raises(NotADirectoryError, match="output parent is not a directory"):
        run_epub_build(invalid_parent, ScriptedAligner())

    with pytest.raises(PermissionError, match="output directory is not writable"):
        run_epub_build(build_request, ScriptedAligner(), writable=lambda _path: False)


def test_default_worker_path_respects_platform_and_environment() -> None:
    assert default_aligner_python("posix", {}) == Path(".venv-bilingual/bin/python")
    assert default_aligner_python("nt", {}) == Path(".venv-bilingual/Scripts/python.exe")
    assert default_aligner_python("posix", {"BILINGUAL_ALIGNER_PYTHON": "/custom/python"}) == Path(
        "/custom/python"
    )


def test_frozen_app_resolves_its_embedded_worker() -> None:
    executable = Path("/Applications/Bilingual Footnotes.app/Contents/MacOS/Bilingual Footnotes")
    assert bundled_worker_command(executable=executable, frozen=True) == (
        "/Applications/Bilingual Footnotes.app/Contents/Resources/semantic-worker/"
        "bilingual-align-worker",
    )
    assert bundled_worker_command(executable=executable, frozen=False) is None


def test_output_name_is_suggested_from_reading_book() -> None:
    assert suggested_output("/books/alice-fr.epub") == ("/books", "alice-fr-bilingual.epub")
    assert suggested_output("") == (None, "bilingual.epub")
