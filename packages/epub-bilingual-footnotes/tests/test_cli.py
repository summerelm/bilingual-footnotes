import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

import pytest
from bilingual_text_align import AlignmentLink, ContainmentMatch, TextUnit
from test_pipeline import make_epub

from epub_bilingual_footnotes.cli import build_parser, main


class ScriptedAligner:
    @property
    def metadata(self) -> Mapping[str, object]:
        return {"name": "scripted"}

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        return tuple(
            AlignmentLink((index,), (index,), 1.0) for index in range(min(len(first), len(second)))
        )


def test_cli_exposes_one_two_epub_mode() -> None:
    args = build_parser().parse_args(
        ["one.epub", "two.epub", "--footnote-source", "first", "-o", "out.epub"]
    )
    assert args.footnote_source == "first"
    assert args.aligner_model == "sentence-transformers/LaBSE"
    assert args.aligner_model_revision == "836121a0533e5664b21c7aacc5d22951f2b8b25b"
    assert args.alignment_max_size == 8
    assert args.embedding_batch_size == 16
    assert args.worker_startup_timeout == 600.0
    assert args.progress_interval == 60.0
    assert args.model_storage.name == "model"
    assert args.aligner_cache.name == "embeddings"


def test_cli_help_explains_reading_and_note_roles(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert exc.value.code == 0
    assert "which EPUB supplies translation notes" in help_text
    assert "FIRST_EPUB the reading book" in help_text
    assert "write the completed reading book" in help_text
    assert "alignment and verification details" in help_text
    assert "before the first run" in help_text
    assert "Both source books remain unchanged" in help_text
    assert "Every output path must be new" in help_text
    assert "elapsed-time updates" in help_text


def test_epub_cli_can_reverse_roles_and_write_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = tmp_path / "one.epub", tmp_path / "two.epub"
    output, artifact = tmp_path / "out.epub", tmp_path / "alignment.json"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])
    main(
        [
            str(first),
            str(second),
            "--footnote-source",
            "first",
            "-o",
            str(output),
            "--alignment-output",
            str(artifact),
            "--aligner-cache",
            str(tmp_path / "cache"),
        ],
        ScriptedAligner(),
    )
    assert output.exists()
    data = json.loads(artifact.read_text())
    assert data["schema_version"] == 3
    assert data["verification"]["rendered_note_count"] == 1
    assert data["run"]["aligner_cache"] == str((tmp_path / "cache").resolve())
    assert data["run"]["pipeline_seconds"] >= 0
    assert set(data["run"]["phase_seconds"]) == {
        "reading",
        "fine-alignment",
        "rendering",
        "verification",
    }
    console = capsys.readouterr().out
    assert "[reading] Reading both EPUBs" in console
    assert "[fine-alignment] Aligning corresponding text" in console
    assert "[rendering] Rendering translation footnotes" in console
    assert "[verification] Verifying the rendered EPUB" in console
    assert "[complete] Build completed and verified" in console
    assert f"Created {output}: 1 base units, 1 footnote units, 1 links, 1 rendered notes" in console


def test_epub_cli_reports_containment_phase(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class ContainmentAligner(ScriptedAligner):
        def locate(
            self, query_texts: Sequence[str], corpus_texts: Sequence[str], config: object = None
        ) -> ContainmentMatch:
            del query_texts, corpus_texts, config
            return ContainmentMatch(0, 1, ((0, 0),), 1.0, 1.0, 1.0, 1.0, None, None, True, ())

    first, second, output = tmp_path / "one.epub", tmp_path / "two.epub", tmp_path / "out.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])

    main(
        [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
        ContainmentAligner(),
    )

    assert "[containment] Locating corresponding global text" in capsys.readouterr().out


def test_epub_cli_without_optional_artifact(tmp_path: Path) -> None:
    first, second, output = tmp_path / "one.epub", tmp_path / "two.epub", tmp_path / "out.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])
    main(
        [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
        ScriptedAligner(),
    )
    assert output.exists()


def test_epub_cli_fails_fast_when_semantic_model_is_missing(tmp_path: Path) -> None:
    first, second, output = tmp_path / "one.epub", tmp_path / "two.epub", tmp_path / "out.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])

    with pytest.raises(SystemExit, match="semantic model is not installed.*align-setup"):
        main(
            [
                str(first),
                str(second),
                "--footnote-source",
                "second",
                "--output",
                str(output),
                "--model-storage",
                str(tmp_path / "missing-model"),
                "--aligner-python",
                str(tmp_path / "missing-python"),
            ]
        )

    assert not output.exists()


def test_cli_forwards_semantic_backend_configuration_and_closes_it(tmp_path: Path) -> None:
    first, second, output = tmp_path / "one.epub", tmp_path / "two.epub", tmp_path / "out.epub"
    cache = tmp_path / "cache"
    model_storage = tmp_path / "model"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])

    class CapturingAligner(ScriptedAligner):
        created: ClassVar[list[object]] = []

        def __init__(self, python: str, **options: object) -> None:
            self.python = python
            self.options = options
            self.closed = False
            self.created.append(self)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            self.closed = True

    main(
        [
            str(first),
            str(second),
            "--footnote-source",
            "second",
            "--output",
            str(output),
            "--aligner-python",
            "/worker/python",
            "--aligner-cache",
            str(cache),
            "--model-storage",
            str(model_storage),
            "--aligner-model",
            "custom/model",
            "--aligner-model-revision",
            "revision-123",
            "--alignment-max-size",
            "6",
            "--embedding-batch-size",
            "12",
        ],
        aligner_factory=CapturingAligner,
    )

    assert len(CapturingAligner.created) == 1
    backend = CapturingAligner.created[0]
    assert isinstance(backend, CapturingAligner)
    assert backend.python == "/worker/python"
    assert backend.options == {
        "model": "custom/model",
        "model_revision": "revision-123",
        "model_storage": model_storage,
        "cache_dir": cache,
        "alignment_max_size": 6,
        "batch_size": 12,
        "startup_timeout": 600.0,
    }
    assert backend.closed
    assert output.exists()


def test_cli_rejects_existing_and_conflicting_outputs(tmp_path: Path) -> None:
    first, second = tmp_path / "one.epub", tmp_path / "two.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])
    with pytest.raises(SystemExit, match="output must be a new path"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(first)],
            ScriptedAligner(),
        )
    output = tmp_path / "out.epub"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="output must be a new path"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
            ScriptedAligner(),
        )
    assert output.read_text(encoding="utf-8") == "keep"
    artifact = tmp_path / "alignment.json"
    artifact.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="output must be a new path"):
        main(
            [
                str(first),
                str(second),
                "--footnote-source",
                "second",
                "-o",
                str(tmp_path / "unused.epub"),
                "--alignment-output",
                str(artifact),
            ],
            ScriptedAligner(),
        )
    assert artifact.read_text(encoding="utf-8") == "keep"
    with pytest.raises(SystemExit, match="output paths must be distinct"):
        main(
            [
                str(first),
                str(second),
                "--footnote-source",
                "second",
                "-o",
                str(tmp_path / "new.epub"),
                "--alignment-output",
                str(tmp_path / "new.epub"),
            ],
            ScriptedAligner(),
        )


def test_cli_reports_missing_input_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out.epub"
    with pytest.raises(SystemExit, match="Error: input EPUB not found"):
        main(
            [
                str(tmp_path / "missing.epub"),
                str(tmp_path / "other.epub"),
                "--footnote-source",
                "second",
                "-o",
                str(output),
            ],
            ScriptedAligner(),
        )
    assert not output.exists()


def test_cli_reports_malformed_epub_without_creating_output(tmp_path: Path) -> None:
    first, second = tmp_path / "broken.epub", tmp_path / "other.epub"
    output = tmp_path / "out.epub"
    first.write_text("not an epub", encoding="utf-8")
    make_epub(second, [(1, "Un.")])

    with pytest.raises(SystemExit, match="Error: Invalid EPUB"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
            ScriptedAligner(),
        )

    assert not output.exists()


def test_cli_explains_rejected_containment_without_creating_output(tmp_path: Path) -> None:
    class RejectingAligner(ScriptedAligner):
        def locate(
            self, query_texts: Sequence[str], corpus_texts: Sequence[str], config: object = None
        ) -> ContainmentMatch:
            del query_texts, corpus_texts, config
            return ContainmentMatch(
                0,
                1,
                ((0, 0),),
                0.1,
                0.1,
                0.1,
                0.0,
                None,
                None,
                False,
                ("similarity is too low",),
            )

    first, second = tmp_path / "one.epub", tmp_path / "two.epub"
    output = tmp_path / "out.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un."), (2, "Deux.")])

    with pytest.raises(SystemExit, match="Check that the two EPUBs are corresponding"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
            RejectingAligner(),
        )

    assert not output.exists()


def test_cli_rejects_invalid_artifact_directory_before_building(tmp_path: Path) -> None:
    first, second = tmp_path / "one.epub", tmp_path / "two.epub"
    output = tmp_path / "out.epub"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])

    with pytest.raises(SystemExit, match="output directory does not exist"):
        main(
            [
                str(first),
                str(second),
                "--footnote-source",
                "second",
                "--output",
                str(output),
                "--alignment-output",
                str(tmp_path / "missing" / "alignment.json"),
            ],
            ScriptedAligner(),
        )

    assert not output.exists()


def test_cli_rejects_output_parent_that_is_not_a_directory(tmp_path: Path) -> None:
    first, second = tmp_path / "one.epub", tmp_path / "two.epub"
    parent = tmp_path / "ordinary-file"
    make_epub(first, [(1, "One.")])
    make_epub(second, [(1, "Un.")])
    parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SystemExit, match="output parent is not a directory"):
        main(
            [
                str(first),
                str(second),
                "--footnote-source",
                "second",
                "--output",
                str(parent / "out.epub"),
            ],
            ScriptedAligner(),
        )


def test_cli_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == "bilingual-footnotes 0.1.0\n"
