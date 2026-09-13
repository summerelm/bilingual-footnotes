from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bilingual_text_align import AlignmentAlgorithm, AlignmentPhase, align_files
from bilingual_text_align.cli import build_parser, main
from bilingual_text_align.model import AlignmentLink, TextUnit
from bilingual_text_align.plain_text import FootnoteSource, align_texts
from bilingual_text_align.segmentation import segment_text, sentence_spans
from bilingual_text_align.service import align_units


class ScriptedAligner:
    @property
    def metadata(self) -> Mapping[str, object]:
        return {"name": "scripted"}

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        del first, second
        return (AlignmentLink((0, 1), (0,), 0.9), AlignmentLink((2,), (1,), 0.8))


def test_plain_text_mapping_can_reverse_footnote_direction() -> None:
    algorithm: AlignmentAlgorithm = ScriptedAligner()
    mapping = align_texts(
        "Bonjour. Comment ça va ? Au revoir.",
        "Hello there. Goodbye.",
        footnote_source=FootnoteSource.FIRST,
        aligner=algorithm,
    )
    assert mapping.mappings[0].base_indices == (0,)
    assert mapping.mappings[0].footnote_indices == (0, 1)
    assert json.loads(mapping.to_json())["aligner"] == {"name": "scripted"}


def test_plain_text_mapping_second_direction() -> None:
    mapping = align_texts(
        "Bonjour. Comment ça va ? Au revoir.",
        "Hello there. Goodbye.",
        footnote_source=FootnoteSource.SECOND,
        aligner=ScriptedAligner(),
    )
    assert mapping.mappings[0].base_indices == (0, 1)
    assert mapping.mappings[0].footnote_indices == (0,)


def test_segmentation_preserves_dialogue_and_groups() -> None:
    units = segment_text("M. Dupont arrive. « Bonjour ! »\n\nDeuxième ligne.")
    assert [unit.text for unit in units] == [
        "M. Dupont arrive.",
        "« Bonjour ! »",
        "Deuxième ligne.",
    ]
    assert [unit.group for unit in units] == [0, 0, 1]
    assert sentence_spans("Sans ponctuation") == ((0, 16),)
    assert sentence_spans("Vraiment?! Ensuite.") == ((0, 10), (11, 19))
    assert sentence_spans("  Fin.  ") == ((2, 6),)


def test_line_mode_and_invalid_mode() -> None:
    assert [unit.text for unit in segment_text(" one\n\n two ", "line")] == ["one", "two"]
    with pytest.raises(ValueError, match="Unsupported"):
        segment_text("text", "paragraph")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([], [TextUnit(0, "x", 0)], "first text"),
        ([TextUnit(1, "x", 0)], [TextUnit(0, "y", 0)], "indices"),
        ([TextUnit(0, " ", 0)], [TextUnit(0, "y", 0)], "blank"),
    ],
)
def test_unit_validation(first: list[TextUnit], second: list[TextUnit], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        align_units(first, second, ScriptedAligner())


class InvalidAligner(ScriptedAligner):
    def __init__(self, links: Sequence[AlignmentLink]) -> None:
        self.links = links

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        del first, second
        return self.links


@pytest.mark.parametrize(
    "links",
    [
        (),
        (AlignmentLink((), (), None),),
        (AlignmentLink((1,), (0,), None),),
        (AlignmentLink((0,), (), None),),
        (AlignmentLink((), (0,), None),),
    ],
)
def test_link_validation(links: Sequence[AlignmentLink]) -> None:
    with pytest.raises(ValueError):
        align_units([TextUnit(0, "x", 0)], [TextUnit(0, "y", 0)], InvalidAligner(links))


def test_cli_writes_mapping_with_injected_aligner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    output = tmp_path / "mapping.json"
    first.write_text("One. Two. Three.", encoding="utf-8")
    second.write_text("Un deux. Trois.", encoding="utf-8")
    main(
        [str(first), str(second), "--footnote-source", "second", "--output", str(output)],
        ScriptedAligner(),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["footnote_source"] == "second"
    console = capsys.readouterr().out
    assert "[alignment] Aligning corresponding text" in console
    assert f"Aligned 2 links; created {output}" in console


def test_file_application_writes_new_output(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    output = tmp_path / "mapping.json"
    first.write_text("One. Two. Three.", encoding="utf-8")
    second.write_text("Un deux. Trois.", encoding="utf-8")

    phases: list[AlignmentPhase] = []
    mapping = align_files(
        first,
        second,
        output,
        footnote_source=FootnoteSource.SECOND,
        aligner=ScriptedAligner(),
        report_progress=phases.append,
    )

    assert len(mapping.mappings) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["footnote_source"] == "second"
    assert phases == [
        AlignmentPhase.READING_INPUTS,
        AlignmentPhase.ALIGNING,
        AlignmentPhase.WRITING_OUTPUT,
    ]


def test_file_application_does_not_race_an_existing_output(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    output = tmp_path / "mapping.json"
    first.write_text("One. Two. Three.", encoding="utf-8")
    second.write_text("Un deux. Trois.", encoding="utf-8")
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output must be a new path"):
        align_files(
            first,
            second,
            output,
            footnote_source=FootnoteSource.SECOND,
            aligner=ScriptedAligner(),
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_exposes_worker_startup_timeout() -> None:
    args = build_parser().parse_args(
        ["one.txt", "two.txt", "--footnote-source", "second", "-o", "out.json"]
    )
    assert args.worker_startup_timeout == 600.0
    assert args.progress_interval == 60.0
    assert args.aligner_cache.name == "embeddings"
    assert (
        build_parser()
        .parse_args(
            [
                "one.txt",
                "two.txt",
                "--footnote-source",
                "second",
                "-o",
                "out.json",
                "--no-aligner-cache",
            ]
        )
        .aligner_cache
        is None
    )


def test_cli_help_explains_required_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert exc.value.code == 0
    assert "which input supplies translation text" in help_text
    assert "FIRST_TEXT the base" in help_text
    assert "write the alignment mapping" in help_text
    assert "advanced alignment tuning" in help_text
    assert "before the first run" in help_text
    assert "bilingual-align-setup" in help_text
    assert "Source files are never changed" in help_text
    assert "elapsed-time updates" in help_text


def test_cli_fails_fast_when_semantic_model_is_missing(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("text", encoding="utf-8")
    output = tmp_path / "out.json"
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


def test_cli_rejects_existing_or_input_output_paths(tmp_path: Path) -> None:
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("One.", encoding="utf-8")
    second.write_text("Un.", encoding="utf-8")
    with pytest.raises(SystemExit, match="output must be a new path"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(first)],
            ScriptedAligner(),
        )
    output = tmp_path / "mapping.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="output must be a new path"):
        main(
            [str(first), str(second), "--footnote-source", "second", "-o", str(output)],
            ScriptedAligner(),
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_reports_missing_input_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "mapping.json"
    with pytest.raises(SystemExit, match="Error:"):
        main(
            [
                str(tmp_path / "missing.txt"),
                str(tmp_path / "second.txt"),
                "--footnote-source",
                "second",
                "-o",
                str(output),
            ],
            ScriptedAligner(),
        )
    assert not output.exists()


def test_cli_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == "bilingual-align 0.1.0\n"
