from __future__ import annotations

import pytest

from bilingual_text_align_ui.tk_app import build_parser


def test_desktop_cli_help_does_not_open_the_window(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert exc.value.code == 0
    assert "Open the Bilingual Footnotes desktop app" in help_text
    assert "processing stays local" in help_text
    assert "includes its semantic worker" in help_text
    assert "Model & storage" in help_text


def test_desktop_cli_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == "bilingual-align-ui 0.1.0\n"
