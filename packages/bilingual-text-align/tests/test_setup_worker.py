from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from bilingual_text_align import setup_worker


def test_setup_parser_uses_public_defaults() -> None:
    args = setup_worker.build_parser().parse_args([])
    assert args.python == "python3.12"
    assert args.venv == Path(".venv-bilingual")
    assert args.model_storage.name == "model"
    assert setup_worker._environment_python(Path("worker"), "posix") == Path("worker/bin/python")
    assert setup_worker._environment_python(Path("worker"), "nt") == Path(
        "worker/Scripts/python.exe"
    )


def test_setup_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        setup_worker.build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == "bilingual-align-setup 0.1.0\n"


def test_setup_help_explains_environment_destination(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        setup_worker.build_parser().parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert exc.value.code == 0
    assert "create the worker environment here" in help_text
    assert "Python 3.12 executable used to create the worker" in help_text
    assert "network access" in help_text
    assert "several gigabytes" in help_text
    assert "--aligner-python" in help_text


def test_setup_requires_uv() -> None:
    with pytest.raises(SystemExit, match="uv is required"):
        setup_worker.main([], locator=lambda _name: None)


def test_setup_reports_missing_python() -> None:
    with pytest.raises(SystemExit, match="Python 3.12 executable not found.*--python"):
        setup_worker.main(
            ["--python", "missing-python"],
            locator=lambda name: "/tools/uv" if name == "uv" else None,
        )


def test_setup_requires_tested_python() -> None:
    with pytest.raises(SystemExit, match="requires Python 3.12; got 3.13"):
        setup_worker.main(
            [],
            locator=lambda name: f"/bin/{name}",
            runner=lambda _command, *, capture_output=False: subprocess.CompletedProcess(
                [], 0, "3.13\n" if capture_output else ""
            ),
        )


def test_setup_creates_and_verifies_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commands: list[Sequence[str]] = []

    def fake_run(
        command: Sequence[str], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "3.12\n" if capture_output else "")

    prepared: list[tuple[Path, Path]] = []

    def prepare(
        python: Path,
        _worker: Path,
        _model: str,
        _revision: str,
        destination: Path,
    ) -> None:
        prepared.append((python, destination))

    environment = tmp_path / "semantic"
    model_storage = tmp_path / "model"

    setup_worker.main(
        [
            "--python",
            "python3.12",
            "--venv",
            str(environment),
            "--model-storage",
            str(model_storage),
        ],
        locator=lambda name: f"/tools/{name}",
        runner=fake_run,
        model_preparer=prepare,
    )

    python = environment / "bin/python"
    assert commands[1] == [
        "/tools/uv",
        "venv",
        "--python",
        "/tools/python3.12",
        str(environment),
    ]
    assert commands[2][:5] == [
        "/tools/uv",
        "pip",
        "install",
        "--python",
        str(python),
    ]
    assert setup_worker.PACKAGES[-1].endswith(setup_worker.VECALIGN_COMMIT)
    assert commands[3] == [
        str(python),
        "-c",
        "import numpy, torch, vecalign, sentence_transformers",
    ]
    assert prepared == [(python, model_storage)]
    assert f"Vecalign worker created at {python}" in capsys.readouterr().out


def test_run_executes_a_command() -> None:
    result = setup_worker._run([sys.executable, "-c", "print('3.12')"], capture_output=True)
    assert result.stdout.strip() == "3.12"


@pytest.mark.parametrize(
    ("failed_call", "message"),
    [
        (2, "could not install the semantic worker dependencies"),
        (3, "installed but its packages could not be imported"),
    ],
)
def test_setup_turns_subprocess_failures_into_actionable_errors(
    failed_call: int, message: str, tmp_path: Path
) -> None:
    calls = 0

    def fake_run(
        command: Sequence[str], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        current = calls
        calls += 1
        if current == failed_call:
            raise subprocess.CalledProcessError(1, command, stderr="dependency detail")
        return subprocess.CompletedProcess(command, 0, "3.12\n" if capture_output else "")

    with pytest.raises(SystemExit, match=message):
        setup_worker.main(
            ["--venv", str(tmp_path / "worker")],
            locator=lambda name: f"/tools/{name}",
            runner=fake_run,
        )
