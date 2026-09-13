from __future__ import annotations

from pathlib import Path

import pytest

from bilingual_text_align import resources_cli
from bilingual_text_align.resources import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    mark_resource,
)


def test_resource_help_explains_commands_and_safety(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        resources_cli.build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert exc.value.code == 0
    assert "model is required" in help_text
    assert "clearing it never removes the model or source books" in help_text
    assert "model verify" in help_text
    assert "cache prune --max-size-gb 5" in help_text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["status", "--help"], "Show model readiness"),
        (["model", "--help"], "Manage the required semantic model"),
        (["model", "install", "--help"], "Network access is required"),
        (["model", "verify", "--help"], "without using the network"),
        (["model", "remove", "--help"], "source books are preserved"),
        (["cache", "--help"], "disposable embeddings"),
        (["cache", "clear", "--help"], "recreated when needed"),
        (["cache", "prune", "--help"], "target maximum cache size"),
    ],
)
def test_every_resource_command_has_specific_help(
    arguments: list[str], expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        resources_cli.build_parser().parse_args(arguments)
    assert exc.value.code == 0
    assert expected in " ".join(capsys.readouterr().out.split())


def test_status_distinguishes_model_and_processing_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    mark_resource(model, MODEL_STORAGE_MARKER)
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    (model / "weights").write_bytes(b"model")
    (cache / "book.npy").write_bytes(b"cache")

    resources_cli.main(["status", "--model-storage", str(model), "--cache-dir", str(cache)])

    output = capsys.readouterr().out
    assert "Semantic model: installed" in output
    assert "Processing cache:" in output


def test_model_commands_delegate_without_conflating_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Path, Path, bool]] = []

    def prepare(
        python: Path,
        worker: Path,
        _model: str,
        _revision: str,
        destination: Path,
        *,
        offline: bool = False,
    ) -> None:
        calls.append((python, destination, offline))
        assert worker.name == "worker.py"

    model = tmp_path / "model"
    python = tmp_path / "python"
    resources_cli.main(
        ["model", "install", "--model-storage", str(model), "--aligner-python", str(python)],
        model_preparer=prepare,
    )
    resources_cli.main(
        ["model", "verify", "--model-storage", str(model), "--aligner-python", str(python)],
        model_preparer=prepare,
    )

    assert calls == [(python, model, False), (python, model, True)]
    assert "Installed semantic model" in capsys.readouterr().out


def test_model_remove_and_cache_clear_touch_only_selected_resource(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    mark_resource(model, MODEL_STORAGE_MARKER)
    mark_resource(cache, EMBEDDING_CACHE_MARKER)
    (model / "weights").write_bytes(b"model")
    (cache / "book.npy").write_bytes(b"cache")

    resources_cli.main(["cache", "clear", "--cache-dir", str(cache)])
    assert model.exists()
    assert not cache.exists()
    resources_cli.main(["model", "remove", "--model-storage", str(model)])
    assert not model.exists()
    assert "Cleared processing cache" in capsys.readouterr().out


def test_cache_prune_validates_limit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="must not be negative"):
        resources_cli.main(
            ["cache", "prune", "--cache-dir", str(tmp_path / "cache"), "--max-size-gb", "-1"]
        )
