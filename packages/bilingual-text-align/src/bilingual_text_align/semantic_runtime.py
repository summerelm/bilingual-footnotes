"""Preconditions for starting the optional semantic alignment runtime."""

from __future__ import annotations

from pathlib import Path

from .model_management import semantic_model_installed


def require_semantic_runtime(python: str | Path, model_storage: str | Path) -> None:
    storage = Path(model_storage).expanduser()
    if not semantic_model_installed(storage):
        raise RuntimeError(
            f"semantic model is not installed at {storage}; run bilingual-align-setup first"
        )
    executable = Path(python).expanduser()
    if not executable.is_file():
        raise RuntimeError(
            f"alignment worker Python not found at {executable}; run bilingual-align-setup first"
        )
