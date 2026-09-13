"""Installation and readiness checks for the semantic model."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .resource_paths import MODEL_STORAGE_MARKER
from .resource_storage import mark_resource


def semantic_model_installed(path: Path) -> bool:
    if not (path / MODEL_STORAGE_MARKER).is_file():
        return False
    return any(
        candidate.is_file()
        and candidate.name != MODEL_STORAGE_MARKER
        and not candidate.name.startswith(".active-")
        for candidate in path.rglob("*")
    )


def prepare_model(
    python: Path,
    worker: Path,
    model: str,
    revision: str,
    destination: Path,
    *,
    offline: bool = False,
    runner: object = subprocess.run,
) -> None:
    prepare_model_command(
        [str(python), str(worker)],
        model,
        revision,
        destination,
        offline=offline,
        runner=runner,
    )


def prepare_model_command(
    worker_command: Sequence[str],
    model: str,
    revision: str,
    destination: Path,
    *,
    offline: bool = False,
    runner: object = subprocess.run,
) -> None:
    """Install or verify a model using a directly executable worker command."""
    command = [
        *worker_command,
        "--model",
        model,
        "--model-revision",
        revision,
        "--model-storage",
        str(destination),
        "--prepare-model",
    ]
    environment = dict(os.environ)
    if offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        if not callable(runner):
            raise TypeError("runner must be callable")
        runner(command, check=True, env=environment)
    except FileNotFoundError as exc:
        raise RuntimeError(f"alignment worker Python not found: {worker_command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        action = "verify" if offline else "install"
        raise RuntimeError(f"could not {action} the semantic model") from exc
    mark_resource(destination, MODEL_STORAGE_MARKER)
