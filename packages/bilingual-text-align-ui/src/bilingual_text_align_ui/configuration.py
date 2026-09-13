"""Desktop configuration resolved independently of either use case."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def default_aligner_python(
    platform: str = os.name, environment: Mapping[str, str] | None = None
) -> Path:
    """Return the configured worker, with the setup command's platform default."""

    configured = (os.environ if environment is None else environment).get(
        "BILINGUAL_ALIGNER_PYTHON"
    )
    if configured:
        return Path(configured)
    relative = Path("Scripts/python.exe") if platform == "nt" else Path("bin/python")
    return Path(".venv-bilingual") / relative


def bundled_worker_command(
    *,
    executable: str | Path | None = None,
    frozen: bool | None = None,
    platform: str | None = None,
) -> tuple[str, ...] | None:
    """Return the worker embedded in a frozen desktop application, if present."""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not is_frozen:
        return None
    application = Path(sys.executable if executable is None else executable).resolve()
    current_platform = sys.platform if platform is None else platform
    if current_platform == "darwin":
        worker = application.parents[1] / "Resources" / "semantic-worker" / "bilingual-align-worker"
    else:
        suffix = ".exe" if current_platform == "win32" else ""
        worker = application.parent / "semantic-worker" / f"bilingual-align-worker{suffix}"
    return (str(worker),)
