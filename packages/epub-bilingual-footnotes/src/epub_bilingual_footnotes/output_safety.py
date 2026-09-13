"""Race-safe publication of completed output files."""

from __future__ import annotations

import os
from pathlib import Path


def publish_new_file(temporary: Path, output: Path) -> None:
    """Atomically publish a same-filesystem temporary without overwriting."""
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(f"Output must be a new path: {output}") from exc
    temporary.unlink()
