"""Stable alignment-artifact serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .model import Artifact
from .output_safety import publish_new_file


def write_artifact(artifact: Artifact, path: str | Path) -> None:
    output = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        publish_new_file(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
