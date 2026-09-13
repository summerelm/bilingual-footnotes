"""Operating-system paths for application-managed resources."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

EMBEDDING_CACHE_MARKER = ".bilingual-footnotes-embeddings"
MODEL_STORAGE_MARKER = ".bilingual-footnotes-model"


def default_storage_root(
    *,
    platform: str = sys.platform,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    user_home = Path.home() if home is None else home
    if configured := values.get("BILINGUAL_FOOTNOTES_STORAGE"):
        return Path(configured).expanduser()
    if platform == "darwin":
        return user_home / "Library" / "Caches" / "Bilingual Footnotes"
    if platform == "win32":
        local = values.get("LOCALAPPDATA")
        return (Path(local) if local else user_home / "AppData" / "Local") / "Bilingual Footnotes"
    xdg = values.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else user_home / ".cache") / "bilingual-footnotes"


def default_model_storage(
    *,
    platform: str = sys.platform,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    return default_storage_root(platform=platform, environment=environment, home=home) / "model"


def default_embedding_cache(
    *,
    platform: str = sys.platform,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    return (
        default_storage_root(platform=platform, environment=environment, home=home) / "embeddings"
    )


def default_worker_python(platform: str = os.name) -> Path:
    relative = Path("Scripts/python.exe") if platform == "nt" else Path("bin/python")
    return Path(".venv-bilingual") / relative
