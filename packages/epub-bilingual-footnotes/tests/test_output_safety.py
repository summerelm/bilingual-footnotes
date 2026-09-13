from __future__ import annotations

from pathlib import Path

import pytest

from epub_bilingual_footnotes.output_safety import publish_new_file


def test_publish_new_file_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary"
    output = tmp_path / "output"
    temporary.write_bytes(b"complete")

    publish_new_file(temporary, output)

    assert output.read_bytes() == b"complete"
    assert not temporary.exists()

    competing = tmp_path / "competing"
    competing.write_bytes(b"new")
    with pytest.raises(FileExistsError, match="Output must be a new path"):
        publish_new_file(competing, output)
    assert output.read_bytes() == b"complete"
    assert competing.read_bytes() == b"new"
