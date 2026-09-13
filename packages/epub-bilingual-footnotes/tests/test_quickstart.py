from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from epub_bilingual_footnotes.extraction import read_book


def test_quickstart_generator_creates_readable_inputs(tmp_path: Path) -> None:
    repository = Path(__file__).parents[3]
    script = repository / "examples" / "quickstart" / "create_samples.py"
    output = tmp_path / "samples"

    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Created four CC0 sample inputs" in result.stdout
    assert (output / "reading-fr.txt").read_text(encoding="utf-8").startswith("Le soleil se lève.")
    assert (output / "notes-en.txt").read_text(encoding="utf-8").startswith("The sun rises.")
    for name in ("reading-fr.epub", "notes-en.epub"):
        path = output / name
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist()[0] == "mimetype"
            assert archive.read("mimetype") == b"application/epub+zip"
        assert len(read_book(path).units) == 3
