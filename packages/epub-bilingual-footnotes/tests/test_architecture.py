from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def test_cross_package_contract_rejects_reverse_dependency(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    copied_sources = tmp_path / "src"
    backend = copied_sources / "bilingual_text_align"
    application = copied_sources / "epub_bilingual_footnotes"
    desktop_ui = copied_sources / "bilingual_text_align_ui"
    shutil.copytree(
        repository / "packages/bilingual-text-align/src/bilingual_text_align",
        backend,
    )
    shutil.copytree(
        repository / "packages/epub-bilingual-footnotes/src/epub_bilingual_footnotes",
        application,
    )
    shutil.copytree(
        repository / "packages/bilingual-text-align-ui/src/bilingual_text_align_ui",
        desktop_ui,
    )
    model = backend / "model.py"
    model.write_text(
        model.read_text(encoding="utf-8")
        + "\nfrom epub_bilingual_footnotes import build_epub  # deliberate violation\n",
        encoding="utf-8",
    )

    lint_imports = shutil.which("lint-imports")
    assert lint_imports is not None
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(copied_sources), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            lint_imports,
            "--config",
            str(repository / ".importlinter"),
            "--contract",
            "package-direction",
            "--no-cache",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Backend never imports EPUB application BROKEN" in output
