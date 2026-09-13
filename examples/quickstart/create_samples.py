"""Create CC0 text and EPUB inputs for the quickstart guide."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

SAMPLES = {
    "reading-fr": (
        "fr",
        "Lecture exemple",
        (
            "Le soleil se lève.",
            "Une voyageuse ouvre la fenêtre.",
            "Elle sourit et commence sa journée.",
        ),
    ),
    "notes-en": (
        "en",
        "Sample reading",
        (
            "The sun rises.",
            "A traveller opens the window.",
            "She smiles and begins her day.",
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("quickstart-output"))
    return parser


def main(argv: list[str] | None = None) -> None:
    output_dir = build_parser().parse_args(argv).output_dir
    paths = [output_dir / f"{name}.{suffix}" for name in SAMPLES for suffix in ("txt", "epub")]
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"Choose a new output directory; sample output already exists: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (language, title, paragraphs) in SAMPLES.items():
        (output_dir / f"{name}.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        _write_epub(output_dir / f"{name}.epub", language, title, paragraphs)
    print(f"Created four CC0 sample inputs in {output_dir}")


def _write_epub(path: Path, language: str, title: str, paragraphs: tuple[str, ...]) -> None:
    content = "\n".join(f"    <p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
    chapter = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}">
  <head><title>{html.escape(title)}</title></head>
  <body>
{content}
  </body>
</html>
"""
    package = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="book-id" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">quickstart-{language}</dc:identifier>
    <dc:title>{html.escape(title)}</dc:title>
    <dc:language>{language}</dc:language>
  </metadata>
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="chapter"/></spine>
</package>
"""
    container = """<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    with ZipFile(path, "x") as archive:
        mimetype = ZipInfo("mimetype")
        mimetype.compress_type = ZIP_STORED
        archive.writestr(mimetype, "application/epub+zip")
        archive.writestr("META-INF/container.xml", container, compress_type=ZIP_DEFLATED)
        archive.writestr("OEBPS/content.opf", package, compress_type=ZIP_DEFLATED)
        archive.writestr("OEBPS/chapter.xhtml", chapter, compress_type=ZIP_DEFLATED)


if __name__ == "__main__":
    main()
