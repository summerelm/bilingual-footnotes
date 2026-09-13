# bilingual-text-align-ui

A local desktop app for
[`epub-bilingual-footnotes`](../epub-bilingual-footnotes). It accepts the EPUB
you want to read, a corresponding translation EPUB, and a new EPUB output
path. Processing remains local and the UI never overwrites an input or existing
output.

## Run the macOS app

Download the macOS archive from the release, move **Bilingual Footnotes.app**
to Applications, and open it. No CLI or separate Python installation is
needed. Open **Model & storage…** and install the model on first use, then
choose the reading EPUB, translation EPUB, and a new output filename.

## Run from Python

Install the commands, prepare the semantic worker, and launch the UI:

```console
uv tool install bilingual-text-align
uv tool install bilingual-text-align-ui
bilingual-align-setup --python python3.12 --venv .venv-bilingual
bilingual-align-ui
```

The default worker path is `.venv-bilingual/bin/python` on macOS and Linux and
`.venv-bilingual/Scripts/python.exe` on Windows. Set
`BILINGUAL_ALIGNER_PYTHON` to use an environment elsewhere.

Open **Model & storage…** to see the required semantic model and optional processing
cache separately. The model can be installed or repaired, verified offline,
and removed. The cache can be cleared without affecting the model or books.
It stores book-derived embeddings, grows as needed, and has no automatic size
limit. Storage actions and EPUB generation cannot run over each other.

Tkinter is included by many Python distributions. On Linux, install the
distribution's Tk package (often `python3-tk`) if the window cannot start.

The reading EPUB supplies the book and the translation EPUB supplies popup
footnotes. The interface uses the website's paper, forest, sage, and vermilion
palette and automatically follows the operating system's light or dark
appearance at launch. The status line reports EPUB reading, alignment,
rendering, and verification with elapsed time. The progress bar animates while
work is in progress and fills when the EPUB has completed successfully.
