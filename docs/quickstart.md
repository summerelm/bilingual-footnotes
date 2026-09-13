# Five-minute quickstart

Use the repository's synthetic CC0 samples to confirm the installed commands,
alignment environment, output paths, and EPUB reader behavior.

The sample-data dedication is recorded in
[`examples/quickstart/CC0.md`](../examples/quickstart/CC0.md).

Setup downloads the alignment dependencies and semantic model and can take
longer than five minutes. The workflow below takes about five minutes afterward.

## 1. Install and prepare

```console
uv tool install bilingual-text-align
uv tool install epub-bilingual-footnotes
bilingual-align-setup --python python3.12 --venv .venv-bilingual
```

Successful setup ends with:

```text
Vecalign worker created at .../.venv-bilingual/bin/python
Semantic model installed at .../model
```

## 2. Create the sample inputs

From the repository root:

```console
python3 examples/quickstart/create_samples.py --output-dir quickstart-output
```

This creates corresponding French and English text files and EPUBs:

```text
quickstart-output/
├── reading-fr.txt
├── notes-en.txt
├── reading-fr.epub
└── notes-en.epub
```

## 3. Align the text files

```console
bilingual-align quickstart-output/reading-fr.txt quickstart-output/notes-en.txt \
  --footnote-source second \
  --output quickstart-output/alignment.json \
  --aligner-python .venv-bilingual/bin/python
```

Success creates `quickstart-output/alignment.json` and prints a line beginning
with:

```text
Aligned ... links; created quickstart-output/alignment.json
```

Open the JSON and confirm that the French units appear as the base text and the
English units appear as translation text.

## 4. Build the reading EPUB

```console
bilingual-footnotes quickstart-output/reading-fr.epub quickstart-output/notes-en.epub \
  --footnote-source second \
  --output quickstart-output/reading-with-notes.epub \
  --alignment-output quickstart-output/epub-alignment.json \
  --aligner-python .venv-bilingual/bin/python
```

The command reports these phases:

```text
[reading] Reading both EPUBs
[containment] Locating corresponding global text
[fine-alignment] Aligning corresponding text
[rendering] Rendering translation footnotes
[verification] Verifying the rendered EPUB
[complete] Build completed and verified
```

Runs lasting longer than 60 seconds also print the active phase and elapsed
time.

Success creates `quickstart-output/reading-with-notes.epub`. Open it in a target
reader and check all three English notes and their return links.

## Run the tutorial again

Choose another output directory when recreating samples. Every generated
result uses a new path.

The commands reuse an operating-system cache of book-derived embeddings by
default. Check its location and size with `bilingual-align-resources status`.
It has no automatic size limit; clear it explicitly when you no longer want
the reuse data.
