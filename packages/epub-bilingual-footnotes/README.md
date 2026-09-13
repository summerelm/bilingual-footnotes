# epub-bilingual-footnotes

Create a new reading EPUB from two corresponding editions. The result
keeps one edition as the reading text and adds linked notes from the other.
Processing runs locally.

## Requirements

- Python 3.11–3.13 for `bilingual-footnotes`
- Python 3.12 and `uv` for the alignment environment
- Several gigabytes of free space for the alignment environment and model

The default model accepts multilingual text. French–English has the current
end-to-end quality evidence. Other language pairs are experimental, especially
when their punctuation and capitalization differ from Latin-script conventions.

## Install and prepare

```console
uv tool install bilingual-text-align
uv tool install epub-bilingual-footnotes
bilingual-align-setup --python python3.12 --venv .venv-bilingual
```

Setup installs the isolated semantic dependencies and pinned model, so it
requires network access. Book processing runs locally afterward.

## Build a book

For a French reading copy with English notes:

```console
bilingual-footnotes french.epub english.epub \
  --footnote-source second \
  --output french-with-english-notes.epub \
  --alignment-output french-english-alignment.json \
  --aligner-python .venv-bilingual/bin/python
```

Choose the reading edition with `--footnote-source`:

| Value | Reading EPUB | Note source |
| --- | --- | --- |
| `second` | first input | second input |
| `first` | second input | first input |

Both inputs remain unchanged. Output and optional alignment paths must be new,
distinct, and located in existing writable directories.

Worker startup allows 10 minutes by default for loading the installed model. A
full novel may take hours on a CPU.
During a build, the command prints the active phase and elapsed time every 60
seconds. Change the frequency with `--progress-interval SECONDS`, or use `0` to
disable updates.

## Check the result

The command reports its current phase and completes after structural
verification. Corresponding editions may contain extra front matter, end
matter, or other one-sided passages. Ambiguous or weak edition matches stop
with an error.

Open the result in the reader applications you plan to use. Some readers show
notes as popups; others navigate to the note and back. Manually sample:

- Dialogue and paragraphs split differently between editions
- Combined notes and passages present in only one edition
- The first and last corresponding passages
- Chapter and content-document boundaries

The optional `--alignment-output` JSON contains hashes, links, gaps, timings,
and the verification summary. Retain it when diagnosing a result or when you
want a reproducible record of how the EPUB was produced.

Scores support comparison and diagnosis. They provide no calibrated measure of
translation quality.

## Model and processing cache

The required semantic model and optional book-embedding cache use separate
directories. The cache is enabled by default, grows as needed, and can be
cleared with `bilingual-align-resources cache clear`. Use
`bilingual-align-resources status` to inspect both, or its `model` and `cache`
commands to manage them independently. Pass `--no-aligner-cache` when a run
should not read or write reusable embeddings.

## Troubleshooting

- Missing-input error: check the input paths and file permissions.
- `Invalid EPUB` or `No prose text found`: use a readable prose EPUB.
- `output must be a new path`: choose an unused path distinct from both inputs.
- Missing output-directory error: create or select the parent directory.
- Worker setup or startup errors: rerun `bilingual-align-setup` and pass the
  created Python path with `--aligner-python`.
- Containment rejection: confirm that the books are corresponding editions and
  avoid repeated or heavily abridged collections.

## Supported scope

The supported product scope covers local EPUB input and new EPUB output. Other
ebook formats and editable alignment input are not currently supported. macOS
and Linux are supported; Windows remains experimental.

See `bilingual-footnotes --help` for all command options.
