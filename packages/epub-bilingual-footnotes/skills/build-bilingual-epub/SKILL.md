---
name: build-bilingual-epub
description: Build, inspect, validate, or diagnose a translation-note EPUB from two corresponding EPUBs. Use the text-alignment skill for plain-text output.
---

# Build a bilingual EPUB

Use this skill to create a new reading EPUB from two corresponding editions.
One edition remains the reading text and the other supplies linked
translation notes.

## Confirm the request

- Resolve two readable local EPUB inputs.
- Determine which edition the user wants to read.
- Choose new, distinct paths for the EPUB and optional alignment JSON.
- Keep books, outputs, caches, and logs within user-approved locations.
- Require explicit authorization and a rights review before uploading,
  publishing, redistributing, or committing book-derived material.

Ask about direction before an expensive run when the requested reading edition
is unclear.

## Prepare the alignment environment

Setup requires `uv`, Python 3.12, network access, and several gigabytes of free
space:

```console
bilingual-align-setup \
  --python python3.12 \
  --venv /absolute/path/to/.venv-bilingual
```

Setup installs both the isolated worker environment and pinned semantic model.
Verify them with a short sample before processing a whole book. The separate
book-embedding cache is enabled by default, grows as needed, and can be cleared
with `bilingual-align-resources cache clear` when the user wants to remove that
reusable data.

## Build the EPUB

`--footnote-source second` keeps the first input as the reading edition.
`--footnote-source first` keeps the second input as the reading edition.

```console
bilingual-footnotes /absolute/path/to/first.epub /absolute/path/to/second.epub \
  --footnote-source second \
  --output /absolute/path/to/new-bilingual.epub \
  --alignment-output /absolute/path/to/new-alignment.json \
  --aligner-python /absolute/path/to/.venv-bilingual/bin/python
```

Retain `--alignment-output` for diagnosis, long-run records, and release
evidence. Worker startup allows 10 minutes for loading the installed model;
increase it with `--worker-startup-timeout SECONDS` on unusually slow machines.
Full novels may run for hours on a CPU. Expect active-phase and elapsed-time
updates every 60 seconds; use `--progress-interval SECONDS` to adjust them.

Use `bilingual-align-resources status` when model or cache state matters. Model
install, verify, and removal are distinct from cache clear or explicit pruning.
Use `--no-aligner-cache` only when the user wants to avoid reuse. Never clear,
prune, or remove resources without authorization or while a run is active.

## Validate the result

The command performs structural verification before reporting completion.
Then check the user-visible result:

- Confirm both source files retain their original hashes.
- Open the EPUB in the reader applications the user plans to use.
- Test popup presentation where available, plus note navigation and backlinks.
- Sample dialogue, grouped notes, omissions, and passages present in one
  edition.
- Inspect the first and last corresponding passages.
- Inspect notes around chapter and content-document boundaries.
- Confirm unrelated front matter, colophons, and end matter remain without
  translation notes.
- Retain the alignment JSON and logs for any result that needs diagnosis.

Scores are diagnostics only. Structural verification establishes file and link
integrity. Semantic quality requires representative manual review.

French–English has the current end-to-end evidence. Treat other language pairs
as experimental, especially when punctuation and capitalization differ from
Latin-script conventions.

## Handle failures

- Missing or invalid inputs: confirm paths, permissions, and readable prose
  content.
- Invalid output paths: choose unused paths in existing writable directories.
- Worker failures: rerun `bilingual-align-setup` and pass the resulting Python
  path with `--aligner-python`.
- Weak or ambiguous edition matches: confirm the books are corresponding,
  broadly complete editions.
- Interrupted long jobs: inspect service state and logs before deciding whether
  to retry.
