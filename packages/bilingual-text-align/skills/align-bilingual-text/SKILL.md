---
name: align-bilingual-text
description: Align corresponding UTF-8 text files, locate one ordered text inside another, inspect alignment JSON, prepare the alignment environment, or diagnose bilingual-text-align. Use the EPUB skill for ebook output.
---

# Align bilingual text

Use this skill when the requested result is an alignment JSON file or an
EPUB-independent text-location operation.

## Confirm the request

- Resolve two readable UTF-8 inputs in corresponding reading order.
- Confirm which input supplies the base text and which supplies the translation.
- Choose a new output path distinct from both inputs.
- Ask about direction before an expensive run when the requested roles are
  unclear.
- Keep source text, outputs, caches, and logs within user-approved locations.
- Require explicit authorization for uploads, publication, redistribution, or
  remote changes.

For ebook output, use the `build-bilingual-epub` skill.

## Prepare the alignment environment

Setup requires `uv`, Python 3.12, network access, and several gigabytes of free
space:

```console
bilingual-align-setup \
  --python python3.12 \
  --venv /absolute/path/to/.venv-bilingual
```

Setup installs both the isolated worker environment and the pinned semantic
model. Confirm their paths in its output. The separate processing cache stores
book-derived embeddings and is enabled by default. It can be cleared with
`bilingual-align-resources cache clear` when the user wants to remove that
reusable data.

## Run alignment

```console
bilingual-align /absolute/path/to/first.txt /absolute/path/to/second.txt \
  --footnote-source second \
  --output /absolute/path/to/new-alignment.json \
  --aligner-python /absolute/path/to/.venv-bilingual/bin/python
```

`--footnote-source second` keeps the first input as the base text and uses the
second as the translation. Select `first` to reverse those roles.

The default sentence mode is best suited to Latin-script punctuation and
capitalization. For pre-segmented multilingual input, use `--unit line` with
one intended unit per non-empty line.

Worker startup allows 10 minutes for loading the installed model. Increase it
with `--worker-startup-timeout SECONDS` on unusually slow machines.
Large alignments may run for hours. Expect an elapsed-time update every 60
seconds; use `--progress-interval SECONDS` to adjust it.

Use `bilingual-align-resources status` when model or cache state matters. Model
install, verify, and removal are distinct from cache clear or explicit pruning.
Use `--no-aligner-cache` only when the user wants to avoid reuse. Never clear,
prune, or remove resources without authorization; management must fail while a
selected resource is in use.

## Validate the result

Parse the JSON and confirm:

- `footnote_source` matches the requested roles.
- Every source unit appears once and in order.
- Empty sides correspond to passages present in one input.
- Grouped links preserve the intended reading sequence.
- Run metadata identifies the configuration used.

Scores are diagnostics only. Manually sample dialogue, grouped links, omissions,
and the first and last corresponding passages. French–English has the current
end-to-end evidence; review other language pairs as experimental.
