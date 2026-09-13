# bilingual-text-align

Align two corresponding UTF-8 texts and save the result as JSON. The command
supports complete parallel texts and pairs where one text appears inside a
larger edition.

## Requirements

- Python 3.11–3.13 for `bilingual-align`
- Python 3.12 and `uv` for the alignment environment
- Several gigabytes of free space for the alignment environment and model

The default model accepts multilingual text. French–English has the current
end-to-end quality evidence. Other language pairs are experimental. The
default sentence splitting works best with Latin-script punctuation and
capitalization; use `--unit line` for text already divided into one unit per
non-empty line.

## Install and prepare

```console
uv tool install bilingual-text-align
bilingual-align-setup --python python3.12 --venv .venv-bilingual
```

Setup installs the isolated semantic dependencies and pinned model, so it
requires network access. Alignment runs locally afterward.

## Align two files

```console
bilingual-align french.txt english.txt \
  --footnote-source second \
  --output alignment.json \
  --aligner-python .venv-bilingual/bin/python
```

Choose the output roles with `--footnote-source`:

| Value | Base text | Translation text |
| --- | --- | --- |
| `second` | first input | second input |
| `first` | second input | first input |

The output path must be new and distinct from both inputs. Use absolute paths
for unattended runs.

Worker startup allows 10 minutes by default for loading the installed model.
Alignment has no short processing deadline because large inputs may take hours on a CPU.
During processing, the command prints an elapsed-time update every 60 seconds.
Change the frequency with `--progress-interval SECONDS`, or use `0` to disable
updates.

## Manage local resources

The semantic model is required and versioned independently from the optional
processing cache. By default both use the operating system's per-user cache
area. The processing cache stores reusable book-derived embeddings, has no
automatic size limit, and can be disabled for a run with `--no-aligner-cache`.

```console
bilingual-align-resources status
bilingual-align-resources model install
bilingual-align-resources model verify
bilingual-align-resources model remove
bilingual-align-resources cache clear
bilingual-align-resources cache prune --max-size-gb SIZE
```

`cache prune` removes the least recently used embeddings until the explicit
target is met. Neither pruning nor clearing removes the model. Model removal
does not remove the processing cache.

## Review the result

The JSON records the selected roles, source units, alignment links, gaps, and
run metadata. Each source unit appears once and in order. Empty sides represent
text present in only one edition. Groups may contain multiple units from either
text.

Scores support comparison and diagnosis. They provide no calibrated measure of
translation quality. Review dialogue, grouped units, omissions, and the start
and end of the corresponding passages before relying on a new text pair.

## Supported scope

- UTF-8, broadly monotonic translations
- One unambiguous occurrence when locating text inside a larger edition
- macOS and Linux command-line workflows
- Experimental Windows and non-French–English usage

See `bilingual-align --help` for all command options.
