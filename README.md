# bilingual-footnotes

Create bilingual reading EPUBs with linked translation notes, or align two
parallel text files. Processing runs locally.

## Choose a workflow

| Need | Guide | Command |
| --- | --- | --- |
| Add translation notes to an EPUB | [`epub-bilingual-footnotes`](packages/epub-bilingual-footnotes) | `bilingual-footnotes` |
| Align two UTF-8 text files | [`bilingual-text-align`](packages/bilingual-text-align) | `bilingual-align` |
| Build a bilingual EPUB in a desktop window | [`bilingual-text-align-ui`](packages/bilingual-text-align-ui) | `bilingual-align-ui` |

Both workflows require corresponding texts in the same reading order.

## Use the desktop app

For the simplest workflow, download the archive for your computer from the
[latest release](https://github.com/summerelm/bilingual-footnotes/releases/latest):

- **Windows x64:** extract the archive and open `Bilingual Footnotes.exe`.
- **Mac with Apple silicon:** extract the `macOS-arm64` archive, move
  **Bilingual Footnotes.app** to Applications, and open it.
- **Mac with an Intel processor:** use the `macOS-x86_64` archive instead.

Choose the EPUB you want to read, its translation, and a new output filename.
The app includes the alignment worker; use **Model & storage…** to install the
required model or clear reusable book embeddings. These builds are not
code-signed or notarized, so your operating system may ask you to confirm before
opening them.

The source books stay unchanged. Open the resulting EPUB in your usual reader;
supported readers show the translations as popup footnotes, while others offer
ordinary note links and backlinks.

## Use the commands or an agent

The commands are useful for automation, repeatable batches, and inspecting the
alignment JSON. They require:

- Python 3.11–3.13 for the commands
- Python 3.12 and `uv` for the alignment environment
- Several gigabytes of free space for the alignment environment and model
- Network access during setup

The supported workflow uses CPU processing. Short samples may finish in
minutes; full novels may take hours. macOS and Linux are supported. Windows
remains experimental.

Long commands print their active phase and elapsed time every 60 seconds.

The default model accepts multilingual text. French–English has the current
end-to-end quality evidence; treat other language pairs as experimental and
review their results carefully.

## Model and processing cache

`bilingual-align-setup` creates the isolated alignment environment and installs
the pinned semantic model. The model is required; the processing cache is not.
The cache stores book-derived embeddings so repeated work can be faster. It
grows as needed and can be cleared with **Clear cache** in the desktop app or
`bilingual-align-resources cache clear` on the command line.

Inspect or manage them independently:

```console
bilingual-align-resources status
bilingual-align-resources model verify
bilingual-align-resources model install
bilingual-align-resources model remove
bilingual-align-resources cache clear
bilingual-align-resources cache prune --max-size-gb SIZE
```

The desktop app provides the same model install, verify, remove, and cache-clear
controls under **Model & storage…**. Resource deletion is refused while
alignment is using the selected directory.

## First run

Follow the [five-minute quickstart](docs/quickstart.md) with the included CC0
sample texts and EPUBs.

## Documentation

- [Five-minute quickstart](docs/quickstart.md)

## Agent skills

- [`align-bilingual-text`](packages/bilingual-text-align/skills/align-bilingual-text/SKILL.md)
- [`build-bilingual-epub`](packages/epub-bilingual-footnotes/skills/build-bilingual-epub/SKILL.md)

Agents must stay within user-approved paths and permissions. Publishing,
redistributing books, changing remotes, and releasing packages require explicit
authorization.

Repository-authored code uses the [MIT license](LICENSE). Package distributions
include their third-party notices.
