# Third-party notices

`bilingual-text-align` interoperates with software and model assets installed
or downloaded separately from the base distribution. Their licenses continue
to apply.

## Vecalign

Vecalign is Copyright 2019 Brian Thompson and is distributed under the Apache
License 2.0.

- Source: <https://github.com/thompsonb/vecalign>
- License: <https://github.com/thompsonb/vecalign/blob/master/LICENSE>

`bilingual-align-setup` installs the exact source revision recorded in
`bilingual_text_align.setup_worker`.

## LaBSE

The default `sentence-transformers/LaBSE` model is distributed under the
Apache License 2.0.

- Model: <https://huggingface.co/sentence-transformers/LaBSE>
- License: <https://www.apache.org/licenses/LICENSE-2.0>

The pinned model revision is downloaded by `bilingual-align-setup` or an
explicit model-install command. Model weights are not included in this package.

## Semantic worker dependencies

The optional worker environment also installs pinned releases of NumPy,
Cython, PyTorch, SciPy, scikit-learn, Transformers, and Sentence Transformers.
Each project retains its own copyright and license; this package does not
relicense them.
