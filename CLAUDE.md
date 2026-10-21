# CLAUDE.md

What the pipeline does and how to run it is in `README.md`. This file is only how to work
on it.

## Cost

- Never run `instruction-gen` (it submits a paid Gemini batch job) or anything with
  `modal run` (it bills a GPU) unless the user asks for that run.
- `limit` in `config.yml` is the user's choice of sample size; leave it as it is.
- Before a change makes Gemini calls cost more (more tokens, more chunks, a higher
  thinking level), say what it will cost.

## Code

- Every setting lives in `config.yml`, with a comment saying what it does and why its
  value was picked. Never give a constructor argument a default in Python.
- Constants that are not tuning knobs (`PROMPT_NAME`, `THINKING_LEVEL`, poll intervals) sit
  at the top of their module, each with a comment saying why.
- A stage module starts with a docstring describing the whole stage. Keep it true when
  the stage changes.
- Comments and docstrings say why, in short plain sentences.
- `hub.py` downloads a source's parquet files and reads them locally. Do not switch it to
  `load_dataset(..., streaming=True)`: a stream not read to the end hangs the process at
  exit, because Python kills pyarrow's HTTP read thread mid-request and arrow's thread
  pool waits for it forever. Closing the stream or turning off pre-buffering does not
  help.

## Adding a stage

1. Write a `Stage` subclass in `kurdish_ins/` with a `name` and
   `run(input_dir, output_dir)`. Read `input_dir / "manifest.jsonl"` and write
   `output_dir / "manifest.jsonl"`.
2. Register it in `REGISTRY` in `pipeline.py`.
3. Add its name to `stages` in `config.yml`, plus a section with a `description` and its
   settings. Every setting is passed to the constructor as a keyword argument.
4. Document it in `README.md`: a row in the stage table and a section of its own.

## Tests

- Run `pipenv run pytest` after every change.
- Tests never load the classifier, download a dataset or call Gemini: they pass a fake
  classifier to `ClassifyStage.classify` and use `FakeBatchClient` for the Batch API.
  Keep new tests that way.
- Test names are sentences describing the behaviour, grouped in classes by function
  (`TestCheck`, `TestRun`).

## Docs

When behaviour, a setting, an output file or a manifest field changes, update in the same
commit: `README.md`, the setting's comment in `config.yml` and the module docstring.

## Commits

- Subject: imperative, says what changes for the user, no prefix
  (`Back off while waiting for an instruction-gen batch job`).
- Body: plain prose, wrapped at about 80 columns, saying what changed and why, with the
  numbers behind the decision when there are some.
- One change per commit.
