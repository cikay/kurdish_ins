# kurdish-ins

Instruction-tuning data for Kurmanji Kurdish, built from the Kurmanji subset of
[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`kmr_Latn`).

The pipeline is built one stage at a time. So far it has one stage, `classify`.

## Installation

```bash
pipenv install --dev
```

Later stages call Gemini, so they will need `GEMINI_API_KEY` in `.env`. `classify` does not.

## Usage

Run every command from the repository root.

```bash
python pipeline.py                     # every stage in config.yml, in order
python pipeline.py --only classify     # one stage
python pipeline.py --start-from STAGE  # a stage and all the stages after it
python pipeline.py --config other.yml  # a different config file
```

A stage can also be run on its own, as a module:

```bash
python -m kurdish_ins.classify
```

Its settings still come from `config.yml`, so running a stage on its own and running it
through `pipeline.py` behave the same.

## Pipeline stages

Each stage writes `data/<stage>/<run>/manifest.jsonl`. The next stage reads the same run's
file. `data/` is gitignored.

The run is `full` when `classify.input_file` is `null`, and otherwise the input file's name
without its extension (`data/sample.jsonl` → `sample`). A sample run and a full run
therefore never overwrite each other:

```
data/classify/full/manifest.jsonl
data/classify/sample/manifest.jsonl
```

| Stage      | Reads                          | Writes                               |
|------------|--------------------------------|--------------------------------------|
| `classify` | FineWeb2, or `input_file`      | `data/classify/<run>/manifest.jsonl` |

### classify

Labels each FineWeb2 entry with its web register using
[TurkuNLP/web-register-classification-multilingual](https://huggingface.co/TurkuNLP/web-register-classification-multilingual)
(for example `IN`/`en` for an encyclopedia article, `NA`/`ne` for news). It drops nothing.
Choosing which registers become instruction pairs is left to a later stage, so changing
that rule never means classifying again.

Each manifest row has `id`, `url`, `domain`, `text`, `register_labels` and
`register_scores` (every label's score, so a later stage can use a different threshold
without running the model again).

Settings in `config.yml`:

| Setting      | Meaning                                                       |
|--------------|---------------------------------------------------------------|
| `dataset`    | Hugging Face dataset id                                       |
| `subset`     | FineWeb2 subset                                               |
| `input_file` | JSONL of entries to classify instead, e.g. `data/sample.jsonl`; `null` means the whole subset |
| `threshold`  | a register is predicted when its score is above this          |
| `batch_size` | entries classified at a time, and windows per model call      |
| `max_length` | tokens per window; 512 is XLM-R's limit                       |
| `stride`     | tokens neighbouring windows share; 0 means no overlap         |

Kurmanji is not one of the classifier's training languages. On a small test set it still
labelled Wikipedia articles `IN`/`en` and news sites `NA`/`ne`.

## How FineWeb2 is read

`kurdish_ins/fineweb.py` downloads the subset's parquet files into the Hugging Face cache
and reads them locally. `kmr_Latn` is a single 562 MB file, and it is downloaded once.

It does not use `load_dataset(..., streaming=True)`, because a stream that is not read to
the end hangs the process at exit. pyarrow reads the next chunk over HTTP on one of its own
threads. Python kills that thread in the middle of the request when it shuts down, and
arrow's thread pool then waits for the thread forever. Closing the stream or turning off
pre-buffering does not cancel a read that has already started. A local file leaves no read
in flight.

## Adding a stage

1. Write a `Stage` subclass in `kurdish_ins/` with a `name` and
   `run(input_dir, output_dir)`. Read `input_dir / "manifest.jsonl"` and write
   `output_dir / "manifest.jsonl"`.
2. Register it in `REGISTRY` in `pipeline.py`.
3. Add its name to `stages` in `config.yml`, plus a section with a `description` and its
   settings. Every setting is passed to the stage's constructor as a keyword argument.
   Stages define no defaults in Python, so all settings live in `config.yml`.

## Scripts

Standalone tools in `scripts/`, run as modules:

```bash
python -m scripts.sample_fineweb   # 200 entries over 300 words, at most 3 per domain → data/sample.jsonl
```

Options: `--total`, `--per-domain`, `--min-words` and `--output`.

## Tests

```bash
pipenv run pytest
```

The tests never load the classifier model or download FineWeb2. They pass a fake
classifier to `ClassifyStage.classify`.

## Layout

```
pipeline.py            runs the stages listed in config.yml
config.yml             stage order and every stage setting
kurdish_ins/
  stage.py             Stage base class, stage_config, run_name, load_manifest
  classify.py          classify stage
  register_classifier.py  the register model and its labels
  fineweb.py           reads FineWeb2 from local parquet files
  logs.py              logging setup
scripts/               standalone tools
tests/
```

The files at the root other than `pipeline.py` (`generate_pairs.py`,
`prepare_register_data.py`, `test_register_classifier.py`, `compare_lengths.py`) are from
before the pipeline. They will be replaced by stages.
