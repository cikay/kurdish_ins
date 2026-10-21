# kurdish-ins

Instruction-tuning data for Kurmanji Kurdish, built from the Kurmanji (`kmr_Latn`) subsets
of [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and
[kurdish-theses-yok](https://huggingface.co/datasets/muzaffercky/kurdish-theses-yok).

The pipeline is built one stage at a time. So far it has two stages, `classify` and
`instruction-gen`.

## Installation

```bash
pipenv install --dev
```

`instruction-gen` calls Gemini, so it needs `GEMINI_API_KEY` in `.env`. `classify` does not.

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

## Running on Modal

`modal_app/app.py` runs a stage on a [Modal](https://modal.com) GPU, with the same stage code
`pipeline.py` runs locally. `pipenv install` installs Modal; log in once:

```bash
modal setup
```

Then, from the repository root:

```bash
modal run --detach -m modal_app.app::classify
modal volume get kurdish-ins /classify/<run> data/modal/   # <run> is full, or e.g. s5
```

It reads `classify.input_file` like a local run. Commented out or `null`, it classifies
every source into `/classify/full`; a file such as `data/s5.jsonl` is uploaded to the
volume for you and classified into `/classify/s5`. The other settings come from
`config.yml` as usual. The GPU, the timeout, and the GPU's batch size and dtype come from
its `modal` block; Modal runs the model in `bfloat16`. The volume `kurdish-ins` holds
`data/` and the Hugging Face cache, so the model and the datasets are downloaded only on
the first run. Running on Modal bills your Modal account.

## Pipeline stages

Each stage writes `data/<stage>/<run>/manifest.jsonl`. The next stage reads the same run's
file. `data/` is gitignored.

The run is `full` when `classify.input_file` is commented out or `null`, and otherwise
the input file's name without its extension (`data/sample.jsonl` → `sample`). A sample run and a full run
therefore never overwrite each other:

```
data/classify/full/manifest.jsonl
data/classify/sample/manifest.jsonl
```

| Stage      | Reads                          | Writes                               |
|------------|--------------------------------|--------------------------------------|
| `classify` | `sources`, or `input_file`     | `data/classify/<run>/manifest.jsonl` |
| `instruction-gen` | `data/classify/<run>/manifest.jsonl` | `data/instruction-gen/<run>/manifest.jsonl` |

### classify

Labels each entry of every source with its web register using
[TurkuNLP/web-register-classification-multilingual](https://huggingface.co/TurkuNLP/web-register-classification-multilingual)
(for example `IN`/`en` for an encyclopedia article, `NA`/`ne` for news). It drops nothing.
Choosing which registers become instruction pairs is left to a later stage, so changing
that rule never means classifying again.

Each manifest row has `source` (the dataset it came from), `id`, `url`, `domain`, `text`, `register_labels` and
`register_scores` (every label's score, so a later stage can use a different threshold
without running the model again).

A label is written only when its own score is above `threshold`. A sub-register does not
add its main register: `en` alone means `IN` scored below the threshold.

When the run ends, the stage prints its stats for each source: entries, words, entries
with no label, how many entries got each main register and sub-register, and how many
entries got each label set (all of an entry's labels together, e.g. `IN en`). It also
writes them to `stats.json`, next to `manifest.jsonl`.

Settings in `config.yml`:

| Setting      | Meaning                                                       |
|--------------|---------------------------------------------------------------|
| `sources`    | the datasets read: each has `dataset` (Hugging Face id), `files` (glob of its parquet files) and `columns` (the dataset's names for `id`, `url` and `text`) |
| `input_file` | JSONL of entries to classify instead, e.g. `data/sample.jsonl`; commented out or `null` means every source |
| `threshold`  | a register is predicted when its score is above this          |
| `batch_size` | entries classified at a time, and windows per model call      |
| `max_length` | tokens per window; 512 is XLM-R's limit                       |
| `stride`     | tokens neighbouring windows share; 0 means no overlap         |
| `dtype`      | precision the model runs in: `float32`, or `bfloat16` (several times faster on a GPU; scores move in the 2nd–3rd decimal) |

Kurmanji is not one of the classifier's training languages. On a small test set it still
labelled Wikipedia articles `IN`/`en` and news sites `NA`/`ne`.

### instruction-gen

Asks Gemini for one question per chosen entry, answered by one continuous span copied from
the entry (the prompt is in `kurdish_ins/prompts.py`), and keeps a pair only when the span
appears verbatim in the entry. The answer is therefore always the source's own Kurmanji.

It uses an entry when any label of `allow_any` scores above its threshold and no label of
`forbid_any` does, and every entry of `keep_sources` whatever its scores. Entries of
`mt_exempt_domains` (Kurdish Wikipedia, Wiktionary and kurdipedia) skip the `MT` check
only, since the classifier gives many Wikipedia articles a moderate `MT` score. The thresholds
are per label and read the scores classify wrote, so changing one needs no new classify run. An entry longer than `chunk_words` is cut at line
breaks, and each chunk gets its own question.

It reads `input_manifest` when that is set, e.g. `data/modal/full/manifest.jsonl` for a
classify run downloaded from Modal, and otherwise `data/classify/<run>/manifest.jsonl`.

`limit` sends only that many chunks, picked at random with `seed`, so two sample runs (with
different prompts, say) ask about the same chunks. A sample run writes its files under
`limit-<limit>/`, e.g. `data/instruction-gen/full/limit-200/`, so it never mixes with the
full run.

Each manifest row has `source`, `id`, `url`, `domain`, `chunk`, `context` (the chunk),
`question`, `answer`, `answer_start` (its offset in `context`), `model`,
`register_labels` and `register_scores`. Every chunk's outcome (`kept`, `no_question`,
`span_not_found`, `too_short` or `bad_response`) and the model's raw reply go to
`responses.jsonl`, and their counts, in total and per label set, to `stats.json`.

The stage resumes: running it again skips the chunks in `responses.jsonl`. So before
running it with a changed prompt, move the folder of the earlier run away.

## How the datasets are read

`kurdish_ins/hub.py` downloads a source's parquet files into the Hugging Face cache and
reads them locally. FineWeb2's `kmr_Latn` is a single 562 MB file, the theses' one 65 MB
file, and each is downloaded once.

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

The tests never load the classifier model or download a dataset. They pass a fake
classifier to `ClassifyStage.classify`.

## Layout

```
pipeline.py            runs the stages listed in config.yml
config.yml             stage order and every stage setting
kurdish_ins/
  stage.py             Stage base class, stage_config, run_name, load_manifest
  classify.py          classify stage
  register_classifier.py  the register model and its labels
  hub.py               reads a dataset from local parquet files
  logs.py              logging setup
modal_app/app.py       runs stages on Modal
scripts/               standalone tools
tests/
```

The files at the root other than `pipeline.py` (`generate_pairs.py`,
`prepare_register_data.py`, `test_register_classifier.py`, `compare_lengths.py`) are from
before the pipeline. They will be replaced by stages.
