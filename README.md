# kurdish-ins

Instruction-tuning data for Kurmanji Kurdish, built from the Kurmanji subset of
[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`kmr_Latn`).

## Scripts

Standalone tools in `scripts/`. Run them as modules, from the repository root:

```bash
python -m scripts.sample_fineweb   # 200 entries over 300 words, at most 3 per domain → data/sample.jsonl
```

Options: `--total`, `--per-domain`, `--min-words` and `--output`.

## How FineWeb2 is read

`kurdish_ins/fineweb.py` downloads the subset's parquet files into the Hugging Face cache
and reads them locally. `kmr_Latn` is a single 562 MB file, and it is downloaded once.

It does not use `load_dataset(..., streaming=True)`, because a stream that is not read to
the end hangs the process at exit. pyarrow reads the next chunk over HTTP on one of its own
threads. Python kills that thread in the middle of the request when it shuts down, and
arrow's thread pool then waits for the thread forever. Closing the stream or turning off
pre-buffering does not cancel a read that has already started. A local file leaves no read
in flight.
