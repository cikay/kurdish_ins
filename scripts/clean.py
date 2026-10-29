#!/usr/bin/env python3
"""Clean the pairs push_hf put on the Hub and write them as a subset of their own.

The thesis PDFs, and some FineWeb2 pages, hold garbled letters: þ for ş where Turkish text
was read in the wrong code page, Ģ for ş where a PDF was turned into text. A model trained
on them learns to write them. Letters whose correct form is certain are replaced
(`replacements` in clean.yml), and a pair whose question or answer still holds garbling
with no single correct form is dropped (`drop_patterns`).

The question, the answer and the context get the same replacements, so every answer is
still a verbatim copy of its context. Only the question and the answer are checked for
drops. The text is put in Unicode NFC first, so a letter written as a base and a combining
mark is the same text as its one-character form.

On the first full run this kept 30,927 of 32,991 pairs and replaced 25,606 letters in them,
most of them þ (17,366).

The pairs go to data/clean/<output_subset>/train-00000-of-00001.parquet, the layout the Hub
uses for a subset, with the counts in data/clean/stats.json. --push also pushes them to
`dataset` as `output_subset`, leaving the other subsets as they are. The token
is HF_TOKEN in .env, as for push_hf.

Usage:
    python -m scripts.clean
    python -m scripts.clean --push
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download

# Its settings are kept apart from config.yml, which holds the pipeline's stages
CONFIG_FILE = Path("clean.yml")
OUTPUT_DIR = Path("data/clean")
# The fields every pair has on the Hub, in their order there
COLUMNS = ["instruction", "output", "context", "id", "domain", "source", "url"]
# The fields that are text, and the ones a model is trained on, which alone are checked for drops
TEXT_COLUMNS = ["instruction", "output", "context"]
CHECKED_COLUMNS = ["instruction", "output"]
# The split push_hf wrote; the dataset has no other
SPLIT = "train"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="also push the cleaned pairs to the Hub")
    return parser.parse_args()


def read_pairs(dataset: str, subset: str) -> list[dict]:
    root = snapshot_download(dataset, repo_type="dataset", allow_patterns=f"{subset}/*.parquet")
    paths = sorted(Path(root).glob(f"{subset}/{SPLIT}-*.parquet"))
    return [row for path in paths for row in pq.read_table(path, columns=COLUMNS).to_pylist()]


def clean(pairs: list[dict], replacements: dict[str, str], drop_patterns: dict[str, str]):
    """The kept pairs with their letters replaced, the dropped pairs counted by the name of
    the pattern that dropped them, and the replaced letters in the kept pairs counted by
    their garbled form."""
    compiled_patterns = {name: re.compile(pattern) for name, pattern in drop_patterns.items()}
    kept, dropped, replaced = [], Counter(), Counter()
    for pair in pairs:
        texts = {column: unicodedata.normalize("NFC", pair[column]) for column in TEXT_COLUMNS}
        cleaned = dict(texts)
        for column in TEXT_COLUMNS:
            for garbled, correct in replacements.items():
                cleaned[column] = cleaned[column].replace(garbled, correct)

        reason = next((name for name, pattern in compiled_patterns.items()
                               if any(pattern.search(cleaned[column]) for column in CHECKED_COLUMNS)), None)

        if reason is not None:
            dropped[reason] += 1
            continue

        for garbled in replacements:
            count = sum(texts[column].count(garbled) for column in TEXT_COLUMNS)
            if count:
                replaced[garbled] += count

        kept.append({**pair, **cleaned})
    return kept, dropped, replaced


def write(pairs: list[dict], output_subset: str) -> Path:
    path = OUTPUT_DIR / output_subset / f"{SPLIT}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pairs).select(COLUMNS), path)
    return path


def push(pairs: list[dict], dataset: str, output_subset: str) -> None:
    from datasets import Dataset

    from scripts.push_hf import hf_token

    Dataset.from_list([{column: pair[column] for column in COLUMNS} for pair in pairs]).push_to_hub(
        dataset, config_name=output_subset, token=hf_token())


def main():
    args = parse_args()
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    pairs = read_pairs(config["dataset"], config["subset"])
    print(f"Read {len(pairs):,} pairs from {config['dataset']} ({config['subset']})")

    kept, dropped, replaced = clean(pairs, config["replacements"], config["drop_patterns"])
    stats = {"pairs": len(pairs), "kept": len(kept), "dropped": dict(dropped.most_common()),
             "replaced": dict(replaced.most_common())}
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    path = write(kept, config["output_subset"])
    (OUTPUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(kept):,} pairs to {path}")

    if args.push:
        push(kept, config["dataset"], config["output_subset"])
        print(f"Pushed {len(kept):,} pairs to {config['dataset']} as the {config['output_subset']} subset")


if __name__ == "__main__":
    main()
