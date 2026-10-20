#!/usr/bin/env python3
"""Save a sample of FineWeb2 Kurmanji entries, spread over many domains.

Reads FineWeb2 kmr_Latn (see kurdish_ins/hub.py) and keeps entries longer than --min-words words, at most
--per-domain entries per domain, until --total entries are saved.

Usage:
    python -m scripts.sample_fineweb
    python -m scripts.sample_fineweb --total 200 --per-domain 3 --min-words 300 --output data/sample.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from kurdish_ins import hub

DATASET = "HuggingFaceFW/fineweb-2"
FILES = "data/kmr_Latn/train/*.parquet"
COLUMNS = {"id": "id", "url": "url", "text": "text"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=200, help="number of entries to save")
    parser.add_argument("--per-domain", type=int, default=3, help="max entries per domain")
    parser.add_argument("--min-words", type=int, default=300, help="keep entries with more words than this")
    parser.add_argument("--output", default="data/sample.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    per_domain = Counter()
    scanned = 0
    for entry in hub.entries(DATASET, FILES, COLUMNS):
        if len(records) >= args.total:
            break
        scanned += 1
        domain = urlparse(entry["url"]).netloc
        if len(entry["text"].split()) < args.min_words or per_domain[domain] >= args.per_domain:
            continue
        per_domain[domain] += 1
        records.append({"source": entry["source"], "id": entry["id"], "url": entry["url"], "domain": domain, "text": entry["text"]})
        print(f"  [{len(records)}/{args.total}] scanned {scanned}", end="\r")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} entries from {len(per_domain)} domains "
          f"(scanned {scanned}) -> {output}")


if __name__ == "__main__":
    main()
