#!/usr/bin/env python3
"""Count the entries instruction-gen's filter keeps and drops, without calling Gemini.

Reads a classify manifest and applies `instruction-gen`'s `keep_sources`, `forbid_any` and
`allow_any` from config.yml, in that order. Entries of `mt_exempt_domains` skip the MT
check, as in the stage. Prints, per source, the entries kept and
dropped, the words and chunks (one Gemini call each) kept, how many entries each
forbid_any label drops and how many kept entries have each allow_any label.

An entry is counted under every label above its threshold, so the per-label counts can add
up to more than the entries.

To see what a change to config.yml does, save the counts before it and compare after it:

Usage:
    python -m scripts.count_filter
    python -m scripts.count_filter --save before.json
    # edit allow_any / forbid_any in config.yml, then:
    python -m scripts.count_filter --compare before.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from kurdish_ins.instruction_gen import InstructionGenStage, chunks
from kurdish_ins.stage import run_name, stage_config

# Why an entry was kept or dropped
KEPT_SOURCE = "kept: keep_sources"
KEPT_ALLOWED = "kept: allowed, not forbidden"
FORBIDDEN = "dropped: forbidden"
NOT_ALLOWED = "dropped: not forbidden, no allow_any label"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--manifest", help="classify manifest (default: the one instruction-gen reads)")
    parser.add_argument("--save", metavar="FILE", help="write the counts to this JSON file")
    parser.add_argument("--compare", metavar="FILE", help="print the change from counts saved with --save")
    return parser.parse_args()


def labels_above(scores, thresholds):
    """The labels of `thresholds` that score above their threshold, in config order."""
    return [label for label, threshold in thresholds.items() if scores.get(label, 0) > threshold]


def count(manifest, settings, stage):
    """{source: {entries, kept, dropped, words, chunks, mt_exempted, reasons, dropped_by, kept_with}}.

    Forbidden labels are checked before allowed ones, so `dropped_by` counts every entry a
    forbidden label drops, whether or not it also had an allowed label. `mt_exempted`
    counts the kept entries MT would have dropped but for `mt_exempt_domains`.
    """
    allow, forbid = settings["allow_any"], settings["forbid_any"]
    counts = defaultdict(lambda: {"entries": 0, "kept": 0, "dropped": 0, "words": 0, "chunks": 0, "mt_exempted": 0,
                                  "reasons": Counter(),
                                  "dropped_by": Counter({label: 0 for label in forbid}),
                                  "kept_with": Counter({label: 0 for label in allow})})
    with open(manifest, encoding="utf-8") as mf:
        for n, line in enumerate(mf, 1):
            record = json.loads(line)
            c = counts[record["source"]]
            c["entries"] += 1
            scores = record["register_scores"]
            forbidden = labels_above(scores, stage.forbid_any(record))
            allowed = labels_above(scores, allow)
            if record["source"] in settings["keep_sources"]:
                reason = KEPT_SOURCE
            elif forbidden:
                reason = FORBIDDEN
                c["dropped_by"].update(forbidden)
            elif not allowed:
                reason = NOT_ALLOWED
            else:
                reason = KEPT_ALLOWED
            c["reasons"][reason] += 1
            if reason.startswith("kept"):
                c["kept"] += 1
                c["kept_with"].update(allowed)
                c["mt_exempted"] += "MT" in labels_above(scores, forbid)
                c["words"] += len(record["text"].split())
                c["chunks"] += sum(len(text.split()) >= settings["min_words"]
                                   for text in chunks(record["text"], settings["chunk_words"]))
            else:
                c["dropped"] += 1
            if n % 10_000 == 0:
                print(f"  [{n:,}]", end="\r", flush=True)
    return {source: {key: dict(value) if isinstance(value, Counter) else value for key, value in c.items()}
            for source, c in counts.items()}


def report(counts):
    lines = []
    for source, c in counts.items():
        lines.append(f"\n{source}")
        lines.append(f"  {c['entries']:,} entries: kept {c['kept']:,} ({c['kept'] / c['entries']:.0%}), "
                     f"dropped {c['dropped']:,}")
        for reason in (KEPT_SOURCE, KEPT_ALLOWED, FORBIDDEN, NOT_ALLOWED):
            if c["reasons"].get(reason):
                lines.append(f"    {reason:<43} {c['reasons'][reason]:,}")
        lines.append(f"  kept: {c['words']:,} words, {c['chunks']:,} chunks, "
                     f"{c['mt_exempted']:,} of them only thanks to mt_exempt_domains")
        lines.append("  dropped by (forbid_any) " + ", ".join(f"{label} {n:,}" for label, n in c["dropped_by"].items()))
        lines.append("  kept with (allow_any)   " + ", ".join(f"{label} {n:,}" for label, n in c["kept_with"].items()))
    return "\n".join(lines)


def compare(before, after):
    """The change in each count per source, and in each label's dropped and kept count."""
    lines = ["\nChange from the saved counts"]
    for source in sorted(set(before) | set(after), key=str):
        b, a = before.get(source, {}), after.get(source, {})
        lines.append(f"\n{source}")
        for key in ("kept", "dropped", "words", "chunks", "mt_exempted"):
            lines.append(f"  {key:<24} {change(b.get(key, 0), a.get(key, 0))}")
        for key, title in (("dropped_by", "dropped by"), ("kept_with", "kept with")):
            old, new = b.get(key, {}), a.get(key, {})
            for label in {**old, **new}:
                lines.append(f"  {title} {label:<{14 - len(title)}} {change(old.get(label, 0), new.get(label, 0))}")
    return "\n".join(lines)


def change(old, new):
    return f"{old:,} -> {new:,} ({new - old:+,})"


def main():
    args = parse_args()
    config = Path(args.config)
    settings = stage_config("instruction-gen", config)
    stage = InstructionGenStage(**settings)
    manifest = Path(args.manifest) if args.manifest else stage.manifest(Path("data/classify") / run_name(config))

    print(f"Counting {manifest}\n  keep_sources {settings['keep_sources']}\n"
          f"  forbid_any   {settings['forbid_any']}\n  allow_any    {settings['allow_any']}\n"
          f"  mt_exempt_domains {settings['mt_exempt_domains']}")
    counts = count(manifest, settings, stage)
    print(report(counts))

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            print(compare(json.load(f), counts))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(counts, f, ensure_ascii=False, indent=2)
        print(f"\nSaved the counts to {args.save}")


if __name__ == "__main__":
    main()
