"""Stage — Read the Kurmanji datasets and label each entry's web register.

This stage only labels; it drops nothing. Deciding which registers become instruction
pairs is a later stage's job, so changing that rule never means classifying again.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from urllib.parse import urlparse

from kurdish_ins import hub
from kurdish_ins.register_classifier import RegisterClassifier
from kurdish_ins.stage import Stage, load_manifest, run_name, stage_config


@dataclass
class ClassifyStats:
    """What the stage did, totalled per source.

    Printed and saved to `stats.json` when the run ends, so changing `threshold` or
    `max_length` in `config.yml` and running again shows what moved. A label set is an
    entry's labels together, e.g. `IN en`; `(none)` is the set of an entry with no label.
    """

    entries: Counter = field(default_factory=Counter)
    words: Counter = field(default_factory=Counter)
    unlabelled: Counter = field(default_factory=Counter)
    labels: defaultdict = field(default_factory=lambda: defaultdict(Counter))
    label_sets: defaultdict = field(default_factory=lambda: defaultdict(Counter))

    def add(self, record: dict) -> None:
        source = record["source"] or "(no source)"
        self.entries[source] += 1
        self.words[source] += len(record["text"].split())
        self.labels[source].update(record["register_labels"])
        self.label_sets[source][" ".join(record["register_labels"]) or "(none)"] += 1
        if not record["register_labels"]:
            self.unlabelled[source] += 1

    def to_dict(self) -> dict:
        """The stats as {source: {entries, words, no_label, main_registers, sub_registers, label_sets}}."""
        return {
            source: {
                "entries": entries,
                "words": self.words[source],
                "no_label": self.unlabelled[source],
                "main_registers": {label: n for label, n in self.labels[source].most_common() if label.isupper()},
                "sub_registers": {label: n for label, n in self.labels[source].most_common() if label.islower()},
                "label_sets": dict(self.label_sets[source].most_common()),
            }
            for source, entries in self.entries.items()
        }

    def report(self) -> str:
        lines = []
        for source, entries in self.entries.items():
            lines.append(f"\n{source}")
            lines.append(f"  {entries:,} entries, {self.words[source]:,} words, "
                         f"{self.unlabelled[source]:,} with no label")
            labels = self.labels[source].most_common()
            for title, wanted in (("main registers", str.isupper), ("sub-registers", str.islower)):
                counts = ", ".join(f"{label} {n:,} ({n / entries:.0%})" for label, n in labels if wanted(label))
                lines.append(f"  {title:<15} {counts or '-'}")
            label_sets = ", ".join(f"{labels} {n:,} ({n / entries:.0%})"
                                   for labels, n in self.label_sets[source].most_common())
            lines.append(f"  {'label sets':<15} {label_sets}")
        return "\n".join(lines)


class ClassifyStage(Stage):
    name = "classify"

    def __init__(self, sources: list[dict], threshold: float, batch_size: int,
                 max_length: int, stride: int, dtype: str, input_file: str | None = None):
        self._sources = sources
        self._input_file = input_file
        self._threshold = threshold
        self._batch_size = batch_size
        self._max_length = max_length
        self._stride = stride
        self._dtype = dtype

    def run(self, input_dir: Path | None, output_dir: Path) -> None:
        source = self._input_file or ", ".join(source["dataset"] for source in self._sources)
        print(f"Classifying the entries of {source} "
              f"(threshold={self._threshold}, batch_size={self._batch_size}, dtype={self._dtype})")
        classifier = RegisterClassifier(self._threshold, self._max_length, self._stride,
                                        self._batch_size, self._dtype)

        output_dir.mkdir(parents=True, exist_ok=True)
        stats = ClassifyStats()

        with open(output_dir / "manifest.jsonl", "w", encoding="utf-8") as mf:
            for count, record in enumerate(self.classify(self._entries(), classifier), 1):
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.add(record)
                print(f"  [{count}]", end="\r")

        with open(output_dir / "stats.json", "w", encoding="utf-8") as sf:
            json.dump(stats.to_dict(), sf, ensure_ascii=False, indent=2)
        print(f"\nDone — written to {output_dir}/manifest.jsonl and stats.json")
        print(stats.report())

    def classify(self, entries, classifier):
        """Yield one manifest record per entry, classifying `batch_size` entries at a time."""
        entries = iter(entries)
        while batch := list(islice(entries, self._batch_size)):
            results = classifier.classify([entry["text"] for entry in batch])
            for entry, (labels, scores) in zip(batch, results):
                yield {
                    # None for an input_file written before entries had a source
                    "source": entry.get("source"),
                    "id": entry["id"],
                    "url": entry["url"],
                    "domain": urlparse(entry["url"]).netloc,
                    "text": entry["text"],
                    "register_labels": labels,
                    "register_scores": scores,
                }

    def _entries(self):
        if self._input_file:
            return load_manifest(Path(self._input_file))
        return chain.from_iterable(
            hub.entries(source["dataset"], source["files"], source["columns"]) for source in self._sources)


if __name__ == "__main__":
    stage = ClassifyStage(**stage_config("classify"))
    stage.run(None, Path("data/classify") / run_name())
