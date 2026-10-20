"""Stage — Read the Kurmanji datasets and label each entry's web register.

This stage only labels; it drops nothing. Deciding which registers become instruction
pairs is a later stage's job, so changing that rule never means classifying again.
"""

import json
from collections import Counter
from itertools import chain, islice
from pathlib import Path
from urllib.parse import urlparse

from kurdish_ins import hub
from kurdish_ins.register_classifier import RegisterClassifier
from kurdish_ins.stage import Stage, load_manifest, run_name, stage_config


class ClassifyStage(Stage):
    name = "classify"

    def __init__(self, sources: list[dict], input_file: str | None, threshold: float,
                 batch_size: int, max_length: int, stride: int):
        self._sources = sources
        self._input_file = input_file
        self._threshold = threshold
        self._batch_size = batch_size
        self._max_length = max_length
        self._stride = stride

    def run(self, input_dir: Path | None, output_dir: Path) -> None:
        source = self._input_file or ", ".join(source["dataset"] for source in self._sources)
        print(f"Classifying the entries of {source} "
              f"(threshold={self._threshold}, batch_size={self._batch_size})")
        classifier = RegisterClassifier(self._threshold, self._max_length, self._stride, self._batch_size)

        output_dir.mkdir(parents=True, exist_ok=True)
        registers = Counter()
        count = 0

        with open(output_dir / "manifest.jsonl", "w", encoding="utf-8") as mf:
            for record in self.classify(self._entries(), classifier):
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                registers.update(label for label in record["register_labels"] if label.isupper())
                if not record["register_labels"]:
                    registers["(none)"] += 1
                count += 1
                print(f"  [{count}]", end="\r")

        print(f"\nDone — {count} entries classified")
        print(f"       main registers: {', '.join(f'{n} {r}' for r, n in registers.most_common())}")
        print(f"       → {output_dir}/manifest.jsonl")

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
