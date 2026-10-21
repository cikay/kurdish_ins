import json
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

CONFIG_FILE = Path("config.yml")


class Stage(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, input_dir: Path | None, output_dir: Path) -> None: ...


def stage_config(name: str, config_path: Path = CONFIG_FILE) -> dict:
    """The tuning knobs `config.yml` gives one stage, ready to pass as keyword arguments.

    Every knob lives there and nowhere else — no stage defines a default in Python, so
    running a stage on its own and running it through `pipeline.py` cannot disagree. The
    exceptions are `classify.input_file` and `instruction-gen.input_manifest`, which may be
    left out (commented out) to mean
    `null`. `description` is documentation, not a knob, so it is dropped.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {key: value for key, value in raw[name].items() if key != "description"}


def run_name(config_path: Path = CONFIG_FILE) -> str:
    """The folder every stage of this run writes into, under `data/<stage>/`.

    `full` when `classify` reads every source, otherwise the stem of its
    `input_file` (`data/sample.jsonl` → `sample`), so a sample run and a full run never
    overwrite each other's manifests.
    """
    input_file = stage_config("classify", config_path).get("input_file")
    if not input_file:
        return "full"
    name = Path(input_file).stem
    if name == "full":
        raise ValueError(f"input_file {input_file} would share the folder of the full run; rename it")
    return name


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
