"""Run the Kurmanji instruction-data pipeline defined in config.yml."""
import argparse
import sys
from pathlib import Path

import yaml

from kurdish_ins import logs
from kurdish_ins.classify import ClassifyStage
from kurdish_ins.stage import Stage, run_name, stage_config

REGISTRY: dict[str, type[Stage]] = {
    "classify": ClassifyStage,
}


def load_config(config_path: Path) -> list[dict]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [{"name": name, **raw[name]} for name in raw["stages"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kurmanji instruction-data pipeline.")
    parser.add_argument(
        "--start-from",
        metavar="STAGE",
        help="Name of the stage to start from (skips all earlier stages).",
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        help="Run a single stage and exit.",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        metavar="FILE",
        help="Path to the pipeline YAML config (default: config.yml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also log the successes.",
    )
    args = parser.parse_args()

    logs.configure(args.debug)

    if args.start_from and args.only:
        print("Error: --start-from and --only are mutually exclusive.")
        sys.exit(1)

    stage_cfgs = load_config(Path(args.config))
    stage_names = [s["name"] for s in stage_cfgs]

    if args.only:
        if args.only not in stage_names:
            print(f"Error: unknown stage '{args.only}'. Valid stages: {', '.join(stage_names)}")
            sys.exit(1)
        stage_cfgs = [next(c for c in stage_cfgs if c["name"] == args.only)]
    elif args.start_from:
        if args.start_from not in stage_names:
            print(f"Error: unknown stage '{args.start_from}'. Valid stages: {', '.join(stage_names)}")
            sys.exit(1)
        stage_cfgs = stage_cfgs[stage_names.index(args.start_from):]

    data_root = Path("data")
    run = run_name(Path(args.config))

    for cfg in stage_cfgs:
        stage: Stage = REGISTRY[cfg["name"]](**stage_config(cfg["name"], Path(args.config)))
        output_dir = data_root / stage.name / run
        prev_idx = stage_names.index(stage.name) - 1
        input_dir = data_root / stage_names[prev_idx] / run if prev_idx >= 0 else None
        print(f"\n=== Stage: {stage.name} ({run}) — {cfg['description']} ===")
        stage.run(input_dir=input_dir, output_dir=output_dir)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
