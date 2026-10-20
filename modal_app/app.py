"""
Modal entry point for the pipeline. The Modal function is a thin wrapper around the same
stage that `pipeline.py` runs locally.

Run it from the repository root, as a module, so `kurdish_ins` imports as it does locally:

    modal run --detach -m modal_app.app::classify
    modal volume get kurdish-ins /classify/<run> data/modal/

It reads `classify.input_file` like a local run. Commented out or `null`, it classifies
every source into `data/classify/full/`; a file such as `data/s5.jsonl` is uploaded to the
volume and classified into `data/classify/s5/`. The volume is mounted at `/root/data` and
the code runs in `/root`, so `data/s5.jsonl` means the same file here as on the Mac.

The other knobs are `config.yml`'s `classify` block, copied into the image, so a run here
and a local run cannot tune differently. Where it runs — GPU, timeout and the GPU's batch
size — is the `modal` block of the same file.
"""

from pathlib import Path

import modal
import yaml

from kurdish_ins.stage import stage_config

VOLUME_NAME = "kurdish-ins"
VOLUME_MOUNT_PATH = "/root/data"

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPOSITORY_ROOT / "config.yml" if modal.is_local() else Path("/root/config.yml")

app = modal.App("kurdish-ins")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# The versions Pipfile.lock resolved to, so the model runs on the code it was checked with
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.1",
        "tokenizers==0.23.3",
        "safetensors==0.8.0",
        "huggingface-hub==2.2.0",
        "pyarrow==26.0.0",
        "pyyaml==6.0.3",
    )
    # The model (2 GB) and the datasets' parquet files (627 MB) are downloaded on the first run
    # and kept on the volume, not in the image
    .env({"HF_HOME": f"{VOLUME_MOUNT_PATH}/huggingface"})
    .add_local_python_source("kurdish_ins")
    .add_local_file(REPOSITORY_ROOT / "config.yml", "/root/config.yml")
)


def modal_settings(stage: str) -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)["modal"][stage]


# GPU and timeout are not set here: they come from the `modal` block of config.yml and are
# applied per call with `with_options` in the entry point below
@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
def classify_stage(batch_size: int) -> None:
    from kurdish_ins.classify import ClassifyStage
    from kurdish_ins.stage import run_name, stage_config

    stage = ClassifyStage(**{**stage_config("classify"), "batch_size": batch_size})
    stage.run(None, Path("data") / "classify" / run_name())
    volume.commit()


@app.local_entrypoint()
def classify():
    input_file = stage_config("classify", CONFIG_FILE).get("input_file")
    if input_file:
        # data/s5.jsonl on the Mac → /s5.jsonl on the volume → data/s5.jsonl on Modal
        with volume.batch_upload(force=True) as batch:
            batch.put_file(REPOSITORY_ROOT / input_file, f"/{Path(input_file).relative_to('data')}")
    settings = modal_settings("classify")
    classify_stage.with_options(
        gpu=settings["gpu"],
        timeout=settings["timeout_minutes"] * 60,
    ).remote(settings["batch_size"])
