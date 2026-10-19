"""Logging setup, shared by pipeline.py and the stage scripts run directly."""

import logging
import sys


# These log every HTTP request and cache lookup, which would bury our own output.
_NOISY = ("httpx", "httpx2", "httpcore", "urllib3", "filelock", "huggingface_hub", "fsspec")


def configure(debug: bool = False) -> None:
    """Sends logs to stdout. INFO reports what went wrong; `debug` adds the successes too."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
