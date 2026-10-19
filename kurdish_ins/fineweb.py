"""FineWeb2 entries, read from parquet files downloaded to the Hugging Face cache.

Not `load_dataset(..., streaming=True)`: a stream not read to the end hangs the process
at exit. pyarrow reads the next chunk over HTTP on its own IO thread, Python kills that
thread at shutdown in the middle of the request, and arrow's thread pool then waits for
it forever (seen with `sample`: `exit` → `ThreadPool::Shutdown` → `condition_variable::wait`).
Closing the stream or turning off pre-buffering does not stop a read already in flight.
A local file leaves no read in flight. `kmr_Latn` is one 562 MB file, downloaded once.
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

COLUMNS = ["id", "url", "text"]


def entries(dataset: str, subset: str):
    """Yield {id, url, text} for every train entry of a subset, in the order streaming gives."""
    root = snapshot_download(dataset, repo_type="dataset", allow_patterns=f"data/{subset}/train/*.parquet")
    for path in sorted(Path(root, "data", subset, "train").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS):
            yield from batch.to_pylist()
