"""Dataset entries, read from parquet files downloaded to the Hugging Face cache.

Not `load_dataset(..., streaming=True)`: a stream not read to the end hangs the process
at exit. pyarrow reads the next chunk over HTTP on its own IO thread, Python kills that
thread at shutdown in the middle of the request, and arrow's thread pool then waits for
it forever (seen with `sample`: `exit` → `ThreadPool::Shutdown` → `condition_variable::wait`).
Closing the stream or turning off pre-buffering does not stop a read already in flight.
A local file leaves no read in flight. FineWeb2's `kmr_Latn` is one 562 MB file, downloaded once.
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def entries(dataset: str, files: str, columns: dict[str, str]):
    """Yield {source, id, url, text} for every row of the dataset's parquet files.

    `files` is a glob inside the dataset repository, e.g. `data/kmr_Latn/train/*.parquet`.
    `columns` maps `id`, `url` and `text` to the dataset's own column names, e.g.
    `{id: tez_id, url: url, text: text}`. `source` is the dataset id.
    """
    root = snapshot_download(dataset, repo_type="dataset", allow_patterns=files)
    for path in sorted(Path(root).glob(files)):
        for batch in pq.ParquetFile(path).iter_batches(columns=list(columns.values())):
            for row in batch.to_pylist():
                yield {"source": dataset, **{key: row[column] for key, column in columns.items()}}
