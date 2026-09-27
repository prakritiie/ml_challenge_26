"""Memory-conscious I/O helpers built on DuckDB + Arrow.

Every function here streams data rather than materializing a full source
file (up to ~5.3M rows) into pandas. Two access patterns are supported:

- `iter_record_batches`: pulls the raw TSV through DuckDB and yields it as a
  sequence of small Arrow RecordBatches (bounded by `batch_size` rows each),
  suitable for a full-file, constant-memory pass.
- `sample_rows`: pulls a small reservoir sample (default ~1000 rows) for
  quick inspection/testing without touching the rest of the file.
"""

from pathlib import Path
from typing import Iterator, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import config


def get_connection(threads: int = 4) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    return con


def get_configured_connection() -> duckdb.DuckDBPyConnection:
    """Connection for the heavier blocking-stage joins: bounded memory with
    disk spilling enabled, so a large equi-join never OOMs this machine --
    it just gets slower and spills to config.DUCKDB_TEMP_DIR instead."""
    config.DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config.DUCKDB_THREADS}")
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{str(config.DUCKDB_TEMP_DIR).replace(chr(39), chr(39)*2)}'")
    return con


def parquet_sql(path: Path) -> str:
    return f"read_parquet('{_quote(path)}')"


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_csv_sql(path: Path) -> str:
    opts = config.CSV_READ_OPTIONS
    return (
        f"SELECT * FROM read_csv('{_quote(path)}', "
        f"delim='{opts['delim']}', header={str(opts['header']).lower()}, "
        f"quote='{opts['quote']}', escape='{opts['escape']}', "
        f"all_varchar={str(opts['all_varchar']).lower()})"
    )


def iter_record_batches(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    batch_size: int = config.BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Stream a raw source TSV as Arrow record batches of `batch_size` rows.

    Uses DuckDB's Arrow reader, which pulls rows out of the CSV lazily
    batch-by-batch -- the full file is never materialized at once.
    """
    relation = con.sql(_read_csv_sql(path))
    reader = relation.fetch_arrow_reader(batch_size=batch_size)
    for batch in reader:
        yield batch


def sample_rows(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    n: int = 1000,
    seed: Optional[int] = 42,
) -> pa.Table:
    """Reservoir-sample ~n rows from a source TSV without a full sort/scan
    into memory (DuckDB's reservoir sampling is a single streaming pass)."""
    sample_clause = f"(reservoir, {seed})" if seed is not None else "(reservoir)"
    query = f"{_read_csv_sql(path)} USING SAMPLE {n} {sample_clause}"
    result = con.sql(query).arrow()
    return result.read_all() if isinstance(result, pa.RecordBatchReader) else result


def write_batches_to_parquet(batches: Iterator[pa.RecordBatch], output_path: Path) -> int:
    """Consume an iterator of RecordBatches and stream them to a single
    Parquet file, one batch at a time. Total rows written is returned."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    try:
        for batch in batches:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def run_sql(con: duckdb.DuckDBPyConnection, path: Path, extra_sql: str = "") -> duckdb.DuckDBPyRelation:
    """Return a DuckDB relation over a raw source file, for ad-hoc querying
    (e.g. the pattern-search demos), without loading it into pandas."""
    return con.sql(_read_csv_sql(path) + extra_sql)
