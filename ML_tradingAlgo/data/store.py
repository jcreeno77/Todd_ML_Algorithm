"""S3 Parquet storage layer for the trading-data pipeline.

This module is the **shared contract** the rest of the data pipeline codes
against. It provides an immutable, append-only Parquet warehouse on S3 with
Hive-style partitioning, plus a tiny per-symbol watermark store.

Public surface (do not rename)::

    write_bars(df, table, partition_cols, source=None) -> None
    read_bars(table, symbol=None, date_range=None, source=None) -> pd.DataFrame
    dedupe(df, keys=("symbol", "ts", "source")) -> pd.DataFrame
    get_watermark(table, symbol) -> pd.Timestamp | None
    set_watermark(table, symbol, ts) -> None

Layout on S3::

    s3://{S3_BUCKET}/{S3_PREFIX}/{table}/{col=value}/.../part-{uuid}.parquet
    s3://{S3_BUCKET}/{S3_PREFIX}/_watermarks/{table}/{symbol}.json

Design notes
------------
* Writes are **immutable**: every ``write_bars`` call emits a brand-new file
  with a UUID4 suffix and never overwrites an existing object. Concurrent
  writers to the same partition therefore each land their own file; reads merge
  all files and call :func:`dedupe` to collapse re-ingested rows (keeping the
  row with the latest ``ingested_at``).
* Three bookkeeping columns are auto-injected on write if absent:
  ``ingested_at`` (UTC now), ``schema_version`` (int16 == 1) and ``source``.
* Configuration (``S3_BUCKET`` / ``S3_PREFIX``) is read from the environment at
  call time so tests can monkeypatch it. The project ``config.py`` is **not**
  imported. An optional ``AWS_ENDPOINT_URL`` env var redirects the underlying
  ``s3fs.S3FileSystem`` (used by tests against a mock S3 server).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

__all__ = [
    "write_bars",
    "read_bars",
    "dedupe",
    "get_watermark",
    "set_watermark",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1
_DEDUPE_KEYS = ("symbol", "ts", "source")


# --------------------------------------------------------------------------- #
# config helpers (read env at call time so tests can monkeypatch)
# --------------------------------------------------------------------------- #
def _bucket() -> str:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set")
    return bucket


def _prefix() -> str:
    return os.environ.get("S3_PREFIX", "")


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _fs() -> s3fs.S3FileSystem:
    """Build a fresh S3 filesystem, honouring AWS_ENDPOINT_URL for mocks."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    client_kwargs = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    # skip_instance_cache so monkeypatched env / endpoints are always respected.
    return s3fs.S3FileSystem(skip_instance_cache=True, client_kwargs=client_kwargs)


def _base_path(table: str) -> str:
    prefix = _prefix().strip("/")
    parts = [_bucket()]
    if prefix:
        parts.append(prefix)
    parts.append(table)
    return "/".join(parts)


# --------------------------------------------------------------------------- #
# partition path helpers
# --------------------------------------------------------------------------- #
def _hive_segment(col: str, value) -> str:
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        if isinstance(value, pd.Timestamp):
            value = value.date() if value.tzinfo is None else value.date()
        value = value.isoformat()
    return f"{col}={value}"


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def write_bars(
    df: pd.DataFrame,
    table: str,
    partition_cols: list[str],
    source: str | None = None,
) -> None:
    """Write ``df`` as immutable, UUID-suffixed Parquet files under ``table``.

    Auto-injects ``ingested_at`` / ``schema_version`` / ``source`` if absent.
    One file is written per distinct partition-column combination present in
    ``df``. Existing files are never overwritten.
    """
    if df is None or len(df) == 0:
        return

    df = df.copy()

    # --- source ---------------------------------------------------------- #
    if "source" not in df.columns:
        if source is None:
            raise ValueError(
                "source must be provided (arg) or present as a 'source' column"
            )
        df["source"] = source
    elif source is not None:
        # explicit arg wins / fills any gaps
        df["source"] = df["source"].fillna(source)

    # --- ingested_at ----------------------------------------------------- #
    if "ingested_at" not in df.columns:
        df["ingested_at"] = _utcnow()
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).astype(
        "datetime64[ns, UTC]"
    )

    # --- schema_version -------------------------------------------------- #
    if "schema_version" not in df.columns:
        df["schema_version"] = SCHEMA_VERSION
    df["schema_version"] = df["schema_version"].astype("int16")

    fs = _fs()
    base = _base_path(table)

    if not partition_cols:
        _write_one(fs, base, df)
        return

    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        raise ValueError(f"partition_cols not in dataframe: {missing}")

    for keys, group in df.groupby(partition_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        segs = [_hive_segment(c, v) for c, v in zip(partition_cols, keys)]
        # partition columns are encoded in the path; drop them from the file
        part_df = group.drop(columns=list(partition_cols))
        part_path = "/".join([base, *segs])
        _write_one(fs, part_path, part_df)


def _write_one(fs: s3fs.S3FileSystem, dir_path: str, df: pd.DataFrame) -> None:
    fname = f"part-{uuid.uuid4().hex}.parquet"
    full = f"{dir_path}/{fname}"
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    with fs.open(full, "wb") as f:
        f.write(buf.getvalue())


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def read_bars(
    table: str,
    symbol: str | None = None,
    date_range: tuple | None = None,
    source: str | None = None,
    dedupe_keys: tuple | None = None,
) -> pd.DataFrame:
    """Read the partitioned dataset for ``table`` and return a deduped frame.

    Optional filters: ``symbol``, ``date_range=(start, end)`` (inclusive, vs
    ``session_date``) and ``source``. Returns an empty DataFrame if the table
    or matching partitions are absent.

    Dedupe keys: if ``dedupe_keys`` is given it is used verbatim. Otherwise they
    are chosen from the columns present so we never over-collapse tables that
    lack a ``ts`` column. Bar tables (with ``ts``) dedupe on
    ``(symbol, ts, source)``; tables without ``ts`` but with ``session_date``
    (events, universe, fundamentals snapshots) dedupe on
    ``(symbol, session_date, source)`` — critical for events, where a symbol can
    have one row per session_date and the default ``(symbol, ts, source)`` would
    collapse to ``(symbol, source)`` and silently drop all but one event.
    """
    fs = _fs()
    base = _base_path(table)

    if not fs.exists(base):
        return pd.DataFrame()

    files = fs.glob(f"{base}/**/*.parquet") or fs.glob(f"{base}/*.parquet")
    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        part_kv = _parse_partitions(path, base)
        # partition-key pruning for symbol before reading
        if symbol is not None and "symbol" in part_kv and part_kv["symbol"] != symbol:
            continue
        if source is not None and "source" in part_kv and part_kv["source"] != source:
            continue
        with fs.open(path, "rb") as f:
            tbl = pq.read_table(f)
        pdf = tbl.to_pandas()
        # re-attach partition columns encoded in the path
        for k, v in part_kv.items():
            if k not in pdf.columns:
                pdf[k] = v
        frames.append(pdf)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = _normalize_session_date(df)
    if "ingested_at" in df.columns and len(df):
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).astype(
            "datetime64[ns, UTC]"
        )

    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if source is not None and "source" in df.columns:
        df = df[df["source"] == source]
    if date_range is not None and "session_date" in df.columns:
        start, end = date_range
        start = _to_date(start)
        end = _to_date(end)
        sd = df["session_date"].map(_to_date)
        df = df[(sd >= start) & (sd <= end)]

    df = df.reset_index(drop=True)

    if dedupe_keys is None:
        if "ts" in df.columns:
            dedupe_keys = ("symbol", "ts", "source")
        elif "session_date" in df.columns:
            dedupe_keys = ("symbol", "session_date", "source")
        else:
            dedupe_keys = ("symbol", "source")
    return dedupe(df, keys=dedupe_keys)


def _parse_partitions(path: str, base: str) -> dict:
    """Extract Hive-style ``col=value`` segments between ``base`` and filename."""
    rel = path[len(base):].lstrip("/")
    segs = rel.split("/")[:-1]  # drop filename
    out = {}
    for seg in segs:
        if "=" in seg:
            k, v = seg.split("=", 1)
            out[k] = v
    return out


def _to_date(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _normalize_session_date(df: pd.DataFrame) -> pd.DataFrame:
    if "session_date" in df.columns and len(df):
        df["session_date"] = df["session_date"].map(_to_date)
    return df


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #
def dedupe(df: pd.DataFrame, keys: tuple = _DEDUPE_KEYS) -> pd.DataFrame:
    """Drop duplicate rows on ``keys``, keeping the latest ``ingested_at``.

    Key columns absent from ``df`` are ignored (dedupe on the intersection).
    """
    if df is None or len(df) == 0:
        return df

    present = [k for k in keys if k in df.columns]
    if not present:
        return df.reset_index(drop=True)

    if "ingested_at" in df.columns:
        order = df["ingested_at"]
        idx = df.assign(_ia=order).sort_values("_ia", kind="stable")
        idx = idx.drop_duplicates(subset=present, keep="last")
        idx = idx.drop(columns="_ia")
        return idx.sort_index().reset_index(drop=True)

    return df.drop_duplicates(subset=present, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# watermarks
# --------------------------------------------------------------------------- #
def _watermark_path(table: str, symbol: str) -> str:
    prefix = _prefix().strip("/")
    parts = [_bucket()]
    if prefix:
        parts.append(prefix)
    parts += ["_watermarks", table, f"{symbol}.json"]
    return "/".join(parts)


def get_watermark(table: str, symbol: str) -> "pd.Timestamp | None":
    """Return the per-symbol watermark as a UTC Timestamp, or None if absent."""
    fs = _fs()
    path = _watermark_path(table, symbol)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as f:
        obj = json.loads(f.read().decode("utf-8"))
    last_ts = obj.get("last_ts")
    if last_ts is None:
        return None
    return pd.Timestamp(last_ts).tz_convert("UTC") if pd.Timestamp(last_ts).tzinfo else pd.Timestamp(last_ts, tz="UTC")


def set_watermark(table: str, symbol: str, ts) -> None:
    """Persist the per-symbol watermark JSON object."""
    fs = _fs()
    path = _watermark_path(table, symbol)
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    obj = {
        "symbol": symbol,
        "last_ts": ts.isoformat(),
        "updated_at": _utcnow().isoformat(),
    }
    with fs.open(path, "wb") as f:
        f.write(json.dumps(obj).encode("utf-8"))
