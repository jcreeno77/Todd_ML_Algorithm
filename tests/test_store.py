"""Tests for the S3 Parquet storage layer (``ML_tradingAlgo.data.store``).

Moto approach that works here
-----------------------------
moto 5.x's ``@mock_aws`` decorator patches *botocore*, but s3fs/pyarrow open
their own aiobotocore sessions that the in-process patch does **not** reliably
intercept (you get real-AWS connection attempts / NoCredentials). The robust
approach for s3fs + pyarrow is therefore **moto server mode**:

  * Start ``ThreadedMotoServer`` on an ephemeral port (a real HTTP endpoint).
  * Point both boto3 (bucket creation) and s3fs at that endpoint via
    ``endpoint_url`` / ``AWS_ENDPOINT_URL``.
  * ``store.py`` honours an optional ``AWS_ENDPOINT_URL`` env var when it builds
    its ``s3fs.S3FileSystem`` so the test can redirect it at the mock server.

This avoids the finicky decorator-interception problem entirely.
"""

from __future__ import annotations

import datetime as dt

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import store

BUCKET = "test-bars-bucket"
PREFIX = "warehouse"


@pytest.fixture(scope="module")
def moto_server():
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    yield endpoint
    server.stop()


@pytest.fixture
def s3_env(moto_server, monkeypatch):
    endpoint = moto_server
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("S3_PREFIX", PREFIX)

    s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
    # Fresh bucket per test to keep them isolated.
    try:
        s3.create_bucket(Bucket=BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Clear any leftover objects from a previous test in the module.
    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=obj["Key"])

    # s3fs caches filesystem instances; clear so each test sees a fresh one.
    import s3fs

    s3fs.S3FileSystem.clear_instance_cache()
    yield endpoint


def _sample_df(symbol="AAA", n=3, session_date=dt.date(2026, 1, 2)):
    base = pd.Timestamp("2026-01-02 14:30", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "session_date": [session_date] * n,
            "open": [1.0 + i for i in range(n)],
            "close": [1.5 + i for i in range(n)],
        }
    )


# --------------------------------------------------------------------------- #
# write -> read round-trip + auto-injection
# --------------------------------------------------------------------------- #
def test_write_read_roundtrip(s3_env):
    df = _sample_df()
    store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source="schwab")

    out = store.read_bars("bars_1m")
    assert len(out) == 3
    assert set(out["symbol"]) == {"AAA"}


def test_auto_injection_columns(s3_env):
    df = _sample_df()
    assert "ingested_at" not in df.columns
    store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source="schwab")

    out = store.read_bars("bars_1m")
    assert "ingested_at" in out.columns
    assert "schema_version" in out.columns
    assert "source" in out.columns
    assert set(out["source"]) == {"schwab"}
    assert set(out["schema_version"]) == {1}
    assert str(out["schema_version"].dtype) == "int16"
    assert str(out["ingested_at"].dtype) == "datetime64[ns, UTC]"


def test_source_from_column_when_arg_none(s3_env):
    df = _sample_df()
    df["source"] = "polygon"
    store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source=None)
    out = store.read_bars("bars_1m")
    assert set(out["source"]) == {"polygon"}


def test_missing_source_raises(s3_env):
    df = _sample_df()
    with pytest.raises(ValueError):
        store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source=None)


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #
def test_dedupe_keeps_latest_ingested_at():
    ts = pd.Timestamp("2026-01-02 14:30", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "ts": [ts, ts],
            "source": ["schwab", "schwab"],
            "close": [10.0, 99.0],
            "ingested_at": [
                pd.Timestamp("2026-01-02 00:00", tz="UTC"),
                pd.Timestamp("2026-01-03 00:00", tz="UTC"),
            ],
        }
    )
    out = store.dedupe(df)
    assert len(out) == 1
    assert out.iloc[0]["close"] == 99.0


def test_dedupe_handles_missing_key_columns():
    # "source" key absent -> dedupe on intersection (symbol, ts)
    ts = pd.Timestamp("2026-01-02 14:30", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "ts": [ts, ts],
            "close": [1.0, 2.0],
            "ingested_at": [
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-01-05", tz="UTC"),
            ],
        }
    )
    out = store.dedupe(df)
    assert len(out) == 1
    assert out.iloc[0]["close"] == 2.0


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def test_filtering_symbol_date_source(s3_env):
    store.write_bars(
        _sample_df("AAA", session_date=dt.date(2026, 1, 2)),
        table="bars_1m",
        partition_cols=["symbol", "session_date"],
        source="schwab",
    )
    store.write_bars(
        _sample_df("BBB", session_date=dt.date(2026, 1, 3)),
        table="bars_1m",
        partition_cols=["symbol", "session_date"],
        source="schwab",
    )
    store.write_bars(
        _sample_df("AAA", session_date=dt.date(2026, 1, 10)),
        table="bars_1m",
        partition_cols=["symbol", "session_date"],
        source="polygon",
    )

    # symbol filter
    out = store.read_bars("bars_1m", symbol="AAA")
    assert set(out["symbol"]) == {"AAA"}

    # source filter
    out = store.read_bars("bars_1m", source="polygon")
    assert set(out["source"]) == {"polygon"}

    # date_range filter (inclusive, against session_date)
    out = store.read_bars(
        "bars_1m",
        symbol="AAA",
        date_range=(dt.date(2026, 1, 1), dt.date(2026, 1, 5)),
    )
    assert set(out["symbol"]) == {"AAA"}
    sds = {pd.Timestamp(x).date() if not isinstance(x, dt.date) else x for x in out["session_date"]}
    assert all(dt.date(2026, 1, 1) <= s <= dt.date(2026, 1, 5) for s in sds)
    # the 2026-01-10 AAA row must be excluded
    assert dt.date(2026, 1, 10) not in sds


def test_read_absent_table_returns_empty(s3_env):
    out = store.read_bars("does_not_exist")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# watermarks
# --------------------------------------------------------------------------- #
def test_watermark_roundtrip(s3_env):
    assert store.get_watermark("bars_1m", "AAA") is None

    ts = pd.Timestamp("2026-01-02 20:00", tz="UTC")
    store.set_watermark("bars_1m", "AAA", ts)

    got = store.get_watermark("bars_1m", "AAA")
    assert got is not None
    assert got == ts
    assert got.tzinfo is not None


# --------------------------------------------------------------------------- #
# concurrent writers -> multiple files, read merges + dedupes
# --------------------------------------------------------------------------- #
def test_concurrent_writers_merge_and_dedupe(s3_env, monkeypatch):
    df = _sample_df("AAA")

    # First writer (older ingested_at)
    monkeypatch.setattr(
        store, "_utcnow", lambda: pd.Timestamp("2026-01-02 00:00", tz="UTC")
    )
    store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source="schwab")

    # Second writer to the SAME partition with newer ingested_at + new value
    df2 = df.copy()
    df2["close"] = df2["close"] + 100.0
    monkeypatch.setattr(
        store, "_utcnow", lambda: pd.Timestamp("2026-01-03 00:00", tz="UTC")
    )
    store.write_bars(df2, table="bars_1m", partition_cols=["symbol"], source="schwab")

    import s3fs

    fs = s3fs.S3FileSystem.current()
    files = fs.glob(f"{BUCKET}/{PREFIX}/bars_1m/symbol=AAA/*.parquet")
    assert len(files) == 2  # two immutable files, nothing overwritten

    out = store.read_bars("bars_1m")
    assert len(out) == 3  # deduped back to 3 rows
    # latest ingested_at wins -> +100 values
    assert sorted(out["close"]) == [101.5, 102.5, 103.5]


# --------------------------------------------------------------------------- #
# regression: tables without a `ts` column (events) must dedupe on
# (symbol, session_date, source), NOT collapse to (symbol, source).
# A symbol that gaps up on multiple days must keep one event PER session_date.
# --------------------------------------------------------------------------- #
def test_events_no_ts_keeps_one_row_per_session_date(s3_env):
    ev1 = pd.DataFrame(
        {"symbol": ["AAA"], "session_date": [dt.date(2026, 1, 2)], "gap_pct": [0.30]}
    )
    ev2 = pd.DataFrame(
        {"symbol": ["AAA"], "session_date": [dt.date(2026, 3, 10)], "gap_pct": [0.42]}
    )
    store.write_bars(ev1, table="events", partition_cols=["session_date"], source="collector")
    store.write_bars(ev2, table="events", partition_cols=["session_date"], source="collector")

    out = store.read_bars("events")
    # Before the fix this returned 1 row (collapsed on (symbol, source)).
    assert len(out) == 2
    assert set(out["session_date"]) == {dt.date(2026, 1, 2), dt.date(2026, 3, 10)}


def test_events_rerun_same_session_dedupes_to_latest(s3_env, monkeypatch):
    ev = pd.DataFrame(
        {"symbol": ["AAA"], "session_date": [dt.date(2026, 1, 2)], "gap_pct": [0.30]}
    )
    monkeypatch.setattr(store, "_utcnow", lambda: pd.Timestamp("2026-01-02", tz="UTC"))
    store.write_bars(ev, table="events", partition_cols=["session_date"], source="collector")
    ev2 = ev.copy()
    ev2["gap_pct"] = 0.35  # corrected detection on rerun
    monkeypatch.setattr(store, "_utcnow", lambda: pd.Timestamp("2026-01-03", tz="UTC"))
    store.write_bars(ev2, table="events", partition_cols=["session_date"], source="collector")

    out = store.read_bars("events")
    assert len(out) == 1  # same (symbol, session_date) collapses
    assert out.iloc[0]["gap_pct"] == 0.35  # latest ingested_at wins


def test_dedupe_keys_override(s3_env):
    df = _sample_df("AAA")
    store.write_bars(df, table="bars_1m", partition_cols=["symbol"], source="schwab")
    # Force collapse to one row per symbol via explicit override.
    out = store.read_bars("bars_1m", dedupe_keys=("symbol", "source"))
    assert len(out) == 1
