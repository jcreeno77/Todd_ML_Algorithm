"""Tests for the live-loop tee (``ML_tradingAlgo.data.tee``).

Uses moto server mode (same approach as ``tests/test_store.py``) so that s3fs /
pyarrow talk to a real local HTTP endpoint rather than the unreliable in-process
botocore patch.

The tee touches the LIVE TRADING hot path, so the key behaviours under test are:
  * ``put_candle`` is strictly non-blocking (returns immediately even with a
    deliberately slow injected writer),
  * batched flushes land correct rows in ``bars_minute`` with the right source,
  * ``stop(drain=True)`` flushes the remainder,
  * a full queue drops candles instead of blocking,
  * ``update_state`` writes a readable JSON snapshot to ``live/state/{id}.json``,
  * a never-before-seen symbol triggers ``gainers.record_observation`` exactly
    once (monkeypatched).
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import store, tee

BUCKET = "test-tee-bucket"
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
    try:
        s3.create_bucket(Bucket=BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=obj["Key"])

    import s3fs

    s3fs.S3FileSystem.clear_instance_cache()
    yield endpoint


SESSION = dt.date(2026, 1, 2)


def _candle(o=1.0, h=2.0, low=0.5, c=1.5, v=1000):
    return [o, h, low, c, v]


@pytest.fixture(autouse=True)
def _no_real_gainers(monkeypatch):
    """Default: stub gainers so tests never hit S3 from the universe writer
    unless a test explicitly wants to observe it."""
    monkeypatch.setattr(tee.gainers, "record_observation", lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# put_candle is non-blocking
# --------------------------------------------------------------------------- #
def test_put_candle_is_non_blocking(monkeypatch):
    """Even with an artificially slow writer, put_candle must return ~instantly."""
    slow_started = threading.Event()

    def slow_write(*args, **kwargs):
        slow_started.set()
        time.sleep(5.0)

    monkeypatch.setattr(tee, "write_bars", slow_write)

    t = tee.LiveTee("99", flush_every=1, flush_interval=0.05)
    t.start()
    try:
        start = time.perf_counter()
        for _ in range(50):
            t.put_candle("AAA", _candle(), "1min", SESSION)
        elapsed = time.perf_counter() - start
        # 50 enqueues should be effectively instantaneous; the slow writer runs
        # on the daemon thread and must not stall the producer.
        assert elapsed < 0.5
    finally:
        # Don't drain (writer sleeps 5s); just let the daemon die.
        t._running = False


def test_put_candle_never_raises_on_full_queue():
    """A full queue drops candles rather than blocking or raising."""
    t = tee.LiveTee("1", flush_every=1000, flush_interval=1000, max_queue=5)
    # Never start the worker -> queue fills and stays full.
    for _ in range(100):
        t.put_candle("AAA", _candle(), "1min", SESSION)  # must not raise/block
    # Queue capped at its maxsize; excess silently dropped.
    assert t._q.qsize() <= 5


# --------------------------------------------------------------------------- #
# batched flush writes correct rows
# --------------------------------------------------------------------------- #
def _read_until(table, expected, timeout=10.0):
    deadline = time.time() + timeout
    out = pd.DataFrame()
    while time.time() < deadline:
        out = store.read_bars(table)
        if len(out) >= expected:
            return out
        time.sleep(0.1)
    return out


def test_batched_flush_writes_bars_minute(s3_env):
    t = tee.LiveTee("1", flush_every=3, flush_interval=30.0)
    t.start()
    try:
        for i in range(3):
            t.put_candle("AAA", _candle(c=1.5 + i, v=100 + i), "1min", SESSION)
        out = _read_until("bars_minute", 3)
    finally:
        t.stop(drain=True)

    assert len(out) == 3
    assert set(out["symbol"]) == {"AAA"}
    assert set(out["source"]) == {"live_tick_agg"}
    assert set(out["resolution"]) == {"1min"}
    assert sorted(out["close"]) == [1.5, 2.5, 3.5]
    assert "ts" in out.columns
    sds = {store._to_date(x) for x in out["session_date"]}
    assert sds == {SESSION}


def test_five_min_goes_to_bars_5min(s3_env):
    t = tee.LiveTee("1", flush_every=2, flush_interval=30.0)
    t.start()
    try:
        t.put_candle("BBB", _candle(), "5min", SESSION)
        t.put_candle("BBB", _candle(), "5min", SESSION)
        out = _read_until("bars_5min", 2)
    finally:
        t.stop(drain=True)

    assert len(out) == 2
    assert set(out["resolution"]) == {"5min"}
    assert set(out["source"]) == {"live_tick_agg"}
    # nothing leaked into the 1-min table
    assert len(store.read_bars("bars_minute")) == 0


# --------------------------------------------------------------------------- #
# stop(drain=True) flushes remainder below the batch threshold
# --------------------------------------------------------------------------- #
def test_stop_drain_flushes_remainder(s3_env):
    # flush_every high so nothing flushes until stop()
    t = tee.LiveTee("1", flush_every=100, flush_interval=30.0)
    t.start()
    t.put_candle("CCC", _candle(c=7.0), "1min", SESSION)
    t.put_candle("CCC", _candle(c=8.0), "1min", SESSION)
    t.stop(drain=True)

    out = store.read_bars("bars_minute")
    assert len(out) == 2
    assert sorted(out["close"]) == [7.0, 8.0]


# --------------------------------------------------------------------------- #
# update_state -> live/state/{id}.json
# --------------------------------------------------------------------------- #
def test_update_state_writes_readable_json(s3_env):
    import s3fs

    t = tee.LiveTee("2", flush_every=1, flush_interval=0.05)
    t.start()
    try:
        t.update_state(
            {
                "watched": ["AAA", "BBB"],
                "positions": {"AAA": 100},
                "last_signal": "buy",
                "last_price": 12.34,
                "pnl": 56.78,
            }
        )
        fs = s3fs.S3FileSystem(
            skip_instance_cache=True,
            client_kwargs={"endpoint_url": s3_env},
        )
        path = f"{BUCKET}/{PREFIX}/live/state/2.json"
        deadline = time.time() + 10.0
        while time.time() < deadline and not fs.exists(path):
            time.sleep(0.1)
        assert fs.exists(path)
        with fs.open(path, "rb") as f:
            obj = json.loads(f.read().decode("utf-8"))
    finally:
        t.stop(drain=True)

    assert obj["watched"] == ["AAA", "BBB"]
    assert obj["last_signal"] == "buy"
    assert obj["last_price"] == 12.34
    assert "updated_at" in obj


# --------------------------------------------------------------------------- #
# new symbol triggers gainers.record_observation exactly once
# --------------------------------------------------------------------------- #
def test_new_symbol_triggers_record_observation(s3_env, monkeypatch):
    calls = []
    done = threading.Event()

    def fake_record(symbol, ctx=None):
        calls.append((symbol, ctx))
        done.set()

    monkeypatch.setattr(tee.gainers, "record_observation", fake_record)

    t = tee.LiveTee("1", flush_every=10, flush_interval=0.05)
    t.start()
    try:
        # same symbol multiple times -> recorded once
        for _ in range(4):
            t.put_candle("DDD", _candle(), "1min", SESSION)
        assert done.wait(timeout=10.0)
        # give the worker a moment to process the rest
        time.sleep(0.3)
    finally:
        t.stop(drain=True)

    ddd_calls = [c for c in calls if c[0] == "DDD"]
    assert len(ddd_calls) == 1
    sym, ctx = ddd_calls[0]
    assert sym == "DDD"
    assert ctx["observed_by"] == "live_scanner"
    assert ctx["session_date"] == SESSION
