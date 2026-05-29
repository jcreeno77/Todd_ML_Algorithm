"""Tests for the reconciliation job (``ML_tradingAlgo.data.reconcile``).

Reconciliation pulls the authoritative ``schwab_pricehistory`` minute bars for a
session whose lower-fidelity ``live_tick_agg`` bars were teed during the live
loop, writes the truth bars alongside, and logs how far the live bars diverged.

Like ``tests/test_store.py`` these run against **moto server mode** (a real HTTP
endpoint) so s3fs + pyarrow are exercised end-to-end. ``schwab_client`` is
monkeypatched so no network / credentials are needed.
"""

from __future__ import annotations

import datetime as dt

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import reconcile, store

BUCKET = "test-reconcile-bucket"
PREFIX = "warehouse"


# --------------------------------------------------------------------------- #
# moto server fixture (copied from tests/test_store.py)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# synthetic bar builders
# --------------------------------------------------------------------------- #
SESSION = dt.date(2026, 5, 20)


def _live_bars(symbol, closes, volumes, start="2026-05-20 14:30"):
    """Build a synthetic ``live_tick_agg`` minute frame for one symbol."""
    base = pd.Timestamp(start, tz="UTC")
    n = len(closes)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "session_date": [SESSION] * n,
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": list(closes),
            "volume": list(volumes),
        }
    )


def _truth_bars(symbol, closes, volumes, start="2026-05-20 14:30"):
    """Build a synthetic ``schwab_pricehistory`` frame (client output shape)."""
    base = pd.Timestamp(start, tz="UTC")
    n = len(closes)
    return pd.DataFrame(
        {
            "ts": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "session_date": [SESSION] * n,
            "symbol": [symbol] * n,
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": list(closes),
            "volume": list(volumes),
            "vwap": [float("nan")] * n,
            "trade_count": [pd.NA] * n,
            "source": ["schwab_pricehistory"] * n,
        }
    )


# --------------------------------------------------------------------------- #
# core behaviour
# --------------------------------------------------------------------------- #
def test_reconcile_writes_truth_and_computes_metrics(s3_env, monkeypatch):
    # Seed live bars for one symbol: 3 minutes.
    store.write_bars(
        _live_bars("AAA", closes=[10.0, 11.0, 12.0], volumes=[100, 200, 300]),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
        source="live_tick_agg",
    )

    # Truth bars: same 3 ts but slightly different close/volume, plus 1 extra ts.
    truth = _truth_bars(
        "AAA",
        closes=[10.5, 11.0, 12.4, 13.0],
        volumes=[110, 200, 330, 400],
    )

    calls = []

    def fake_get_minute_bars(symbol, start, end, extended_hours=True):
        calls.append((symbol, start, end, extended_hours))
        return truth

    monkeypatch.setattr(reconcile.schwab_client, "get_minute_bars", fake_get_minute_bars)

    summary = reconcile.reconcile_session(SESSION)

    # Client was called once for the single symbol, extended hours on.
    assert len(calls) == 1
    assert calls[0][0] == "AAA"
    assert calls[0][3] is True

    # Truth bars were written and are readable back filtered by source.
    back = store.read_bars(
        "bars_minute",
        date_range=(SESSION, SESSION),
        source="schwab_pricehistory",
    )
    assert len(back) == 4
    assert set(back["symbol"]) == {"AAA"}

    # Diff metrics. Matched ts: 14:30, 14:31, 14:32 (truth 14:33 is extra).
    #   close abs errors: |10-10.5|, |11-11|, |12-12.4| = 0.5, 0.0, 0.4
    #   volume abs errors: |100-110|, |200-200|, |300-330| = 10, 0, 30
    assert summary["session_date"] == SESSION
    assert summary["symbols"] == ["AAA"]
    assert summary["n_live_bars"] == 3
    assert summary["n_truth_bars"] == 4
    assert summary["n_matched"] == 3
    assert summary["close_mae"] == pytest.approx((0.5 + 0.0 + 0.4) / 3)
    assert summary["volume_mae"] == pytest.approx((10 + 0 + 30) / 3)


def test_reconcile_multiple_symbols(s3_env, monkeypatch):
    store.write_bars(
        _live_bars("AAA", closes=[10.0, 11.0], volumes=[100, 200]),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
        source="live_tick_agg",
    )
    store.write_bars(
        _live_bars("BBB", closes=[5.0, 6.0], volumes=[50, 60]),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
        source="live_tick_agg",
    )

    truth = {
        "AAA": _truth_bars("AAA", closes=[10.0, 11.0], volumes=[100, 200]),
        "BBB": _truth_bars("BBB", closes=[5.2, 6.0], volumes=[50, 60]),
    }

    def fake_get_minute_bars(symbol, start, end, extended_hours=True):
        return truth[symbol]

    monkeypatch.setattr(reconcile.schwab_client, "get_minute_bars", fake_get_minute_bars)

    summary = reconcile.reconcile_session(SESSION)

    assert summary["symbols"] == ["AAA", "BBB"]
    assert summary["n_live_bars"] == 4
    assert summary["n_truth_bars"] == 4
    assert summary["n_matched"] == 4
    # Only BBB's first bar differs: close err 0.2 over 4 matched bars.
    assert summary["close_mae"] == pytest.approx(0.2 / 4)
    assert summary["volume_mae"] == pytest.approx(0.0)

    back = store.read_bars(
        "bars_minute", date_range=(SESSION, SESSION), source="schwab_pricehistory"
    )
    assert set(back["symbol"]) == {"AAA", "BBB"}


def test_reconcile_no_live_bars(s3_env, monkeypatch):
    calls = []

    def fake_get_minute_bars(symbol, start, end, extended_hours=True):
        calls.append(symbol)
        return _truth_bars("ZZZ", closes=[1.0], volumes=[1])

    monkeypatch.setattr(reconcile.schwab_client, "get_minute_bars", fake_get_minute_bars)

    summary = reconcile.reconcile_session(SESSION)

    # No live bars -> no client calls, zeroed summary.
    assert calls == []
    assert summary["session_date"] == SESSION
    assert summary["symbols"] == []
    assert summary["n_live_bars"] == 0
    assert summary["n_truth_bars"] == 0
    assert summary["n_matched"] == 0
    assert summary["close_mae"] == 0.0
    assert summary["volume_mae"] == 0.0


def test_reconcile_session_accepts_string_date(s3_env, monkeypatch):
    store.write_bars(
        _live_bars("AAA", closes=[10.0], volumes=[100]),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
        source="live_tick_agg",
    )

    monkeypatch.setattr(
        reconcile.schwab_client,
        "get_minute_bars",
        lambda symbol, start, end, extended_hours=True: _truth_bars(
            "AAA", closes=[10.0], volumes=[100]
        ),
    )

    summary = reconcile.reconcile_session("2026-05-20")
    assert summary["session_date"] == SESSION
    assert summary["n_matched"] == 1


def test_session_bounds_are_utc_for_et_day():
    start, end = reconcile._session_bounds_utc(SESSION)
    # 2026-05-20 is EDT (UTC-4): ET midnight -> 04:00 UTC.
    assert start == pd.Timestamp("2026-05-20 04:00:00", tz="UTC")
    assert end == pd.Timestamp("2026-05-21 03:59:59", tz="UTC")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_invokes_reconcile_session(monkeypatch):
    captured = {}

    def fake_reconcile_session(session_date):
        captured["date"] = session_date
        return {"session_date": session_date, "n_matched": 0}

    monkeypatch.setattr(reconcile, "reconcile_session", fake_reconcile_session)

    reconcile.main(["--date", "2026-05-20"])

    assert captured["date"] == dt.date(2026, 5, 20)
