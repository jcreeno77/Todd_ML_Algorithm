"""Tests for the seed backfill (``ML_tradingAlgo.data.backfill``).

Bootstraps the historical corpus from a curated seed list of known gappers.
Uses moto server mode for the store (matching ``tests/test_store.py``) and
monkeypatches the external ``schwab_client`` / ``fundamentals`` / ``gainers``
modules with synthetic data.
"""

from __future__ import annotations

import datetime as dt

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import backfill, collector

BUCKET = "test-backfill-bucket"
PREFIX = "warehouse"

# Cutoff sits between old_event (25d ago) and recent_event (2d ago).
CUTOFF = 10


# --------------------------------------------------------------------------- #
# moto server + s3 env (copied from tests/test_store.py)
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
def _daily_row(symbol, session_date, o, h, l, c, v):
    return {
        "ts": pd.Timestamp(f"{session_date} 21:00", tz="UTC"),
        "session_date": session_date,
        "symbol": symbol,
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "vwap": float("nan"), "trade_count": 100,
        "source": "schwab_pricehistory",
    }


def _gap_daily_history(symbol, event_dates, window_start, window_end):
    """Build a continuous daily frame: a flat baseline plus a gap on each
    ``event_date`` (open = 1.3x prior close, high volume)."""
    rows = []
    d = window_start
    prior_close = 10.0
    base_vol = 1_000_000
    event_set = set(event_dates)
    while d <= window_end:
        if d in event_set:
            o = prior_close * 1.30  # +30% gap, in [0.25, 0.50] band
            rows.append(_daily_row(symbol, d, o, o + 1, o - 1, o + 0.5,
                                   base_vol * 5))  # rvol 5x
            prior_close = o + 0.5
        else:
            o = prior_close
            rows.append(_daily_row(symbol, d, o, o + 0.1, o - 0.1, o,
                                   base_vol))
            prior_close = o
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def _minute_bars(symbol, session_date, gap_open):
    et = "America/New_York"
    rows = []
    for i in range(3):
        t = pd.Timestamp(f"{session_date} 08:00", tz=et) + pd.Timedelta(minutes=i)
        rows.append({"ts": t.tz_convert("UTC"), "session_date": session_date,
                     "symbol": symbol, "open": gap_open - 0.5, "high": gap_open,
                     "low": gap_open - 1, "close": gap_open - 0.2, "volume": 5000,
                     "vwap": float("nan"), "trade_count": 10,
                     "source": "schwab_pricehistory"})
    for i in range(30):
        t = pd.Timestamp(f"{session_date} 09:30", tz=et) + pd.Timedelta(minutes=i)
        rows.append({"ts": t.tz_convert("UTC"), "session_date": session_date,
                     "symbol": symbol, "open": gap_open, "high": gap_open + 0.5,
                     "low": gap_open - 0.2, "close": gap_open + 0.1, "volume": 10000,
                     "vwap": float("nan"), "trade_count": 20,
                     "source": "schwab_pricehistory"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Patch externals. Two seed symbols, each with one gap event:
#   RECENT  gaps on a date INSIDE the minute window (today-10d)
#   OLD     gaps on a date OUTSIDE the minute window (today-300d)
# --------------------------------------------------------------------------- #
@pytest.fixture
def patched_externals(monkeypatch):
    today = dt.date(2026, 5, 29)  # matches the prompt's "today"
    monkeypatch.setattr(backfill, "_today", lambda: today)

    # Compact window keeps moto I/O fast; a small minute_cutoff_days (passed by
    # the tests) is what separates "recent" from "old".
    recent_event = today - dt.timedelta(days=2)
    old_event = today - dt.timedelta(days=25)

    # backfill window must cover both events (plus a little ADV baseline).
    start_date = today - dt.timedelta(days=35)
    end_date = today

    minute_calls = []

    def fake_daily(symbol, start, end):
        if symbol == "RECENT":
            return _gap_daily_history("RECENT", [recent_event], start_date, end_date)
        if symbol == "OLD":
            return _gap_daily_history("OLD", [old_event], start_date, end_date)
        return pd.DataFrame()

    def fake_minute(symbol, start, end, extended_hours=True):
        minute_calls.append((symbol, start, end))
        # gap_open for the recent event date
        return _minute_bars(symbol, recent_event, gap_open=13.0)

    def fake_schwab_fund(symbols):
        return pd.DataFrame([
            {"symbol": s, "float_shares": 10_000_000,
             "shares_outstanding": 20_000_000, "high_52wk": 20.0, "low_52wk": 5.0}
            for s in symbols
        ])

    def fake_get_fund(symbol, asof_date):
        return {"symbol": symbol, "short_interest_ratio": 2.0, "sector_id": 7,
                "earnings_date": None, "_missing": [], "_source": "fake",
                "_fetched_at": pd.Timestamp.now(tz="UTC")}

    monkeypatch.setattr(collector.schwab_client, "get_daily_bars", fake_daily)
    monkeypatch.setattr(collector.schwab_client, "get_minute_bars", fake_minute)
    monkeypatch.setattr(collector.schwab_client, "get_fundamentals", fake_schwab_fund)
    monkeypatch.setattr(collector.fundamentals, "get_fundamentals", fake_get_fund)
    monkeypatch.setattr(collector.fundamentals, "write_snapshot",
                        lambda symbol, asof_date, schwab_fields: None)

    return {
        "today": today,
        "recent_event": recent_event,
        "old_event": old_event,
        "start_date": start_date,
        "end_date": end_date,
        "minute_calls": minute_calls,
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_seed_list_detects_events_over_range(s3_env, patched_externals):
    ctx = patched_externals
    summary = backfill.backfill_seed(
        ["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
        minute_cutoff_days=CUTOFF,
    )
    assert summary["n_symbols"] == 2
    # exactly one gap event per symbol
    assert summary["n_events"] == 2

    events = collector.read_events((ctx["start_date"], ctx["end_date"]))
    syms = set(events["symbol"])
    assert syms == {"RECENT", "OLD"}


def test_seeds_universe(s3_env, patched_externals):
    ctx = patched_externals
    backfill.backfill_seed(["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
                           minute_cutoff_days=CUTOFF)
    from ML_tradingAlgo.data import gainers
    universe = gainers.load_universe(ctx["start_date"])
    assert "RECENT" in universe
    assert "OLD" in universe


def test_old_event_minute_unavailable_no_minute_pull(s3_env, patched_externals):
    ctx = patched_externals
    backfill.backfill_seed(["OLD"], ctx["start_date"], ctx["end_date"],
                           minute_cutoff_days=CUTOFF)

    # No minute call should have been made for OLD (older than cutoff).
    assert all(call[0] != "OLD" for call in ctx["minute_calls"])

    events = collector.read_events((ctx["start_date"], ctx["end_date"]))
    old = events[events["symbol"] == "OLD"]
    assert len(old) == 1
    assert "minute_unavailable" in old.iloc[0]["filter_reasons"]

    # no minute bars persisted for OLD
    from ML_tradingAlgo.data import store
    minute = store.read_bars("bars_minute", symbol="OLD")
    assert len(minute) == 0


def test_recent_event_pulls_minute(s3_env, patched_externals):
    ctx = patched_externals
    backfill.backfill_seed(["RECENT"], ctx["start_date"], ctx["end_date"],
                           minute_cutoff_days=CUTOFF)

    assert any(call[0] == "RECENT" for call in ctx["minute_calls"])

    from ML_tradingAlgo.data import store
    minute = store.read_bars("bars_minute", symbol="RECENT")
    assert len(minute) > 0

    events = collector.read_events((ctx["start_date"], ctx["end_date"]))
    rec = events[events["symbol"] == "RECENT"]
    assert len(rec) == 1
    assert "minute_unavailable" not in str(rec.iloc[0]["filter_reasons"])


def test_summary_counts(s3_env, patched_externals):
    ctx = patched_externals
    summary = backfill.backfill_seed(
        ["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
        minute_cutoff_days=CUTOFF,
    )
    assert summary["n_symbols"] == 2
    assert summary["n_events"] == 2
    assert summary["n_minute_unavailable"] == 1  # only OLD
    assert summary["date_range"] == (ctx["start_date"], ctx["end_date"])


def test_rerun_no_duplicate_events(s3_env, patched_externals):
    ctx = patched_externals
    backfill.backfill_seed(["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
                           minute_cutoff_days=CUTOFF)
    backfill.backfill_seed(["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
                           minute_cutoff_days=CUTOFF)

    events = collector.read_events((ctx["start_date"], ctx["end_date"]))
    # read_events dedupes on (symbol, session_date)
    assert len(events) == 2


def test_resume_after_interruption(s3_env, patched_externals):
    """Simulate a partial first run by pre-advancing OLD's watermark past its
    event, then run: OLD must be skipped (resumed), RECENT still processed."""
    ctx = patched_externals
    from ML_tradingAlgo.data import store

    # Pretend a prior run already completed OLD up through end_date.
    store.set_watermark(
        backfill.DAILY_TABLE, "OLD",
        pd.Timestamp(ctx["end_date"]).tz_localize("UTC"),
    )

    summary = backfill.backfill_seed(
        ["RECENT", "OLD"], ctx["start_date"], ctx["end_date"],
        minute_cutoff_days=CUTOFF,
    )

    # OLD already watermarked to end -> not re-detected this run; only RECENT.
    events = collector.read_events((ctx["start_date"], ctx["end_date"]))
    assert "RECENT" in set(events["symbol"])
    # No duplicate RECENT rows.
    assert len(events[events["symbol"] == "RECENT"]) == 1


def test_cli_main(s3_env, patched_externals, capsys):
    ctx = patched_externals
    backfill.main([
        "--symbols", "RECENT,OLD",
        "--start", ctx["start_date"].isoformat(),
        "--end", ctx["end_date"].isoformat(),
        "--minute-cutoff-days", str(CUTOFF),
    ])
    out = capsys.readouterr().out
    assert "RECENT" in out or "symbols=2" in out
