"""Tests for the nightly collector / orchestrator (``ML_tradingAlgo.data.collector``).

The bulk of these tests exercise the **pure filter functions** which do no I/O
and need no mocks. The orchestrator tests use moto server mode (matching
``tests/test_store.py``) for the store, and monkeypatch the external
``gainers`` / ``schwab_client`` / ``fundamentals`` modules with synthetic data.
"""

from __future__ import annotations

import datetime as dt

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import collector

BUCKET = "test-collector-bucket"
PREFIX = "warehouse"


# --------------------------------------------------------------------------- #
# Pure filter tests: coarse
# --------------------------------------------------------------------------- #
def _good_daily_bar():
    # gap = (12.5 - 10) / 10 = 0.25  (lower edge, inclusive)
    return {"open": 12.5, "high": 13.0, "low": 12.0, "close": 12.8, "volume": 5_000_000}


def _good_fundamentals():
    return {"float_shares": 10_000_000, "short_interest_ratio": 2.0, "sector_id": 7}


def test_coarse_passes_in_band():
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is True
    assert dec.reasons == []


def test_coarse_gap_below_band_fails():
    # gap = 0.20 -> below 0.25
    dec = collector.filter_event_coarse(
        {"open": 12.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is False
    assert any("gap" in r for r in dec.reasons)


def test_coarse_gap_above_band_fails():
    # gap = 0.60 -> above 0.50
    dec = collector.filter_event_coarse(
        {"open": 16.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is False
    assert any("gap" in r for r in dec.reasons)


def test_coarse_gap_lower_edge_inclusive():
    dec = collector.filter_event_coarse(
        {"open": 12.5, "volume": 5_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is True


def test_coarse_gap_upper_edge_inclusive():
    dec = collector.filter_event_coarse(
        {"open": 15.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is True


def test_coarse_price_below_band_fails():
    # gap good (0.30) but open price 0.65 < 1
    dec = collector.filter_event_coarse(
        {"open": 0.65, "volume": 5_000_000}, prior_close=0.5,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is False
    assert any("price" in r for r in dec.reasons)


def test_coarse_price_above_band_fails():
    # gap 0.30, open 39 > 30
    dec = collector.filter_event_coarse(
        {"open": 39.0, "volume": 5_000_000}, prior_close=30.0,
        fundamentals=_good_fundamentals()
    )
    assert dec.passed is False
    assert any("price" in r for r in dec.reasons)


def test_coarse_float_missing_fails_closed():
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals={"short_interest_ratio": 2.0}
    )
    assert dec.passed is False
    assert "float_unknown" in dec.reasons


def test_coarse_float_nan_fails_closed():
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals={"float_shares": float("nan")}
    )
    assert dec.passed is False
    assert "float_unknown" in dec.reasons


def test_coarse_float_too_large_fails():
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 5_000_000}, prior_close=10.0,
        fundamentals={"float_shares": 60_000_000}
    )
    assert dec.passed is False
    assert any("float" in r for r in dec.reasons)


def test_coarse_rvol_proxy_below_threshold_fails():
    # daily volume 1M, avg 1M -> rvol 1.0 < 3
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 1_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals(), avg_daily_volume_20d=1_000_000,
    )
    assert dec.passed is False
    assert any("rvol" in r for r in dec.reasons)


def test_coarse_rvol_proxy_above_threshold_passes():
    # daily volume 4M, avg 1M -> rvol 4.0 > 3
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 4_000_000}, prior_close=10.0,
        fundamentals=_good_fundamentals(), avg_daily_volume_20d=1_000_000,
    )
    assert dec.passed is True


def test_coarse_rvol_skipped_when_avg_unavailable():
    # No avg provided -> rvol check skipped, still passes.
    dec = collector.filter_event_coarse(
        {"open": 13.0, "volume": 1_000}, prior_close=10.0,
        fundamentals=_good_fundamentals(), avg_daily_volume_20d=None,
    )
    assert dec.passed is True


def test_coarse_multiple_reasons_accumulate():
    # bad gap + bad price + missing float
    dec = collector.filter_event_coarse(
        {"open": 0.6, "volume": 5_000_000}, prior_close=10.0,
        fundamentals={}
    )
    assert dec.passed is False
    assert len(dec.reasons) >= 2


# --------------------------------------------------------------------------- #
# Pure filter tests: fine
# --------------------------------------------------------------------------- #
def _minute_frame(with_premarket=True):
    """Build a minute frame with premarket and regular-session bars (ET)."""
    et = "America/New_York"
    rows = []
    if with_premarket:
        for i in range(3):
            t = pd.Timestamp("2026-01-02 08:00", tz=et) + pd.Timedelta(minutes=i)
            rows.append({"ts": t.tz_convert("UTC"), "open": 11.0, "high": 11.5 + i,
                         "low": 10.5 - i, "close": 11.2, "volume": 1000})
    # regular session: first 30 minutes from 09:30
    for i in range(30):
        t = pd.Timestamp("2026-01-02 09:30", tz=et) + pd.Timedelta(minutes=i)
        rows.append({"ts": t.tz_convert("UTC"), "open": 12.0, "high": 12.5,
                     "low": 11.8, "close": 12.3, "volume": 2000})
    return pd.DataFrame(rows)


def test_fine_computes_premarket_fields():
    coarse = collector.EventDecision(passed=True, reasons=[])
    dec, fields = collector.filter_event_fine(
        _minute_frame(with_premarket=True), coarse, avg_daily_volume_20d=1_000_000
    )
    assert dec.passed is True
    # premarket high = max of (11.5, 12.5, 13.5) = 13.5
    assert fields["premarket_high"] == pytest.approx(13.5)
    # premarket low = min of (10.5, 9.5, 8.5) = 8.5
    assert fields["premarket_low"] == pytest.approx(8.5)
    assert fields["premarket_volume"] == 3000
    assert fields["open_price"] == pytest.approx(12.0)


def test_fine_rvol_at_open_computed():
    coarse = collector.EventDecision(passed=True, reasons=[])
    dec, fields = collector.filter_event_fine(
        _minute_frame(), coarse, avg_daily_volume_20d=1_000_000
    )
    # first-30-min volume = 30*2000 = 60000
    # expected = 1_000_000 * 30/390 = 76923.07...
    # rvol = 60000 / 76923 = 0.78
    assert fields["rvol_at_open"] == pytest.approx(60000 / (1_000_000 * 30 / 390))


def test_fine_premarket_unavailable_path():
    coarse = collector.EventDecision(passed=True, reasons=[])
    dec, fields = collector.filter_event_fine(
        _minute_frame(with_premarket=False), coarse, avg_daily_volume_20d=1_000_000
    )
    assert fields["premarket_high"] is None
    assert fields["premarket_low"] is None
    assert fields["premarket_volume"] is None
    assert "premarket_unavailable" in dec.reasons
    # premarket-unavailable must NOT by itself fail the event
    assert dec.passed is True


def test_fine_preserves_failed_coarse():
    coarse = collector.EventDecision(passed=False, reasons=["gap_out_of_band"])
    dec, fields = collector.filter_event_fine(
        _minute_frame(), coarse, avg_daily_volume_20d=1_000_000
    )
    assert dec.passed is False
    assert "gap_out_of_band" in dec.reasons


def test_fine_empty_frame_premarket_unavailable():
    coarse = collector.EventDecision(passed=True, reasons=[])
    dec, fields = collector.filter_event_fine(
        pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]),
        coarse, avg_daily_volume_20d=1_000_000,
    )
    assert fields["premarket_high"] is None
    assert "premarket_unavailable" in dec.reasons


# --------------------------------------------------------------------------- #
# Orchestrator tests (moto + monkeypatch)
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


def _make_daily_bars(symbol, asof_date, gap_open, prior_close, volume):
    """Two daily bars: a prior day and the asof day."""
    et = "America/New_York"
    prior = asof_date - dt.timedelta(days=1)
    rows = [
        {
            "ts": pd.Timestamp(f"{prior} 21:00", tz="UTC"),
            "session_date": prior, "symbol": symbol,
            "open": prior_close, "high": prior_close, "low": prior_close,
            "close": prior_close, "volume": 1_000_000, "vwap": float("nan"),
            "trade_count": 100, "source": "schwab_pricehistory",
        },
        {
            "ts": pd.Timestamp(f"{asof_date} 21:00", tz="UTC"),
            "session_date": asof_date, "symbol": symbol,
            "open": gap_open, "high": gap_open + 1, "low": gap_open - 1,
            "close": gap_open + 0.5, "volume": volume, "vwap": float("nan"),
            "trade_count": 200, "source": "schwab_pricehistory",
        },
    ]
    return pd.DataFrame(rows)


def _make_minute_bars(symbol, asof_date, gap_open):
    et = "America/New_York"
    rows = []
    for i in range(3):
        t = pd.Timestamp(f"{asof_date} 08:00", tz=et) + pd.Timedelta(minutes=i)
        rows.append({"ts": t.tz_convert("UTC"), "session_date": asof_date,
                     "symbol": symbol, "open": gap_open - 0.5, "high": gap_open,
                     "low": gap_open - 1, "close": gap_open - 0.2, "volume": 5000,
                     "vwap": float("nan"), "trade_count": 10,
                     "source": "schwab_pricehistory"})
    for i in range(30):
        t = pd.Timestamp(f"{asof_date} 09:30", tz=et) + pd.Timedelta(minutes=i)
        rows.append({"ts": t.tz_convert("UTC"), "session_date": asof_date,
                     "symbol": symbol, "open": gap_open, "high": gap_open + 0.5,
                     "low": gap_open - 0.2, "close": gap_open + 0.1, "volume": 10000,
                     "vwap": float("nan"), "trade_count": 20,
                     "source": "schwab_pricehistory"})
    return pd.DataFrame(rows)


@pytest.fixture
def patched_externals(monkeypatch):
    """Patch gainers / schwab_client / fundamentals with synthetic returns.

    Universe = [GAPR (a real candidate), DUD (gap too small)].
    """
    asof = dt.date(2026, 1, 2)

    from ML_tradingAlgo.data import collector as col

    monkeypatch.setattr(
        col.gainers, "load_universe", lambda since_date: ["GAPR", "DUD"]
    )

    def fake_daily(symbol, start, end):
        if symbol == "GAPR":
            return _make_daily_bars("GAPR", asof, gap_open=13.0,
                                    prior_close=10.0, volume=5_000_000)
        return _make_daily_bars("DUD", asof, gap_open=10.5,
                                prior_close=10.0, volume=5_000_000)

    def fake_minute(symbol, start, end, extended_hours=True):
        return _make_minute_bars(symbol, asof, gap_open=13.0)

    def fake_schwab_fund(symbols):
        return pd.DataFrame([
            {"symbol": s, "float_shares": 10_000_000,
             "shares_outstanding": 20_000_000, "high_52wk": 20.0, "low_52wk": 5.0}
            for s in symbols
        ])

    monkeypatch.setattr(col.schwab_client, "get_daily_bars", fake_daily)
    monkeypatch.setattr(col.schwab_client, "get_minute_bars", fake_minute)
    monkeypatch.setattr(col.schwab_client, "get_fundamentals", fake_schwab_fund)

    def fake_get_fund(symbol, asof_date):
        return {"symbol": symbol, "short_interest_ratio": 2.0, "sector_id": 7,
                "earnings_date": None, "_missing": [], "_source": "fake",
                "_fetched_at": pd.Timestamp.now(tz="UTC")}

    monkeypatch.setattr(col.fundamentals, "get_fundamentals", fake_get_fund)
    monkeypatch.setattr(col.fundamentals, "write_snapshot",
                        lambda symbol, asof_date, schwab_fields: None)

    return asof


def test_run_nightly_writes_events(s3_env, patched_externals):
    asof = patched_externals
    summary = collector.run_nightly(asof)

    assert summary["asof_date"] == asof
    assert summary["n_symbols"] == 2
    assert summary["n_candidates"] == 1  # only GAPR passes coarse
    assert summary["n_events_passed"] == 1

    events = collector.read_events((asof, asof))
    assert len(events) == 1
    assert events.iloc[0]["symbol"] == "GAPR"
    assert bool(events.iloc[0]["passed_filters"]) is True


def test_run_nightly_writes_bars(s3_env, patched_externals):
    asof = patched_externals
    collector.run_nightly(asof)

    from ML_tradingAlgo.data import store
    daily = store.read_bars("bars_daily", symbol="GAPR")
    assert len(daily) > 0
    minute = store.read_bars("bars_minute", symbol="GAPR")
    assert len(minute) > 0


def test_run_nightly_dedupes_on_rerun(s3_env, patched_externals):
    asof = patched_externals
    collector.run_nightly(asof)
    collector.run_nightly(asof)

    events = collector.read_events((asof, asof))
    # dedupe on (symbol, session_date) -> one row per symbol/date
    assert len(events) == 1
    assert events.iloc[0]["symbol"] == "GAPR"


def test_run_nightly_advances_watermark(s3_env, patched_externals):
    asof = patched_externals
    from ML_tradingAlgo.data import store
    assert store.get_watermark("bars_daily", "GAPR") is None
    collector.run_nightly(asof)
    wm = store.get_watermark("bars_daily", "GAPR")
    assert wm is not None


def test_read_events_keeps_latest_detected_at(s3_env, patched_externals):
    asof = patched_externals
    collector.run_nightly(asof)
    collector.run_nightly(asof)
    events = collector.read_events((asof, asof))
    assert len(events) == 1
    # the surviving row should carry a single detected_at
    assert "detected_at" in events.columns
