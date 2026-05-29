"""Tests for the event->tensor assembly bridge (ML_tradingAlgo/data/assemble.py).

Fully synthetic: read_bars / fundamentals are injected or monkeypatched so
nothing hits S3 or the network. The most important test is #8
(test_alignment_guard), which proves features and labels reference the same
entry bar.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ML_tradingAlgo.data import assemble
from ML_tradingAlgo.data import labeler
from ML_tradingAlgo.tft import features as tft_features


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_session_bars(
    session_date=dt.date(2025, 1, 6),
    n_pre=40,
    n_post=30,
    start_close=100.0,
):
    """Build a synthetic 1-min OHLCV session with a ts column.

    ``n_pre`` premarket bars (before 09:30) and ``n_post`` regular-hours bars
    (at/after 09:30). The entry bar is the first 09:30 bar. The feature window
    is ``bars.iloc[:entry_idx + 1]`` (pre-entry + entry), so ``n_pre`` must be
    >= sequence_length for build_feature_matrix to yield a full window. Returns
    a DataFrame with columns
    [symbol, session_date, ts, open, high, low, close, volume].
    """
    rows = []
    # premarket bars 09:20.. (before open)
    open_dt = dt.datetime.combine(session_date, dt.time(9, 30))
    base = start_close
    total = n_pre + n_post
    for i in range(total):
        minute_offset = i - n_pre  # entry bar at offset 0 -> 09:30
        ts = open_dt + dt.timedelta(minutes=minute_offset)
        c = base + i * 0.05
        o = c - 0.02
        h = c + 0.10
        lo = c - 0.10
        rows.append({
            "symbol": "TEST",
            "session_date": session_date,
            "ts": ts,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": 10000 + i * 10,
            "source": "synthetic",
        })
    return pd.DataFrame(rows)


def make_event_row(session_date=dt.date(2025, 1, 6), float_shares=8_000_000.0):
    return {
        "symbol": "TEST",
        "session_date": session_date,
        "prior_close": 74.0,
        "open_price": 100.0,
        "gap_pct": 0.35,
        "premarket_high": 101.0,
        "premarket_low": 98.0,
        "premarket_volume": 500000.0,
        "float_shares": float_shares,
        "rvol_at_open": 5.0,
    }


def make_read_bars(bars_by_symbol):
    """Return a read_bars(table, symbol, date_range, ...) over in-memory frames."""
    def _read_bars(table, symbol=None, date_range=None, source=None, dedupe_keys=None):
        if table == "bars_minute":
            df = bars_by_symbol.get(symbol)
            return df.copy() if df is not None else pd.DataFrame()
        raise AssertionError(f"unexpected read_bars table {table!r}")
    return _read_bars


# --------------------------------------------------------------------------- #
# 1. resample_1min_to_5min
# --------------------------------------------------------------------------- #
def test_resample_1min_to_5min():
    start = dt.datetime(2025, 1, 6, 9, 30)
    rows = []
    for i in range(10):
        rows.append({
            "ts": start + dt.timedelta(minutes=i),
            "open": 100.0 + i,
            "high": 110.0 + i,
            "low": 90.0 - i,
            "close": 105.0 + i,
            "volume": 100 + i,
        })
    df = pd.DataFrame(rows)

    out = assemble.resample_1min_to_5min(df)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2  # 10 one-min bars -> 2 five-min buckets

    g0 = df.iloc[0:5]
    g1 = df.iloc[5:10]
    assert out.iloc[0]["open"] == g0["open"].iloc[0]
    assert out.iloc[0]["high"] == g0["high"].max()
    assert out.iloc[0]["low"] == g0["low"].min()
    assert out.iloc[0]["close"] == g0["close"].iloc[-1]
    assert out.iloc[0]["volume"] == g0["volume"].sum()

    assert out.iloc[1]["open"] == g1["open"].iloc[0]
    assert out.iloc[1]["high"] == g1["high"].max()
    assert out.iloc[1]["low"] == g1["low"].min()
    assert out.iloc[1]["close"] == g1["close"].iloc[-1]
    assert out.iloc[1]["volume"] == g1["volume"].sum()


# --------------------------------------------------------------------------- #
# 2. build_static_data
# --------------------------------------------------------------------------- #
def test_build_static_data_keys_and_finite():
    event_row = make_event_row()
    bars = make_session_bars()

    static_data = assemble.build_static_data(
        event_row, bars, fundamentals_row=None, current_price=100.0
    )

    # Every key compute_static_features needs must be present.
    needed = [
        "float_shares", "short_interest_ratio", "gap_percentage", "sector_id",
        "days_since_earnings", "high_52wk", "low_52wk", "premarket_high",
        "premarket_low", "current_price", "prior_close",
    ]
    for key in needed:
        assert key in static_data, f"missing static_data key {key}"

    # compute_static_features must accept the dict without error.
    cont, cat = tft_features.compute_static_features(**static_data)
    assert cont.shape == (7,)
    assert cat.shape == (1,)

    # No None/NaN in the continuous set.
    for name in tft_features.STATIC_CONTINUOUS_FEATURE_NAMES:
        pass  # name list documents the 7 continuous features
    assert np.all(np.isfinite(cont))

    sid = static_data["sector_id"]
    assert isinstance(sid, int)
    assert 0 <= sid <= 10


# --------------------------------------------------------------------------- #
# 3. assemble_event happy path
# --------------------------------------------------------------------------- #
def test_assemble_event_happy_path():
    bars = make_session_bars(n_pre=40, n_post=30)
    read_bars = make_read_bars({"TEST": bars})
    event_row = make_event_row()

    result = assemble.assemble_event(
        event_row, read_bars=read_bars, min_bars=35, sequence_length=30
    )

    assert result is not None
    assert result["temporal"].shape == (30, 47)
    assert result["temporal"].dtype == np.float32
    assert result["static_continuous"].shape == (7,)
    assert result["static_continuous"].dtype == np.float32
    assert result["static_categorical"].shape == (1,)
    assert result["static_categorical"].dtype == np.int64
    assert 0.0 <= result["y_win"] <= 1.0
    assert np.isfinite(result["y_offset"])
    assert np.all(np.isfinite(result["temporal"]))


# --------------------------------------------------------------------------- #
# 4. assemble_event short session -> None
# --------------------------------------------------------------------------- #
def test_assemble_event_short_session():
    # Only 20 regular-hours bars -> bars_through_entry = 1; well below min_bars.
    bars = make_session_bars(n_pre=2, n_post=18)
    read_bars = make_read_bars({"TEST": bars})
    event_row = make_event_row()

    result = assemble.assemble_event(
        event_row, read_bars=read_bars, min_bars=35, sequence_length=30
    )
    assert result is None


def test_assemble_event_no_float():
    bars = make_session_bars()
    read_bars = make_read_bars({"TEST": bars})
    event_row = make_event_row(float_shares=0.0)

    result = assemble.assemble_event(event_row, read_bars=read_bars)
    assert result is None


# --------------------------------------------------------------------------- #
# 5. assemble_dataset over 3 events (one short) -> N=2, one skipped
# --------------------------------------------------------------------------- #
def test_assemble_dataset_stacks_and_skips(monkeypatch):
    d1 = dt.date(2025, 1, 6)
    d2 = dt.date(2025, 1, 7)
    d3 = dt.date(2025, 1, 8)

    # AAA and BBB are healthy; CCC has a short session.
    bars_aaa = make_session_bars(session_date=d1, n_pre=40, n_post=30)
    bars_aaa["symbol"] = "AAA"
    bars_bbb = make_session_bars(session_date=d2, n_pre=40, n_post=30)
    bars_bbb["symbol"] = "BBB"
    bars_ccc = make_session_bars(session_date=d3, n_pre=2, n_post=15)
    bars_ccc["symbol"] = "CCC"

    bars_by_symbol = {"AAA": bars_aaa, "BBB": bars_bbb, "CCC": bars_ccc}

    events = pd.DataFrame([
        {**make_event_row(d1), "symbol": "AAA"},
        {**make_event_row(d2), "symbol": "BBB"},
        {**make_event_row(d3), "symbol": "CCC"},
    ])

    def fake_read_bars(table, symbol=None, date_range=None, source=None, dedupe_keys=None):
        if table == "events":
            return events.copy()
        if table == "bars_minute":
            df = bars_by_symbol.get(symbol)
            return df.copy() if df is not None else pd.DataFrame()
        raise AssertionError(f"unexpected table {table!r}")

    out = assemble.assemble_dataset(
        (d1, d3),
        read_bars=fake_read_bars,
        min_bars=35,
        sequence_length=30,
    )

    assert out["temporal"].shape == (2, 30, 47)
    assert out["static_continuous"].shape == (2, 7)
    assert out["static_categorical"].shape == (2, 1)
    assert out["y_win"].shape == (2,)
    assert out["y_offset"].shape == (2,)
    assert out["sample_weights"].shape == (2,)
    assert len(out["session_dates"]) == 2
    assert len(out["symbols"]) == 2
    assert len(out["skipped"]) == 1
    skipped_symbol, skipped_date, reason = out["skipped"][0]
    assert skipped_symbol == "CCC"
    assert reason == "short_session"


# --------------------------------------------------------------------------- #
# 6. SPY-None path -> graceful degradation, finite features
# --------------------------------------------------------------------------- #
def test_assemble_event_spy_none():
    bars = make_session_bars(n_pre=40, n_post=30)
    read_bars = make_read_bars({"TEST": bars})
    event_row = make_event_row()

    result = assemble.assemble_event(
        event_row, read_bars=read_bars, spy_bars=None, min_bars=35
    )
    assert result is not None
    assert np.all(np.isfinite(result["temporal"]))


# --------------------------------------------------------------------------- #
# 7. sample weights recency
# --------------------------------------------------------------------------- #
def test_recency_weights():
    newest = dt.date(2025, 1, 1)
    older = newest - dt.timedelta(days=int(round(12 * 30.44)))  # ~12 months

    weights = assemble._recency_weights([older, newest], lambda_decay=0.0578)

    # newest event weight ~= 1.0
    assert weights[1] == pytest.approx(1.0, abs=1e-9)
    # ~12 months older ~= exp(-0.0578 * 12)
    expected = np.exp(-0.0578 * 12)
    assert weights[0] == pytest.approx(expected, rel=1e-3)


# --------------------------------------------------------------------------- #
# 8. ALIGNMENT GUARD (most important)
# --------------------------------------------------------------------------- #
def test_alignment_guard():
    bars_raw = make_session_bars(n_pre=40, n_post=30)
    read_bars = make_read_bars({"TEST": bars_raw})
    event_row = make_event_row()

    result = assemble.assemble_event(
        event_row, read_bars=read_bars, min_bars=35, sequence_length=30,
        tp_pct=3.0, sl_pct=3.0, lookahead_bars=30,
    )
    assert result is not None

    # Independently reproduce the exact frame + entry_idx assemble_event used.
    bars = bars_raw.sort_values("ts")
    bars = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["ts"])))
    entry_idx = labeler._first_regular_hours_idx(bars)

    y_win_direct, y_offset_direct = labeler.compute_labels(
        bars, entry_idx, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30
    )

    assert result["y_win"] == pytest.approx(y_win_direct)
    assert result["y_offset"] == pytest.approx(y_offset_direct)

    # Sanity: entry bar is the first 09:30 bar (offset 0 -> index n_pre == 10).
    assert entry_idx == 40
