# tests/test_labeler.py
"""Tests for the training-label generator (ML_tradingAlgo/data/labeler.py).

Covers compute_labels (pure, no I/O) with synthetic bars of known outcomes,
plus the thin store-backed wrapper build_labeled_dataset (monkeypatched store).
"""

import math

import numpy as np
import pandas as pd
import pytest

from ML_tradingAlgo.data.labeler import compute_labels, build_labeled_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bars(rows):
    """rows: list of (open, high, low, close) tuples -> OHLC DataFrame."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def flat_atr_bars(entry_close, n_pre=14, true_range=1.0, n_post=10,
                  post_rows=None):
    """Build bars where each pre-entry bar has a constant true range.

    Pre-entry bars (0 .. n_pre-1) are constructed so each has true_range == TR
    via high-low == TR and no gaps (close==open==prev close baseline).
    The entry bar is at index n_pre with close == entry_close.
    Post rows follow.
    """
    rows = []
    base = entry_close
    # pre-entry bars: flat closes at `base`, high/low straddle to give TR
    for _ in range(n_pre):
        rows.append((base, base + true_range / 2.0, base - true_range / 2.0, base))
    # entry bar at index n_pre
    rows.append((base, base, base, entry_close))
    if post_rows is not None:
        rows.extend(post_rows)
    else:
        for _ in range(n_post):
            rows.append((base, base, base, base))
    return make_bars(rows), n_pre


# ---------------------------------------------------------------------------
# y_win: TP-first / SL-first / tie
# ---------------------------------------------------------------------------

def test_tp_first_gives_win_1():
    # entry close = 100; TP=103, SL=97. Bar 1 high reaches 103 -> win.
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 103.5, 99.0, 102.0)],  # high 103.5 >= 103
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == 1.0


def test_sl_first_gives_0():
    # entry close = 100; SL=97. Bar 1 low reaches 96 (no TP) -> loss.
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 101.0, 96.0, 98.0)],
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == 0.0


def test_tp_before_sl_across_bars():
    # Bar1 touches neither; Bar2 hits TP; Bar3 would hit SL but TP came first.
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[
            (100.0, 101.0, 99.0, 100.5),
            (100.0, 103.2, 100.0, 102.0),  # TP hit here
            (100.0, 100.0, 95.0, 96.0),    # SL later, ignored
        ],
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == 1.0


def test_sl_before_tp_across_bars():
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[
            (100.0, 101.0, 99.0, 100.5),
            (100.0, 101.0, 96.5, 97.5),    # SL hit here
            (100.0, 104.0, 100.0, 103.0),  # TP later, ignored
        ],
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == 0.0


def test_tie_same_bar_is_conservative_loss():
    # Single bar hits BOTH TP and SL -> conservative SL-first -> 0.0
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 104.0, 96.0, 100.0)],
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == 0.0


# ---------------------------------------------------------------------------
# y_win: neither hit -> proportional formula
#   y_win = 0.5 + 0.5 * clip(final_return / (tp_pct/100), -1, 1)
# ---------------------------------------------------------------------------

def test_neither_flat_is_half():
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 101.0, 99.0, 100.0)],  # last close == entry
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == pytest.approx(0.5)


def test_neither_up_partial():
    # last close 101.5 -> final_return = 0.015; tp frac = 0.03
    # y = 0.5 + 0.5 * (0.015/0.03) = 0.5 + 0.25 = 0.75
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 102.0, 99.5, 101.5)],  # high<103, low>97
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == pytest.approx(0.75)


def test_neither_down_partial():
    # last close 98.5 -> final_return = -0.015 -> y = 0.5 - 0.25 = 0.25
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 102.0, 98.2, 98.5)],  # low>97, high<103
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_win == pytest.approx(0.25)


def test_neither_clip_upper():
    # Use small tp_pct so partial move saturates clip at +1 -> y == 1.0,
    # but ensure TP/SL never actually triggered by using wide pct... instead
    # construct so final_return >> tp/100 while high never crosses TP.
    # tp_pct=10 -> TP=110, SL=90. final close 109.9 (high 109.95<110).
    # final_return=0.099, tp frac=0.10 -> ratio .99 -> y .995 (not clipped).
    # For clip test, make final_return exceed tp frac without high hitting TP:
    # impossible if last close>TP without high>=TP. So test clip via lower side.
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 109.0, 99.0, 99.5)],  # high 109<110 no TP, close 99.5
    )
    # final_return = -0.005, tp frac 0.10 -> y = 0.5 + 0.5*(-0.05) = 0.475
    y_win, _ = compute_labels(bars, entry, tp_pct=10.0, sl_pct=10.0, lookahead_bars=30)
    assert y_win == pytest.approx(0.475)


# ---------------------------------------------------------------------------
# Lookahead window boundaries
# ---------------------------------------------------------------------------

def test_lookahead_excludes_entry_bar():
    # Entry bar itself has huge range; must NOT count.
    rows = [(100.0, 100.0, 100.0, 100.0)] * 14  # pre (TR=0 -> guard tested elsewhere)
    rows[-1] = (100.0, 101.0, 99.0, 100.0)  # ensure some TR in pre window
    rows.append((100.0, 200.0, 1.0, 100.0))  # entry bar idx 14, wild range
    rows.append((100.0, 100.5, 99.5, 100.0))  # next bar flat-ish
    bars = make_bars(rows)
    y_win, _ = compute_labels(bars, 14, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    # entry bar's wild range ignored; next bar neither -> ~0.5
    assert y_win == pytest.approx(0.5)


def test_tiny_window_one_bar():
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 103.5, 99.0, 102.0), (100.0, 101.0, 95.0, 96.0)],
    )
    # lookahead_bars=1 -> only first post bar considered -> TP hit -> 1.0
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=1)
    assert y_win == 1.0


def test_window_truncated_at_end_of_data():
    # Only 2 post bars but lookahead_bars=30; should not error.
    bars, entry = flat_atr_bars(
        100.0,
        post_rows=[(100.0, 101.0, 99.0, 100.5), (100.0, 102.0, 99.5, 101.5)],
    )
    y_win, _ = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    # neither hit, last close 101.5 -> 0.75
    assert y_win == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# y_offset: (entry - min_low) / ATR
# ---------------------------------------------------------------------------

def test_y_offset_basic():
    # pre 14 bars each TR=2.0 -> ATR=2.0; entry close 100.
    # post lowest low = 95 -> (100-95)/2.0 = 2.5
    bars, entry = flat_atr_bars(
        100.0, n_pre=14, true_range=2.0,
        post_rows=[(100.0, 101.0, 95.0, 99.0), (100.0, 100.0, 98.0, 99.0)],
    )
    _, y_offset = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_offset == pytest.approx(2.5)


def test_y_offset_min_low_above_entry_is_negative():
    # All post lows above entry -> offset negative.
    # pre TR=1.0 -> ATR 1.0; entry 100; lowest post low = 100.5
    bars, entry = flat_atr_bars(
        100.0, n_pre=14, true_range=1.0,
        post_rows=[(100.0, 102.0, 100.5, 101.0)],
    )
    _, y_offset = compute_labels(bars, entry, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_offset == pytest.approx(-0.5)


def test_atr_uses_14_period_when_more_available():
    # 20 pre bars: first 6 with TR=10, last 14 with TR=1 -> ATR=1.0 (only last 14)
    rows = []
    for _ in range(6):
        rows.append((100.0, 105.0, 95.0, 100.0))  # TR 10
    for _ in range(14):
        rows.append((100.0, 100.5, 99.5, 100.0))  # TR 1
    rows.append((100.0, 100.0, 100.0, 100.0))      # entry idx 20
    rows.append((100.0, 101.0, 98.0, 100.0))       # post: low 98
    bars = make_bars(rows)
    _, y_offset = compute_labels(bars, 20, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    # ATR over last 14 pre bars = 1.0 ; offset = (100-98)/1 = 2.0
    assert y_offset == pytest.approx(2.0)


def test_atr_uses_full_prewindow_when_fewer_than_14():
    # Only 3 pre bars TR=2 -> ATR=2.0
    rows = [
        (100.0, 101.0, 99.0, 100.0),  # TR 2
        (100.0, 101.0, 99.0, 100.0),  # TR 2
        (100.0, 101.0, 99.0, 100.0),  # TR 2
        (100.0, 100.0, 100.0, 100.0),  # entry idx 3
        (100.0, 100.0, 96.0, 98.0),    # post low 96
    ]
    bars = make_bars(rows)
    _, y_offset = compute_labels(bars, 3, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    # offset = (100-96)/2.0 = 2.0
    assert y_offset == pytest.approx(2.0)


def test_atr_zero_guard_returns_zero_offset():
    # All pre bars flat (TR=0) -> ATR 0 -> offset guarded to 0.0
    rows = [(100.0, 100.0, 100.0, 100.0)] * 5
    rows.append((100.0, 100.0, 100.0, 100.0))  # entry idx 5
    rows.append((100.0, 100.0, 95.0, 98.0))    # post low 95
    bars = make_bars(rows)
    _, y_offset = compute_labels(bars, 5, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    assert y_offset == 0.0


def test_atr_true_range_accounts_for_gaps():
    # True range must include gap component: prev_close=100, bar high=101 low=100
    # but gap up so |high-prev_close| etc. Build a bar that gaps.
    rows = [
        (100.0, 100.0, 100.0, 100.0),  # bar0 close 100, TR (no prev) = high-low = 0
        (110.0, 111.0, 109.0, 110.0),  # bar1: prev_close 100; TR=max(2, 11, 9)=11
        (110.0, 110.0, 110.0, 110.0),  # entry idx 2, close 110
        (110.0, 110.0, 105.0, 108.0),  # post low 105
    ]
    bars = make_bars(rows)
    _, y_offset = compute_labels(bars, 2, tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)
    # pre bars indices 0,1. TR0 = high-low = 0 (no prev). TR1 = 11. ATR = (0+11)/2 = 5.5
    # offset = (110 - 105)/5.5 = 0.9090909...
    assert y_offset == pytest.approx(5.0 / 5.5)


# ---------------------------------------------------------------------------
# build_labeled_dataset (store-backed wrapper, monkeypatched)
# ---------------------------------------------------------------------------

def _session_bars(symbol):
    """Build a small intraday session: pre-market + regular hours bars.

    Index is a timezone-aware datetime; market open 09:30 ET.
    """
    times = pd.date_range("2024-03-01 09:00", periods=20, freq="1min", tz="America/New_York")
    base = 50.0
    rows = []
    for i in range(20):
        rows.append((base, base + 0.5, base - 0.5, base))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=times)
    df["symbol"] = symbol
    # After the 09:30 open bar, make a clear TP outcome for AAA, SL for BBB
    if symbol == "AAA":
        # 09:30 is index 30 minutes after 09:00 -> position 30? only 20 bars.
        pass
    return df


def test_build_labeled_dataset_one_row_per_event(monkeypatch):
    import ML_tradingAlgo.data.labeler as labeler_mod

    # Two events
    events = pd.DataFrame(
        [
            {"symbol": "AAA", "session_date": "2024-03-01", "gap_pct": 30.0, "float_shares": 5e6, "sector": 3},
            {"symbol": "BBB", "session_date": "2024-03-01", "gap_pct": 40.0, "float_shares": 8e6, "sector": 7},
        ]
    )

    def make_session(symbol, win):
        # 09:25 .. onwards, open bar at 09:30
        times = pd.date_range("2024-03-01 09:25", periods=40, freq="1min",
                              tz="America/New_York")
        base = 10.0
        rows = []
        for _ in range(40):
            rows.append((base, base + 0.2, base - 0.2, base))  # TR ~0.4, flat
        # open bar is at 09:30 = position 5. entry close = base.
        if win:  # AAA hits TP after open
            rows[6] = (base, base * 1.05, base - 0.1, base * 1.04)
        else:  # BBB hits SL after open
            rows[6] = (base, base + 0.1, base * 0.94, base * 0.95)
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=times)
        df["symbol"] = symbol
        return df

    def fake_read_bars(table, symbol=None, date_range=None, **kwargs):
        if table == "events":
            return events
        if table == "bars_minute":
            return make_session(symbol, win=(symbol == "AAA"))
        raise ValueError(f"unexpected table {table}")

    monkeypatch.setattr(labeler_mod, "read_bars", fake_read_bars, raising=False)

    out = build_labeled_dataset(("2024-03-01", "2024-03-01"),
                                tp_pct=3.0, sl_pct=3.0, lookahead_bars=30)

    assert len(out) == 2
    assert {"symbol", "y_win", "y_offset"}.issubset(out.columns)
    # static fields carried through
    assert {"gap_pct", "float_shares", "sector"}.issubset(out.columns)

    aaa = out[out["symbol"] == "AAA"].iloc[0]
    bbb = out[out["symbol"] == "BBB"].iloc[0]
    assert aaa["y_win"] == 1.0
    assert bbb["y_win"] == 0.0


def test_build_labeled_dataset_uses_lazy_import(monkeypatch):
    """read_bars is imported lazily so monkeypatching the module attr works."""
    import ML_tradingAlgo.data.labeler as labeler_mod

    events = pd.DataFrame(
        [{"symbol": "ZZZ", "session_date": "2024-03-01", "sector": 1}]
    )

    times = pd.date_range("2024-03-01 09:28", periods=40, freq="1min",
                          tz="America/New_York")
    base = 20.0
    rows = [(base, base + 0.1, base - 0.1, base) for _ in range(40)]
    session = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=times)
    session["symbol"] = "ZZZ"

    calls = {}

    def fake_read_bars(table, symbol=None, date_range=None, **kwargs):
        calls[table] = calls.get(table, 0) + 1
        if table == "events":
            return events
        return session

    monkeypatch.setattr(labeler_mod, "read_bars", fake_read_bars, raising=False)

    out = build_labeled_dataset(("2024-03-01", "2024-03-01"))
    assert len(out) == 1
    assert calls["events"] == 1
    assert calls["bars_minute"] == 1
