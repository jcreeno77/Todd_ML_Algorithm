# tests/test_massive_client.py
"""Massive client: canonical-frame mapping, pagination, fundamentals proxy.

No network: ``massive_client._request`` is monkeypatched with canned payloads.
"""
import datetime as dt

import pandas as pd
import pytest

from ML_tradingAlgo.data import massive_client as mc


def _agg(t_ms, o, h, l, c, v, vw=None, n=None):
    bar = {"t": t_ms, "o": o, "h": h, "l": l, "c": c, "v": v}
    if vw is not None:
        bar["vw"] = vw
    if n is not None:
        bar["n"] = n
    return bar


def test_aggs_to_frame_schema_and_session_date():
    # 2025-05-15 13:30:00 UTC == 09:30 ET (regular open) -> session_date 2025-05-15
    t = int(pd.Timestamp("2025-05-15 13:30:00", tz="UTC").timestamp() * 1000)
    df = mc._aggs_to_frame([_agg(t, 10.0, 11.0, 9.5, 10.5, 1234.6, vw=10.2, n=42)], "FOO")

    assert list(df.columns) == mc.PRICE_COLUMNS
    row = df.iloc[0]
    assert row["symbol"] == "FOO"
    assert row["session_date"] == dt.date(2025, 5, 15)
    assert str(df["ts"].dt.tz) == "UTC"
    assert row["open"] == 10.0 and row["close"] == 10.5
    assert row["volume"] == 1235  # rounded float -> int
    assert row["vwap"] == 10.2
    assert int(row["trade_count"]) == 42
    assert row["source"] == "massive_aggregates"


def test_aggs_to_frame_empty():
    df = mc._aggs_to_frame([], "FOO")
    assert list(df.columns) == mc.PRICE_COLUMNS
    assert len(df) == 0


def test_get_minute_bars_parses_payload(monkeypatch):
    t = int(pd.Timestamp("2025-05-15 12:00:00", tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(
        mc, "_request",
        lambda *a, **k: {"results": [_agg(t, 1, 2, 0.5, 1.5, 100, vw=1.2, n=3)]},
    )
    df = mc.get_minute_bars("BAR", "2025-05-15", "2025-05-15")
    assert len(df) == 1 and df.iloc[0]["symbol"] == "BAR"
    assert list(df.columns) == mc.PRICE_COLUMNS


def test_pagination_follows_next_url(monkeypatch):
    t0 = int(pd.Timestamp("2025-05-15 12:00:00", tz="UTC").timestamp() * 1000)
    t1 = t0 + 60_000
    pages = [
        {"results": [_agg(t0, 1, 2, 0.5, 1.5, 100)], "next_url": "https://api.massive.com/next"},
        {"results": [_agg(t1, 1.5, 2.5, 1.0, 2.0, 200)]},
    ]
    calls = {"n": 0}

    def fake_request(path_or_url, params=None):
        i = calls["n"]; calls["n"] += 1
        return pages[i]

    monkeypatch.setattr(mc, "_request", fake_request)
    df = mc.get_daily_bars("BAZ", "2025-05-15", "2025-05-16")
    assert calls["n"] == 2           # followed next_url exactly once
    assert len(df) == 2              # both pages concatenated


def test_get_fundamentals_float_proxy_and_52wk(monkeypatch):
    # ticker details via _request; daily bars via get_daily_bars.
    monkeypatch.setattr(
        mc, "_request",
        lambda *a, **k: {"results": {"share_class_shares_outstanding": 12_000_000,
                                     "weighted_shares_outstanding": 11_000_000}},
    )
    daily = pd.DataFrame({"high": [5.0, 9.0, 7.0], "low": [4.0, 3.0, 6.0]})
    monkeypatch.setattr(mc, "get_daily_bars", lambda *a, **k: daily)

    out = mc.get_fundamentals(["XYZ"])
    row = out.iloc[0]
    assert row["symbol"] == "XYZ"
    assert row["float_shares"] == 12_000_000.0   # share_class proxy
    assert row["shares_outstanding"] == 12_000_000
    assert row["high_52wk"] == 9.0 and row["low_52wk"] == 3.0
    assert row["source"] == "massive_reference"
