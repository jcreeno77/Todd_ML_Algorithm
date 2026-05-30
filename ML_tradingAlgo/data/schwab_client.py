"""Thin wrapper over the ``schwab-py`` library for market-data retrieval.

This module exposes a small, stable surface for the trading pipeline:

    get_minute_bars(symbol, start, end, extended_hours=True) -> pd.DataFrame
    get_daily_bars(symbol, start, end)                       -> pd.DataFrame
    get_fundamentals(symbols)                                -> pd.DataFrame
    get_quote_snapshot(symbols)                              -> pd.DataFrame

All price DataFrames share one canonical schema (see ``PRICE_COLUMNS``) with a
tz-aware UTC ``ts`` column and an ET ``session_date`` derived with a real
``zoneinfo`` timezone so that DST transitions are handled correctly.

Authentication is built from environment variables only (no project config):

    SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_TOKEN_PATH

The schwab-py client is cached at module level.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import schwab
from schwab.client import Client

__all__ = [
    "get_minute_bars",
    "get_daily_bars",
    "get_fundamentals",
    "get_quote_snapshot",
    "get_movers",
    "get_live_quote",
    "get_prior_close",
]

_ET = ZoneInfo("America/New_York")

PRICE_COLUMNS = [
    "ts",
    "session_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "source",
]

MOVERS_COLUMNS = [
    "symbol",
    "description",
    "last_price",
    "net_change",
    "net_pct_change",
    "volume",
    "total_volume",
    "trades",
    "market_share",
    "ts",
    "source",
]

_PRICE_SOURCE = "schwab_pricehistory"
_FUNDAMENTAL_SOURCE = "schwab_fundamental"
_MOVERS_SOURCE = "schwab_movers"

# Module-level cached client.
_CLIENT = None

# Backoff schedule (seconds) for HTTP 429 responses: 1, 2, 4, 8 -> up to 4 retries.
_MAX_RETRIES = 4


# --------------------------------------------------------------------------- #
# Auth / client
# --------------------------------------------------------------------------- #
def _get_client():
    """Build (and cache) a schwab-py client from environment credentials."""
    global _CLIENT
    if _CLIENT is None:
        app_key = os.environ.get("SCHWAB_APP_KEY")
        app_secret = os.environ.get("SCHWAB_APP_SECRET")
        token_path = os.environ.get("SCHWAB_TOKEN_PATH")
        _CLIENT = schwab.auth.client_from_token_file(
            token_path, app_key, app_secret
        )
    return _CLIENT


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #
def _request_with_retry(fn, *args, **kwargs):
    """Call ``fn`` returning an httpx-style Response, retrying on HTTP 429.

    Uses exponential backoff (1, 2, 4, 8 s). ``time.sleep`` is patchable so
    tests can run instantly.
    """
    backoff = 1
    last_response = None
    for attempt in range(_MAX_RETRIES + 1):
        response = fn(*args, **kwargs)
        last_response = response
        if getattr(response, "status_code", 200) != 429:
            return response
        if attempt < _MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"Schwab API returned HTTP 429 after {_MAX_RETRIES} retries"
        + (f" (last status {last_response.status_code})" if last_response else "")
    )


# --------------------------------------------------------------------------- #
# Price-history shaping
# --------------------------------------------------------------------------- #
def _candles_to_frame(payload, symbol):
    """Convert a Schwab price-history payload into the canonical price frame."""
    candles = payload.get("candles", []) if payload else []

    if not candles:
        empty = pd.DataFrame(columns=PRICE_COLUMNS)
        empty["ts"] = pd.to_datetime(empty["ts"], utc=True)
        empty["trade_count"] = empty["trade_count"].astype("Int32")
        return empty

    df = pd.DataFrame(candles)

    # Schwab candle datetimes are epoch milliseconds -> tz-aware UTC.
    ts = pd.to_datetime(df["datetime"].astype("int64"), unit="ms", utc=True)

    # session_date is the ET *calendar* trading date for that instant.
    session_date = ts.dt.tz_convert(_ET).dt.date

    out = pd.DataFrame(
        {
            "ts": ts,
            "session_date": session_date,
            "symbol": symbol,
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("int64"),
        }
    )

    # vwap: NaN unless Schwap supplies it.
    if "vwap" in df.columns:
        out["vwap"] = df["vwap"].astype("float64")
    else:
        out["vwap"] = np.float64(np.nan)

    # trade_count: nullable Int32.
    if "tradeCount" in df.columns:
        out["trade_count"] = df["tradeCount"].astype("Int32")
    elif "trade_count" in df.columns:
        out["trade_count"] = df["trade_count"].astype("Int32")
    else:
        out["trade_count"] = pd.array([pd.NA] * len(out), dtype="Int32")

    out["source"] = _PRICE_SOURCE

    return out[PRICE_COLUMNS]


# --------------------------------------------------------------------------- #
# Public API: price history
# --------------------------------------------------------------------------- #
def get_minute_bars(symbol, start, end, extended_hours: bool = True) -> pd.DataFrame:
    """Per-minute OHLCV bars for ``symbol`` between ``start`` and ``end``.

    When ``extended_hours`` is True (default) premarket / afterhours bars are
    requested via Schwab's ``need_extended_hours_data`` flag.
    """
    client = _get_client()
    response = _request_with_retry(
        client.get_price_history_every_minute,
        symbol,
        start_datetime=start,
        end_datetime=end,
        need_extended_hours_data=extended_hours,
    )
    return _candles_to_frame(response.json(), symbol)


def get_daily_bars(symbol, start, end) -> pd.DataFrame:
    """Daily OHLCV bars for ``symbol`` between ``start`` and ``end``."""
    client = _get_client()
    response = _request_with_retry(
        client.get_price_history_every_day,
        symbol,
        start_datetime=start,
        end_datetime=end,
    )
    return _candles_to_frame(response.json(), symbol)


# --------------------------------------------------------------------------- #
# Public API: fundamentals
# --------------------------------------------------------------------------- #
def get_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """Fundamental data (float, shares, 52-week range) for ``symbols``.

    ``float_shares`` is ``fundamental.marketCapFloat * 1e6`` to match the legacy
    ``TD_Ameritrade_Data.py`` convention (Schwab reports float in millions).
    """
    client = _get_client()
    response = _request_with_retry(
        client.get_instruments,
        list(symbols),
        projection=Client.Instrument.Projection.FUNDAMENTAL,
    )
    payload = response.json() or {}

    rows = []
    for sym in symbols:
        entry = payload.get(sym)
        if not entry:
            continue
        fund = entry.get("fundamental", {})
        market_cap_float = fund.get("marketCapFloat")
        float_shares = (
            float(market_cap_float) * 1e6 if market_cap_float is not None else np.nan
        )
        rows.append(
            {
                "symbol": sym,
                "float_shares": float_shares,
                "shares_outstanding": fund.get("sharesOutstanding"),
                "high_52wk": fund.get("high52"),
                "low_52wk": fund.get("low52"),
                "source": _FUNDAMENTAL_SOURCE,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "float_shares",
            "shares_outstanding",
            "high_52wk",
            "low_52wk",
            "source",
        ],
    )


# --------------------------------------------------------------------------- #
# Public API: quote snapshot
# --------------------------------------------------------------------------- #
def get_quote_snapshot(symbols: list[str]) -> pd.DataFrame:
    """Current-quote snapshot (last/bid/ask/volume) for ``symbols``."""
    client = _get_client()
    response = _request_with_retry(client.get_quotes, list(symbols))
    payload = response.json() or {}

    now = pd.Timestamp.now(tz="UTC")

    rows = []
    for sym in symbols:
        entry = payload.get(sym)
        if not entry:
            continue
        quote = entry.get("quote", {})
        rows.append(
            {
                "symbol": sym,
                "last_price": quote.get("lastPrice"),
                "bid": quote.get("bidPrice"),
                "ask": quote.get("askPrice"),
                "total_volume": quote.get("totalVolume"),
                "ts": now,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["symbol", "last_price", "bid", "ask", "total_volume", "ts"],
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# --------------------------------------------------------------------------- #
# Public API: market movers (screener)
# --------------------------------------------------------------------------- #
def _resolve_movers_enum(value, enum_cls):
    """Accept an enum member, its NAME (``"EQUITY_ALL"``) or VALUE and return
    the matching member. ``None`` passes through (Schwab treats it as default).
    """
    if value is None or isinstance(value, enum_cls):
        return value
    try:
        return enum_cls[value]  # by NAME
    except (KeyError, TypeError):
        return enum_cls(value)  # by VALUE


def get_movers(
    index: str = "EQUITY_ALL",
    sort_order: str = "PERCENT_CHANGE_UP",
    frequency=None,
) -> pd.DataFrame:
    """Top market movers for ``index`` as a canonical screener frame.

    A *live snapshot* (not historical) of the biggest movers Schwab tracks for
    the given index -- e.g. the default ``EQUITY_ALL`` sorted by
    ``PERCENT_CHANGE_UP`` surfaces the session's strongest gainers across all
    equities. Schwab returns at most a few dozen rows, so this is a discovery
    signal to seed the universe, NOT a full-universe scan (see
    ``scan_gappers`` for the market-wide grouped-daily screen).

    ``net_pct_change`` is the *intraday* percent change (as a fraction), which
    is NOT the open-vs-prior-close gap the backfill coarse screen uses -- treat
    it as a proxy and re-screen survivors before trusting them as gappers.

    ``index`` / ``sort_order`` accept either the ``Client.Movers`` enum members
    or their string names; ``frequency`` accepts the enum, its int value, or
    ``None`` (Schwab default).
    """
    client = _get_client()
    index = _resolve_movers_enum(index, Client.Movers.Index)
    sort_order = _resolve_movers_enum(sort_order, Client.Movers.SortOrder)
    frequency = _resolve_movers_enum(frequency, Client.Movers.Frequency)

    kwargs = {}
    if sort_order is not None:
        kwargs["sort_order"] = sort_order
    if frequency is not None:
        kwargs["frequency"] = frequency

    response = _request_with_retry(client.get_movers, index, **kwargs)
    payload = response.json() or {}
    screeners = payload.get("screeners", []) if isinstance(payload, dict) else []

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for item in screeners:
        # Schwab's documented keys, with legacy/alt fallbacks for robustness
        # (verified against the docs; not yet against a live token).
        pct = item.get("netPercentChange")
        if pct is None:
            pct = item.get("netPercentChangeInDouble")  # legacy TDA name
        rows.append(
            {
                "symbol": item.get("symbol"),
                "description": item.get("description"),
                "last_price": item.get("lastPrice", item.get("last")),
                "net_change": item.get("netChange", item.get("change")),
                "net_pct_change": pct,
                "volume": item.get("volume"),
                "total_volume": item.get("totalVolume"),
                "trades": item.get("trades"),
                "market_share": item.get("marketShare"),
                "ts": now,
                "source": _MOVERS_SOURCE,
            }
        )

    df = pd.DataFrame(rows, columns=MOVERS_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# --------------------------------------------------------------------------- #
# Public API: single-symbol live helpers
# --------------------------------------------------------------------------- #
def get_live_quote(symbol: str) -> dict:
    """Lightweight single-symbol live quote for the trading loop.

    Returns a plain dict with the four fields the live loop needs::

        {"last_price": float, "total_volume": int,
         "high_52wk": float, "low_52wk": float}

    Fields are parsed from the Schwab ``get_quotes`` payload under
    ``<SYMBOL>.quote`` (``lastPrice``, ``totalVolume``, ``52WeekHigh``,
    ``52WeekLow``). Missing keys yield ``None`` rather than raising.
    """
    client = _get_client()
    response = _request_with_retry(client.get_quotes, [symbol])
    payload = response.json() or {}

    quote = (payload.get(symbol) or {}).get("quote", {})
    return {
        "last_price": quote.get("lastPrice"),
        "total_volume": quote.get("totalVolume"),
        "high_52wk": quote.get("52WeekHigh"),
        "low_52wk": quote.get("52WeekLow"),
    }


def get_prior_close(symbol: str) -> float:
    """Most recent daily close for ``symbol``.

    Reuses :func:`get_daily_bars` over a short recent window (the last ~7
    calendar days up to today) and returns the last row's ``close`` as a float.
    """
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=7)
    bars = get_daily_bars(symbol, start, end)
    return float(bars["close"].iloc[-1])
