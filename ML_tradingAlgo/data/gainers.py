"""Universe (gainers) tracking for the trading-data pipeline.

A thin observation log on top of :mod:`ML_tradingAlgo.data.store`. Every time
the live scanner (or a manual seed / backfill) considers a symbol, it records an
*observation* into the ``universe`` table. The collector then asks
:func:`load_universe` for the set of symbols seen since a given date so it knows
what to fetch history for.

Public surface (other modules code against these signatures)::

    record_observation(symbol, ctx=None) -> None
    load_universe(since_date) -> list[str]
    seed_from_manual(symbols, session_date=None) -> None

Storage shape
-------------
Table ``"universe"`` partitioned by ``session_date``. Each row::

    symbol, observed_at (UTC now), observed_by, premarket_gap_pct,
    premarket_volume, session_date

``record_observation`` is best-effort and cheap: it is reached (indirectly) from
the live hot path, so it must not raise on absent context.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from ML_tradingAlgo.data.store import read_bars, write_bars

__all__ = ["record_observation", "load_universe", "seed_from_manual"]

_TABLE = "universe"
_SOURCE = "universe_observed"
_PARTITION_COLS = ["session_date"]
_ET = ZoneInfo("America/New_York")


def _et_today() -> dt.date:
    """Today's date in the US/Eastern trading timezone."""
    return dt.datetime.now(tz=_ET).date()


def _to_date(value) -> dt.date:
    """Normalize a date / datetime / ISO string into a plain ``date``."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def record_observation(symbol: str, ctx: dict | None = None) -> None:
    """Append a single observation row for ``symbol`` to the ``universe`` table.

    ``ctx`` (optional) may carry ``observed_by`` (default ``"live_scanner"``),
    ``premarket_gap_pct``, ``premarket_volume`` and ``session_date`` (default
    today's ET date). Best-effort: missing context is filled with defaults.
    """
    ctx = ctx or {}
    session_date = ctx.get("session_date")
    session_date = _to_date(session_date) if session_date is not None else _et_today()

    row = {
        "symbol": symbol,
        "observed_at": pd.Timestamp.now(tz="UTC"),
        "observed_by": ctx.get("observed_by", "live_scanner"),
        "premarket_gap_pct": ctx.get("premarket_gap_pct"),
        "premarket_volume": ctx.get("premarket_volume"),
        "session_date": session_date,
    }
    df = pd.DataFrame([row])
    write_bars(df, table=_TABLE, partition_cols=_PARTITION_COLS, source=_SOURCE)


def load_universe(since_date) -> list[str]:
    """Return the sorted, unique list of symbols observed on/after ``since_date``.

    ``since_date`` may be a ``date``, ``datetime`` or ISO string. Returns an
    empty list when nothing matches.
    """
    start = _to_date(since_date)
    today = _et_today()
    end = today if today >= start else start
    df = read_bars(_TABLE, date_range=(start, end))
    if df is None or len(df) == 0 or "symbol" not in df.columns:
        return []
    return sorted(df["symbol"].dropna().unique().tolist())


def seed_from_manual(symbols: list[str], session_date=None) -> None:
    """Record manual ``observed_by="manual"`` observations for ``symbols``.

    ``session_date`` defaults to today's ET date.
    """
    session_date = _to_date(session_date) if session_date is not None else _et_today()
    for symbol in symbols:
        record_observation(
            symbol,
            {"session_date": session_date, "observed_by": "manual"},
        )
