"""Nightly data collector / orchestrator for the momentum gap-up pipeline.

This module turns the daily *gainers* universe into a set of refined trading
**events** persisted in the warehouse. It is built around two pure, I/O-free
filter functions (easily unit-tested) and a thin orchestrator that wires them to
the storage / market-data / fundamentals layers.

Pipeline (see :func:`run_nightly`)
----------------------------------
1. Load the recent universe from :mod:`gainers`.
2. Per symbol: pull daily bars incrementally (watermark), refresh the
   fundamentals snapshot, compute a 20-day average daily volume.
3. Pass 1 -- :func:`filter_event_coarse` on the asof-date daily bar -> candidates.
4. Per candidate: pull extended-hours 1-minute bars; Pass 2 --
   :func:`filter_event_fine`.
5. Write one row per candidate to the ``events`` table, plus the pulled
   daily/minute bars to ``bars_daily`` / ``bars_minute``.
6. Advance the per-symbol watermark as the final step.

Filter thresholds
-----------------
* Coarse (daily only):
    - gap_pct = (open - prior_close) / prior_close in [0.25, 0.50]
    - open price in [1, 30]
    - float_shares < 50_000_000 (FAIL CLOSED if float missing)
    - daily-volume RVOL proxy (volume / avg_daily_volume_20d) > 3 when avg known
* Fine (minute data):
    - premarket_high/low/volume from bars before 09:30 ET
    - rvol_at_open = first-30-min volume / (avg_daily_volume_20d * 30/390)
    - missing premarket rows -> fields None + reason, but does NOT fail the event
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import math

import pandas as pd

from ML_tradingAlgo.data.store import (
    read_bars,
    write_bars,
    get_watermark,
    set_watermark,
)
from ML_tradingAlgo.data import schwab_client
from ML_tradingAlgo.data import gainers
from ML_tradingAlgo.data import fundamentals

__all__ = [
    "EventDecision",
    "filter_event_coarse",
    "filter_event_fine",
    "run_nightly",
    "read_events",
    "main",
]

_ET = ZoneInfo("America/New_York")

# Filter thresholds.
GAP_MIN = 0.25
GAP_MAX = 0.50
PRICE_MIN = 1.0
PRICE_MAX = 30.0
FLOAT_MAX = 50_000_000
RVOL_MIN = 3.0

# Universe lookback for the nightly run.
UNIVERSE_LOOKBACK_DAYS = 7
# Daily-bar history pulled to compute the 20-day average daily volume.
DAILY_HISTORY_DAYS = 60
AVG_VOLUME_WINDOW = 20
# Fraction of a regular session (390 min) covered by the first 30 minutes.
OPEN_WINDOW_MIN = 30
SESSION_MIN = 390

EVENTS_TABLE = "events"
DAILY_TABLE = "bars_daily"
MINUTE_TABLE = "bars_minute"


# --------------------------------------------------------------------------- #
# Decision container
# --------------------------------------------------------------------------- #
@dataclass
class EventDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


# --------------------------------------------------------------------------- #
# Pure filter: coarse (daily only)
# --------------------------------------------------------------------------- #
def filter_event_coarse(
    daily_bar: dict,
    prior_close: float,
    fundamentals: dict,
    avg_daily_volume_20d: float | None = None,
) -> EventDecision:
    """Coarse daily-only screen. Returns an :class:`EventDecision`.

    Reasons are accumulated for every failing check; an empty ``reasons`` list
    with ``passed=True`` means the symbol cleared the coarse screen.
    """
    reasons: list[str] = []

    daily_open = daily_bar.get("open")

    # --- gap_pct ---------------------------------------------------------- #
    if _is_missing(daily_open) or _is_missing(prior_close) or prior_close == 0:
        reasons.append("gap_uncomputable")
    else:
        gap_pct = (daily_open - prior_close) / prior_close
        if not (GAP_MIN <= gap_pct <= GAP_MAX):
            reasons.append(f"gap_out_of_band({gap_pct:.4f})")

    # --- price ------------------------------------------------------------ #
    if _is_missing(daily_open):
        reasons.append("price_unknown")
    elif not (PRICE_MIN <= daily_open <= PRICE_MAX):
        reasons.append(f"price_out_of_band({daily_open})")

    # --- float (fail closed) --------------------------------------------- #
    float_shares = fundamentals.get("float_shares") if fundamentals else None
    if _is_missing(float_shares):
        reasons.append("float_unknown")
    elif float_shares >= FLOAT_MAX:
        reasons.append(f"float_too_large({float_shares})")

    # --- daily-volume RVOL proxy (only when avg available) --------------- #
    if not _is_missing(avg_daily_volume_20d) and avg_daily_volume_20d:
        volume = daily_bar.get("volume")
        if _is_missing(volume):
            reasons.append("volume_unknown")
        else:
            rvol = volume / avg_daily_volume_20d
            if rvol <= RVOL_MIN:
                reasons.append(f"rvol_below_threshold({rvol:.2f})")

    return EventDecision(passed=len(reasons) == 0, reasons=reasons)


# --------------------------------------------------------------------------- #
# Pure filter: fine (minute data)
# --------------------------------------------------------------------------- #
def filter_event_fine(
    minute_bars: pd.DataFrame,
    coarse: EventDecision,
    avg_daily_volume_20d: float | None,
) -> tuple[EventDecision, dict]:
    """Refine a coarse decision using minute bars.

    Computes premarket high/low/volume (bars strictly before 09:30 ET) and
    ``rvol_at_open`` (first-30-min regular-session volume vs the pro-rated
    average daily volume). Missing premarket rows are reported via the
    ``premarket_unavailable`` reason but never fail the event on their own.

    Returns ``(EventDecision, fields)`` where ``fields`` carries the computed
    values used by the events row.
    """
    reasons = list(coarse.reasons)
    fields = {
        "premarket_high": None,
        "premarket_low": None,
        "premarket_volume": None,
        "rvol_at_open": None,
        "open_price": None,
    }

    if minute_bars is None or len(minute_bars) == 0:
        reasons.append("premarket_unavailable")
        return EventDecision(passed=coarse.passed, reasons=reasons), fields

    df = minute_bars.copy()
    ts = pd.to_datetime(df["ts"], utc=True)
    et = ts.dt.tz_convert(_ET)
    minutes = et.dt.hour * 60 + et.dt.minute

    open_min = 9 * 60 + 30  # 09:30 ET
    premarket = df[minutes < open_min]
    regular = df[minutes >= open_min]

    # --- premarket fields ------------------------------------------------- #
    if len(premarket) == 0:
        reasons.append("premarket_unavailable")
    else:
        fields["premarket_high"] = float(premarket["high"].max())
        fields["premarket_low"] = float(premarket["low"].min())
        fields["premarket_volume"] = int(premarket["volume"].sum())

    # --- open price (first regular-session open) -------------------------- #
    if len(regular) > 0:
        first_open = regular.sort_values("ts").iloc[0]["open"]
        fields["open_price"] = float(first_open)

        # --- rvol_at_open ------------------------------------------------- #
        if not _is_missing(avg_daily_volume_20d) and avg_daily_volume_20d:
            first_30 = regular.sort_values("ts").head(OPEN_WINDOW_MIN)
            open_volume = float(first_30["volume"].sum())
            expected = avg_daily_volume_20d * OPEN_WINDOW_MIN / SESSION_MIN
            if expected:
                fields["rvol_at_open"] = open_volume / expected

    return EventDecision(passed=coarse.passed, reasons=reasons), fields


# --------------------------------------------------------------------------- #
# data-pull helpers
# --------------------------------------------------------------------------- #
def _compute_avg_daily_volume(symbol: str, asof_date: dt.date) -> float | None:
    """Average daily volume over the last ``AVG_VOLUME_WINDOW`` stored days.

    Uses the daily bars already persisted in the warehouse (strictly before the
    asof date so the event-day's volume does not contaminate its own baseline).
    """
    start = asof_date - dt.timedelta(days=DAILY_HISTORY_DAYS)
    daily = read_bars(DAILY_TABLE, symbol=symbol, date_range=(start, asof_date))
    if daily is None or len(daily) == 0 or "volume" not in daily.columns:
        return None
    prior = daily[daily["session_date"].map(_to_date) < asof_date]
    if len(prior) == 0:
        return None
    recent = prior.sort_values("session_date").tail(AVG_VOLUME_WINDOW)
    vol = pd.to_numeric(recent["volume"], errors="coerce").dropna()
    if len(vol) == 0:
        return None
    return float(vol.mean())


def _merge_fundamentals(symbol: str, asof_date: dt.date,
                        schwab_fund: pd.DataFrame) -> dict:
    """Merge Schwab float/52wk with the fundamentals short-interest/sector dict."""
    fund: dict = {}
    if schwab_fund is not None and len(schwab_fund):
        match = schwab_fund[schwab_fund["symbol"] == symbol]
        if len(match):
            row = match.iloc[0]
            fund["float_shares"] = row.get("float_shares")
            fund["shares_outstanding"] = row.get("shares_outstanding")
            fund["high_52wk"] = row.get("high_52wk")
            fund["low_52wk"] = row.get("low_52wk")

    extra = fundamentals.get_fundamentals(symbol, asof_date)
    fund["short_interest_ratio"] = extra.get("short_interest_ratio")
    fund["sector_id"] = extra.get("sector_id")
    fund["earnings_date"] = extra.get("earnings_date")
    return fund


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_nightly(asof_date) -> dict:
    """Run the nightly collection for ``asof_date`` and return a summary dict."""
    # Heads-up if the Schwab refresh token is near its 7-day hard expiry. Routed
    # through the channel-agnostic notifier (logs by default, Discord webhook if
    # configured); never blocks the run.
    try:
        from ML_tradingAlgo.data.token_health import check_token_freshness
        from ML_tradingAlgo.data.notify import notify
        check_token_freshness(notify=notify)
    except Exception:
        pass

    asof = _to_date(asof_date)
    since = asof - dt.timedelta(days=UNIVERSE_LOOKBACK_DAYS)

    symbols = gainers.load_universe(since)
    n_symbols = len(symbols)

    candidates: list[dict] = []
    pulled_daily: dict[str, pd.DataFrame] = {}

    # --- Pass 1: coarse on daily bars ------------------------------------ #
    for symbol in symbols:
        watermark = get_watermark(DAILY_TABLE, symbol)
        if watermark is not None:
            start = _to_date(watermark) - dt.timedelta(days=1)
        else:
            start = asof - dt.timedelta(days=DAILY_HISTORY_DAYS)

        daily = schwab_client.get_daily_bars(symbol, start, asof)
        if daily is not None and len(daily):
            write_bars(daily, table=DAILY_TABLE,
                       partition_cols=["symbol", "session_date"])
        pulled_daily[symbol] = daily

        # refresh fundamentals snapshot (persisted by fundamentals module)
        schwab_fund = schwab_client.get_fundamentals([symbol])
        schwab_fields = {}
        if schwab_fund is not None and len(schwab_fund):
            m = schwab_fund[schwab_fund["symbol"] == symbol]
            if len(m):
                r = m.iloc[0]
                schwab_fields = {
                    "float_shares": r.get("float_shares"),
                    "high_52wk": r.get("high_52wk"),
                    "low_52wk": r.get("low_52wk"),
                }
        fundamentals.write_snapshot(symbol, asof, schwab_fields)

        fund = _merge_fundamentals(symbol, asof, schwab_fund)
        avg_vol = _compute_avg_daily_volume(symbol, asof)

        # locate the asof-date daily bar + the prior close
        asof_bar, prior_close = _split_asof_and_prior(daily, asof)
        if asof_bar is None:
            continue

        coarse = filter_event_coarse(asof_bar, prior_close, fund,
                                     avg_daily_volume_20d=avg_vol)
        if coarse.passed:
            candidates.append({
                "symbol": symbol,
                "daily_bar": asof_bar,
                "prior_close": prior_close,
                "fundamentals": fund,
                "avg_vol": avg_vol,
                "coarse": coarse,
            })

    # --- Pass 2: fine on minute bars ------------------------------------- #
    event_rows: list[dict] = []
    detected_at = pd.Timestamp.now(tz="UTC")
    n_events_passed = 0

    for cand in candidates:
        symbol = cand["symbol"]
        start = dt.datetime.combine(asof, dt.time(0, 0))
        end = dt.datetime.combine(asof, dt.time(23, 59))
        minute = schwab_client.get_minute_bars(symbol, start, end,
                                               extended_hours=True)
        if minute is not None and len(minute):
            write_bars(minute, table=MINUTE_TABLE,
                       partition_cols=["symbol", "session_date"])

        fine, fields = filter_event_fine(minute, cand["coarse"], cand["avg_vol"])

        gap_pct = _gap_pct(cand["daily_bar"], cand["prior_close"])
        open_price = fields["open_price"]
        if open_price is None:
            open_price = cand["daily_bar"].get("open")

        if fine.passed:
            n_events_passed += 1

        event_rows.append({
            "symbol": symbol,
            "session_date": asof,
            "prior_close": cand["prior_close"],
            "open_price": open_price,
            "gap_pct": gap_pct,
            "premarket_high": fields["premarket_high"],
            "premarket_low": fields["premarket_low"],
            "premarket_volume": fields["premarket_volume"],
            "float_shares": cand["fundamentals"].get("float_shares"),
            "rvol_at_open": fields["rvol_at_open"],
            "passed_filters": bool(fine.passed),
            "filter_reasons": ";".join(fine.reasons),
            "detected_at": detected_at,
        })

    # --- write events ---------------------------------------------------- #
    if event_rows:
        events_df = pd.DataFrame(event_rows)
        write_bars(events_df, table=EVENTS_TABLE,
                   partition_cols=["session_date"], source="collector")

    # --- advance watermarks (LAST step) ---------------------------------- #
    for symbol in symbols:
        daily = pulled_daily.get(symbol)
        if daily is not None and len(daily) and "ts" in daily.columns:
            last_ts = pd.to_datetime(daily["ts"], utc=True).max()
            set_watermark(DAILY_TABLE, symbol, last_ts)

    return {
        "asof_date": asof,
        "n_symbols": n_symbols,
        "n_candidates": len(candidates),
        "n_events_passed": n_events_passed,
    }


def _gap_pct(daily_bar: dict, prior_close: float) -> float | None:
    open_ = daily_bar.get("open")
    if _is_missing(open_) or _is_missing(prior_close) or prior_close == 0:
        return None
    return (open_ - prior_close) / prior_close


def _split_asof_and_prior(daily: pd.DataFrame, asof: dt.date):
    """Return ``(asof_bar_dict, prior_close)`` from the pulled daily frame."""
    if daily is None or len(daily) == 0:
        return None, None
    df = daily.copy()
    df["_sd"] = df["session_date"].map(_to_date)
    df = df.sort_values("_sd")
    asof_rows = df[df["_sd"] == asof]
    if len(asof_rows) == 0:
        return None, None
    asof_row = asof_rows.iloc[-1]
    asof_bar = {
        "open": asof_row.get("open"),
        "high": asof_row.get("high"),
        "low": asof_row.get("low"),
        "close": asof_row.get("close"),
        "volume": asof_row.get("volume"),
    }
    prior_rows = df[df["_sd"] < asof]
    prior_close = (
        prior_rows.iloc[-1]["close"] if len(prior_rows) else None
    )
    return asof_bar, prior_close


# --------------------------------------------------------------------------- #
# events read helper (dedupe on (symbol, session_date), keep latest detected_at)
# --------------------------------------------------------------------------- #
def read_events(date_range=None) -> pd.DataFrame:
    """Read the ``events`` table deduped on ``(symbol, session_date)``.

    The store's default append-only model can leave multiple rows for the same
    ``(symbol, session_date)`` after reruns. This helper collapses them, keeping
    the row with the latest ``detected_at`` (falling back to ``ingested_at``).
    """
    df = read_bars(EVENTS_TABLE, date_range=date_range)
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()

    df = df.copy()
    if "detected_at" in df.columns:
        order_col = "detected_at"
        df[order_col] = pd.to_datetime(df[order_col], utc=True)
    elif "ingested_at" in df.columns:
        order_col = "ingested_at"
    else:
        order_col = None

    subset = [c for c in ("symbol", "session_date") if c in df.columns]
    if not subset:
        return df.reset_index(drop=True)

    if order_col is not None:
        df = df.sort_values(order_col, kind="stable")
    df = df.drop_duplicates(subset=subset, keep="last")
    return df.sort_values(subset).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        prog="python -m ML_tradingAlgo.data.collector",
        description="Run the nightly gap-up event collector.",
    )
    parser.add_argument(
        "--date", required=True,
        help="As-of session date in YYYY-MM-DD format.",
    )
    args = parser.parse_args(argv)
    asof = dt.date.fromisoformat(args.date)
    summary = run_nightly(asof)
    print(
        f"collector run asof={summary['asof_date']} "
        f"symbols={summary['n_symbols']} "
        f"candidates={summary['n_candidates']} "
        f"events_passed={summary['n_events_passed']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
