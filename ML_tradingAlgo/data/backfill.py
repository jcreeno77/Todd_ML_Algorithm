"""Seed backfill: bootstrap the historical corpus for the momentum pipeline.

This module turns a *curated seed list* of historically-known gappers into the
same refined ``events`` rows (plus daily/minute bars) that the nightly
:mod:`ML_tradingAlgo.data.collector` produces -- but over a long historical
date range rather than a single as-of night.

Reuse over duplication
----------------------
We deliberately do **not** re-implement any filter or merge logic. The whole
detection pipeline lives in :mod:`collector`; this module is a thin orchestrator
that calls collector's *public* pure filters and its module-level helpers:

    * ``collector.filter_event_coarse`` / ``collector.filter_event_fine`` -- the
      two pure screens (Pass 1 / Pass 2).
    * ``collector._merge_fundamentals`` -- Schwab float/52wk merged with the
      fundamentals short-interest/sector dict.
    * ``collector._compute_avg_daily_volume`` -- 20-day ADV from the warehouse.
    * ``collector._split_asof_and_prior`` / ``collector._gap_pct`` -- locate the
      event-day daily bar and its prior close, and compute the gap.
    * ``collector.read_events`` -- deduped event reads on (symbol, session_date).
    * the table-name constants ``EVENTS_TABLE`` / ``DAILY_TABLE`` / ``MINUTE_TABLE``.

We do **not** loop ``collector.run_nightly`` per date: that would (a) re-pull
daily bars once per (symbol, date) instead of once per symbol for the whole
range, and (b) always attempt the minute pull, ignoring Schwab's ~6-month
1-minute history limit. Instead we pull daily bars **once per symbol** across
the full window, then walk each day reusing the helpers above.

Minute cutoff
-------------
Schwab only serves ~6 months of 1-minute data. For each coarse-passing event we
pull minute bars + run the fine screen **only** when the event date is within
``minute_cutoff_days`` of today. Older events keep their daily-only decision and
record a ``minute_unavailable`` reason; no minute request is issued for them.

Idempotent & resumable
----------------------
A per-symbol daily watermark (``store.get_watermark`` / ``set_watermark`` on the
``bars_daily`` table) records how far a symbol has been processed. If a previous
run already advanced the watermark to/under ``end_date`` the symbol is skipped on
re-run. Event writes are append-only but ``collector.read_events`` dedupes on
``(symbol, session_date)``, so a partial-then-resumed run never yields duplicate
events.
"""

from __future__ import annotations

import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from ML_tradingAlgo.data import collector, gainers, schwab_client, fundamentals
from ML_tradingAlgo.data.store import (
    read_bars,
    write_bars,
    get_watermark,
    set_watermark,
)
from ML_tradingAlgo.data.collector import (
    filter_event_coarse,
    filter_event_fine,
    read_events,
)

__all__ = ["backfill_seed", "main"]

_ET = ZoneInfo("America/New_York")

# Reuse collector's table names so we write to the exact same warehouse tables.
EVENTS_TABLE = collector.EVENTS_TABLE
DAILY_TABLE = collector.DAILY_TABLE
MINUTE_TABLE = collector.MINUTE_TABLE


def _today() -> dt.date:
    """Today's date in the US/Eastern trading timezone (patchable in tests)."""
    return dt.datetime.now(tz=_ET).date()


def _to_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def backfill_seed(
    symbols: list[str],
    start_date,
    end_date,
    minute_cutoff_days: int = 180,
) -> dict:
    """Bootstrap the corpus for a curated seed list of known gappers.

    For each symbol the daily history over ``[start_date, end_date]`` is pulled
    **once**, then every trading day in the window is screened with collector's
    coarse filter. Coarse-passing days within ``minute_cutoff_days`` of today
    also get a minute pull + fine screen; older days are kept daily-only with a
    ``minute_unavailable`` reason. Returns a summary dict.
    """
    start = _to_date(start_date)
    end = _to_date(end_date)
    today = _today()
    cutoff = today - dt.timedelta(days=minute_cutoff_days)

    # 1. Seed the universe so these symbols are "known".
    gainers.seed_from_manual(symbols, session_date=start)

    n_events = 0
    n_minute_unavailable = 0
    detected_at = pd.Timestamp.now(tz="UTC")

    for symbol in symbols:
        # --- resume: skip symbols already processed through end_date -------- #
        watermark = get_watermark(DAILY_TABLE, symbol)
        if watermark is not None and _to_date(watermark) >= end:
            continue

        # --- ONE daily pull per symbol for the whole window ---------------- #
        daily = schwab_client.get_daily_bars(symbol, start, end)
        if daily is not None and len(daily):
            write_bars(daily, table=DAILY_TABLE,
                       partition_cols=["symbol", "session_date"])

        # --- refresh fundamentals snapshot (once per symbol) --------------- #
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
        fundamentals.write_snapshot(symbol, end, schwab_fields)

        if daily is None or len(daily) == 0:
            continue

        # candidate event dates = each daily session_date in the window.
        event_dates = sorted(
            {d for d in daily["session_date"].map(_to_date)
             if start <= d <= end}
        )

        event_rows: list[dict] = []
        for event_date in event_dates:
            # Reuse collector to locate the event-day bar + prior close.
            asof_bar, prior_close = collector._split_asof_and_prior(daily, event_date)
            if asof_bar is None:
                continue

            fund = collector._merge_fundamentals(symbol, event_date, schwab_fund)
            avg_vol = collector._compute_avg_daily_volume(symbol, event_date)

            coarse = filter_event_coarse(asof_bar, prior_close, fund,
                                         avg_daily_volume_20d=avg_vol)
            if not coarse.passed:
                continue

            # --- minute cutoff ------------------------------------------- #
            within_cutoff = event_date >= cutoff
            if within_cutoff:
                day_start = dt.datetime.combine(event_date, dt.time(0, 0))
                day_end = dt.datetime.combine(event_date, dt.time(23, 59))
                minute = schwab_client.get_minute_bars(
                    symbol, day_start, day_end, extended_hours=True
                )
                if minute is not None and len(minute):
                    write_bars(minute, table=MINUTE_TABLE,
                               partition_cols=["symbol", "session_date"])
                fine, fields = filter_event_fine(minute, coarse, avg_vol)
            else:
                # Older than Schwab's 1-min window: keep daily-only.
                fine = collector.EventDecision(
                    passed=coarse.passed,
                    reasons=list(coarse.reasons) + ["minute_unavailable"],
                )
                fields = {
                    "premarket_high": None, "premarket_low": None,
                    "premarket_volume": None, "rvol_at_open": None,
                    "open_price": None,
                }
                n_minute_unavailable += 1

            open_price = fields["open_price"]
            if open_price is None:
                open_price = asof_bar.get("open")

            event_rows.append({
                "symbol": symbol,
                "session_date": event_date,
                "prior_close": prior_close,
                "open_price": open_price,
                "gap_pct": collector._gap_pct(asof_bar, prior_close),
                "premarket_high": fields["premarket_high"],
                "premarket_low": fields["premarket_low"],
                "premarket_volume": fields["premarket_volume"],
                "float_shares": fund.get("float_shares"),
                "rvol_at_open": fields["rvol_at_open"],
                "passed_filters": bool(fine.passed),
                "filter_reasons": ";".join(fine.reasons),
                "detected_at": detected_at,
            })

        if event_rows:
            events_df = pd.DataFrame(event_rows)
            write_bars(events_df, table=EVENTS_TABLE,
                       partition_cols=["session_date"], source="backfill")
            n_events += len(event_rows)

        # --- advance the watermark (LAST step, mirrors collector) ---------- #
        if "ts" in daily.columns and len(daily):
            last_ts = pd.to_datetime(daily["ts"], utc=True).max()
            set_watermark(DAILY_TABLE, symbol, last_ts)

    return {
        "n_symbols": len(symbols),
        "n_events": n_events,
        "n_minute_unavailable": n_minute_unavailable,
        "date_range": (start, end),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ML_tradingAlgo.data.backfill",
        description="Bootstrap the historical corpus from a curated seed list.",
    )
    parser.add_argument(
        "--symbols", required=True,
        help="Comma-separated seed symbols, e.g. AAA,BBB,CCC.",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--minute-cutoff-days", type=int, default=180,
        help="Events older than this many days skip the 1-min pull (default 180).",
    )
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    summary = backfill_seed(
        symbols, start, end, minute_cutoff_days=args.minute_cutoff_days
    )
    print(
        f"backfill symbols={summary['n_symbols']} "
        f"events={summary['n_events']} "
        f"minute_unavailable={summary['n_minute_unavailable']} "
        f"range={summary['date_range'][0]}..{summary['date_range'][1]} "
        f"[{','.join(symbols)}]"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
