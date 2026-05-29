"""Next-day reconciliation of teed live bars against authoritative Schwab bars.

During a live session a tee writes lower-fidelity ``source="live_tick_agg"``
minute bars to the ``bars_minute`` table. This job runs the following day, pulls
the authoritative ``source="schwab_pricehistory"`` bars for the same session,
writes them alongside the live bars (so both fidelities coexist in the
warehouse), and logs how far the live bars diverged from truth.

Public surface::

    reconcile_session(session_date) -> dict   # diff-metrics summary
    main(argv=None) -> None                    # CLI entry point

CLI::

    python3 -m ML_tradingAlgo.data.reconcile --date YYYY-MM-DD

Diff metrics (returned summary dict)
------------------------------------
* ``close_mae``  -- mean absolute error of ``close`` over bars matched on
  ``(symbol, ts)`` between the live and the pricehistory frames.
* ``volume_mae`` -- mean absolute error of ``volume`` over the same matched set.
* ``n_matched``  -- number of ``(symbol, ts)`` bars present in *both* sources.
* ``n_live_bars`` / ``n_truth_bars`` -- row counts of each source for the day.

Configuration (``S3_BUCKET`` / ``S3_PREFIX`` / credentials) is read from the
environment by the underlying :mod:`store` layer; this module imports nothing
from the project ``config.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from zoneinfo import ZoneInfo

import pandas as pd

from ML_tradingAlgo.data import schwab_client
from ML_tradingAlgo.data.store import read_bars, write_bars

__all__ = ["reconcile_session", "main"]

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

_TABLE = "bars_minute"
_LIVE_SOURCE = "live_tick_agg"
_TRUTH_SOURCE = "schwab_pricehistory"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _session_bounds_utc(session_date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC bounds covering the full ET calendar day for ``session_date``.

    Returns ``(start, end)`` as tz-aware UTC Timestamps spanning ET 00:00:00 to
    23:59:59 inclusive, so extended-hours bars on both ends are captured.
    """
    start_et = dt.datetime.combine(session_date, dt.time(0, 0, 0), tzinfo=_ET)
    end_et = dt.datetime.combine(session_date, dt.time(23, 59, 59), tzinfo=_ET)
    start = pd.Timestamp(start_et).tz_convert("UTC")
    end = pd.Timestamp(end_et).tz_convert("UTC")
    return start, end


def _diff_metrics(live: pd.DataFrame, truth: pd.DataFrame) -> tuple[int, float, float]:
    """Mean-absolute close/volume error over bars matched on ``(symbol, ts)``."""
    if live.empty or truth.empty:
        return 0, 0.0, 0.0

    keys = ["symbol", "ts"]
    merged = live[keys + ["close", "volume"]].merge(
        truth[keys + ["close", "volume"]],
        on=keys,
        suffixes=("_live", "_truth"),
        how="inner",
    )
    n_matched = len(merged)
    if n_matched == 0:
        return 0, 0.0, 0.0

    close_mae = float((merged["close_live"] - merged["close_truth"]).abs().mean())
    volume_mae = float(
        (merged["volume_live"].astype("float64") - merged["volume_truth"].astype("float64"))
        .abs()
        .mean()
    )
    return n_matched, close_mae, volume_mae


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def reconcile_session(session_date) -> dict:
    """Reconcile teed live bars for ``session_date`` against Schwab truth bars.

    See module docstring for the diff-metric definitions. Returns a summary dict
    ``{session_date, symbols, n_live_bars, n_truth_bars, n_matched, close_mae,
    volume_mae}`` and logs it. When no live bars exist for the day, no Schwab
    calls are made and a zeroed summary is returned.
    """
    session_date = _to_date(session_date)

    live = read_bars(
        _TABLE,
        date_range=(session_date, session_date),
        source=_LIVE_SOURCE,
    )

    if live is None or live.empty:
        summary = {
            "session_date": session_date,
            "symbols": [],
            "n_live_bars": 0,
            "n_truth_bars": 0,
            "n_matched": 0,
            "close_mae": 0.0,
            "volume_mae": 0.0,
        }
        logger.info("reconcile_session %s: no live bars; %s", session_date, summary)
        return summary

    symbols = sorted(live["symbol"].unique().tolist())
    start, end = _session_bounds_utc(session_date)

    truth_frames = []
    for symbol in symbols:
        df = schwab_client.get_minute_bars(symbol, start, end, extended_hours=True)
        if df is not None and not df.empty:
            truth_frames.append(df)
            write_bars(
                df,
                table=_TABLE,
                partition_cols=["symbol", "session_date"],
            )

    truth = (
        pd.concat(truth_frames, ignore_index=True)
        if truth_frames
        else pd.DataFrame(columns=["symbol", "ts", "close", "volume"])
    )

    n_matched, close_mae, volume_mae = _diff_metrics(live, truth)

    summary = {
        "session_date": session_date,
        "symbols": symbols,
        "n_live_bars": int(len(live)),
        "n_truth_bars": int(len(truth)),
        "n_matched": int(n_matched),
        "close_mae": close_mae,
        "volume_mae": volume_mae,
    }
    logger.info("reconcile_session %s: %s", session_date, summary)
    return summary


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
        prog="python3 -m ML_tradingAlgo.data.reconcile",
        description="Reconcile teed live minute bars against authoritative Schwab bars.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Session date to reconcile (YYYY-MM-DD).",
    )
    args = parser.parse_args(argv)

    session_date = dt.date.fromisoformat(args.date)
    reconcile_session(session_date)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
