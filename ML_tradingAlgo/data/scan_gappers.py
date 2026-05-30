"""Market-wide gap-up scanner -- find recent low-float momentum runners.

Implements the Warrior-Trading-style "Gap and Go" pre-filter against the whole
US-stock universe, reusing the *exact* coarse-screen thresholds the collector /
backfill apply (so anything this surfaces is a genuine backfill candidate):

    * gap_pct = (open - prior_close) / prior_close in [GAP_MIN, GAP_MAX]   (25-50%)
    * open price in [PRICE_MIN, PRICE_MAX]                                  ($1-$30)
    * daily-volume RVOL = volume / trailing-ADV > RVOL_MIN                  (>3x)
    * float_shares < FLOAT_MAX                                             (<50M)

Cost
----
The scan pulls one *grouped daily* request per trading session (≈12k tickers
each) over the lookback window -- a handful of GETs total, not one-per-symbol.
Float enrichment then costs one reference GET per *surviving* symbol (a few
dozen), all through ``massive_client._request`` which already backs off on the
free tier's ~5/min ceiling. No S3, no per-symbol minute pulls.

Output
------
Prints a per-event table (symbol, date, gap%, open, rvol, float) and a single
comma-joined ticker list ready to paste into ``backfill --symbols``. Optionally
writes the full result to JSON via ``--out``.

Usage
-----
    python3 -m ML_tradingAlgo.data.scan_gappers --days 10
    python3 -m ML_tradingAlgo.data.scan_gappers --days 10 --out gappers.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from zoneinfo import ZoneInfo

import pandas as pd

from ML_tradingAlgo.data import massive_client as mc
from ML_tradingAlgo.data.collector import (
    GAP_MIN, GAP_MAX, PRICE_MIN, PRICE_MAX, FLOAT_MAX, RVOL_MIN,
)
from ML_tradingAlgo.data.progress import track

__all__ = ["scan_gappers", "main"]

_ET = ZoneInfo("America/New_York")


def _today() -> dt.date:
    return dt.datetime.now(tz=_ET).date()


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def _collect_sessions(end: dt.date, n_sessions: int) -> pd.DataFrame:
    """Pull grouped-daily frames backward from ``end`` until ``n_sessions`` have
    data, returning one long frame [session_date, symbol, open..volume].

    Non-trading days return empty and are skipped. A calendar buffer (~1.7x +
    pad) covers weekends/holidays so we still reach ``n_sessions``.
    """
    frames: list[pd.DataFrame] = []
    got = 0
    day = end
    max_lookback = int(n_sessions * 1.7) + 12
    for _ in range(max_lookback):
        # Markets are closed Sat/Sun -> skip without spending an API call.
        if day.weekday() >= 5:
            day = day - dt.timedelta(days=1)
            continue
        try:
            g = mc.get_grouped_daily(day)
        except RuntimeError as e:
            # Free tier blocks the current day until EOD; skip and keep going.
            if "before end of day" in str(e):
                day = day - dt.timedelta(days=1)
                continue
            raise
        if len(g):
            frames.append(g)
            got += 1
            if got >= n_sessions:
                break
        day = day - dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=mc.GROUPED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _float_shares(symbol: str) -> float | None:
    """One reference GET -> share_class_shares_outstanding (float PROXY)."""
    try:
        details = mc._ticker_details(symbol)
    except Exception:
        return None
    shares = details.get("share_class_shares_outstanding") or details.get(
        "weighted_shares_outstanding"
    )
    return float(shares) if shares is not None else None


def scan_gappers(
    end: dt.date | None = None,
    days: int = 10,
    adv_window: int = 20,
    *,
    gap_min: float = GAP_MIN,
    gap_max: float = GAP_MAX,
    price_min: float = PRICE_MIN,
    price_max: float = PRICE_MAX,
    rvol_min: float = RVOL_MIN,
    float_max: float = FLOAT_MAX,
    check_float: bool = True,
) -> dict:
    """Scan the last ``days`` trading sessions for coarse-screen-passing gappers.

    Returns ``{"events": [...], "symbols": [...], "sessions": [...]}`` where each
    event dict carries symbol/session_date/gap_pct/open/rvol/float_shares.
    """
    end = end or _today()
    long = _collect_sessions(end, days + adv_window + 1)
    if long.empty:
        return {"events": [], "symbols": [], "sessions": []}

    long = long.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    g = long.groupby("symbol", sort=False)
    long["prior_close"] = g["close"].shift(1)
    # Trailing ADV over the prior ``adv_window`` sessions (excludes event day).
    long["adv"] = (
        g["volume"].rolling(adv_window, min_periods=2).mean()
        .reset_index(level=0, drop=True)
        .groupby(long["symbol"]).shift(1)
    )

    long["gap_pct"] = (long["open"] - long["prior_close"]) / long["prior_close"]
    long["rvol"] = long["volume"] / long["adv"]

    sessions = sorted(long["session_date"].unique())
    event_sessions = set(sessions[-days:])
    cand = long[long["session_date"].isin(event_sessions)].copy()

    # --- coarse screen (mirrors collector.filter_event_coarse) -------------- #
    cand = cand[cand["prior_close"].notna() & (cand["prior_close"] > 0)]
    cand = cand[(cand["gap_pct"] >= gap_min) & (cand["gap_pct"] <= gap_max)]
    cand = cand[(cand["open"] >= price_min) & (cand["open"] <= price_max)]
    # RVOL enforced only when ADV is known (matches coarse's "when avg known").
    cand = cand[cand["adv"].isna() | (cand["rvol"] > rvol_min)]

    cand = cand.sort_values(["gap_pct"], ascending=False)

    # --- float enrichment (one reference GET per surviving symbol) ---------- #
    float_by_sym: dict[str, float | None] = {}
    if check_float:
        for sym in track(list(cand["symbol"].unique()), "scan:float"):
            float_by_sym[sym] = _float_shares(sym)

    events = []
    for _, r in cand.iterrows():
        sym = r["symbol"]
        fl = float_by_sym.get(sym) if check_float else None
        if check_float:
            # Fail closed on missing float; drop if too large (coarse semantics).
            if fl is None or fl >= float_max:
                continue
        events.append({
            "symbol": str(sym),
            "session_date": str(r["session_date"]),
            "gap_pct": round(float(r["gap_pct"]), 4),
            "open": round(float(r["open"]), 4),
            "volume": int(r["volume"]),
            "rvol": None if pd.isna(r["rvol"]) else round(float(r["rvol"]), 2),
            "float_shares": None if fl is None else int(fl),
        })

    # Unique symbols, preserving descending-gap order of first appearance.
    seen: list[str] = []
    for e in events:
        if e["symbol"] not in seen:
            seen.append(e["symbol"])

    return {
        "events": events,
        "symbols": seen,
        "sessions": [str(s) for s in sessions[-days:]],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(
        prog="python -m ML_tradingAlgo.data.scan_gappers",
        description="Market-wide low-float gap-up scanner (Warrior-style pre-filter).",
    )
    p.add_argument("--end", default=None, help="Last session to scan (YYYY-MM-DD); default today ET.")
    p.add_argument("--days", type=int, default=10, help="Event-window trading sessions (default 10).")
    p.add_argument("--adv-window", type=int, default=20, help="Trailing sessions for RVOL ADV (default 20).")
    p.add_argument("--gap-min", type=float, default=GAP_MIN)
    p.add_argument("--gap-max", type=float, default=GAP_MAX)
    p.add_argument("--price-min", type=float, default=PRICE_MIN)
    p.add_argument("--price-max", type=float, default=PRICE_MAX)
    p.add_argument("--rvol-min", type=float, default=RVOL_MIN)
    p.add_argument("--float-max", type=float, default=float(FLOAT_MAX))
    p.add_argument("--no-float", action="store_true", help="Skip float lookup (faster; backfill still screens float).")
    p.add_argument("--out", default=None, help="Optional path to write full JSON result.")
    args = p.parse_args(argv)

    end = dt.date.fromisoformat(args.end) if args.end else None
    res = scan_gappers(
        end=end, days=args.days, adv_window=args.adv_window,
        gap_min=args.gap_min, gap_max=args.gap_max,
        price_min=args.price_min, price_max=args.price_max,
        rvol_min=args.rvol_min, float_max=args.float_max,
        check_float=not args.no_float,
    )

    events, symbols = res["events"], res["symbols"]
    if res["sessions"]:
        print(f"scanned sessions: {res['sessions'][0]} .. {res['sessions'][-1]} "
              f"({len(res['sessions'])} days)")
    print(f"gap-up events: {len(events)} | unique symbols: {len(symbols)}\n")
    if events:
        print(f"{'SYMBOL':<8}{'DATE':<12}{'GAP%':>8}{'OPEN':>9}{'RVOL':>8}{'FLOAT(M)':>11}")
        for e in events:
            fl = "" if e["float_shares"] is None else f"{e['float_shares']/1e6:.1f}"
            rv = "" if e["rvol"] is None else f"{e['rvol']:.1f}"
            print(f"{e['symbol']:<8}{e['session_date']:<12}"
                  f"{e['gap_pct']*100:>7.1f}%{e['open']:>9.2f}{rv:>8}{fl:>11}")
        print(f"\n--symbols \"{','.join(symbols)}\"")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
