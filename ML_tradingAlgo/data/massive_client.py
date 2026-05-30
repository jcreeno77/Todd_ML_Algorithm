"""Thin wrapper over the Massive.com market-data REST API (Polygon-compatible).

Drop-in alternative to :mod:`schwab_client` with the SAME public surface and the
SAME canonical price frame (see ``PRICE_COLUMNS``), so the backfill/collector can
switch sources via the ``DATA_PROVIDER`` env var without any downstream change:

    get_minute_bars(symbol, start, end, extended_hours=True) -> pd.DataFrame
    get_daily_bars(symbol, start, end)                       -> pd.DataFrame
    get_fundamentals(symbols)                                -> pd.DataFrame

Why Massive over Schwab here: the free tier serves ~2 years of 1-minute history
(vs Schwab's ~180 days) and authenticates with a plain API key (no OAuth flow).

Auth: ``MASSIVE_API_KEY`` passed as the ``apiKey`` query parameter.

Rate limits: the free tier throttles (~5 req/min) with HTTP 429. We avoid 429s
*proactively* with a token bucket (``_throttle``, ``MASSIVE_RATE_PER_MIN``) paced
to the ceiling, and cache closed sessions' grouped-daily frames to disk
(``MASSIVE_CACHE_DIR``) so repeated scans cost no calls. The reactive
``Retry-After`` backoff remains as a fallback. See ``docs/massive-data.md``.

float_shares caveat: Massive's reference endpoint exposes ``share_class_shares_outstanding``,
NOT true public float. We use shares-outstanding as a float PROXY (current, not
as-of-date). Good enough for a baseline screen; flagged so it can be tightened later.
52-week high/low are computed from a trailing daily-aggregate pull.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import httpx

__all__ = [
    "get_minute_bars",
    "get_daily_bars",
    "get_fundamentals",
    "PRICE_COLUMNS",
]

_ET = ZoneInfo("America/New_York")
_BASE = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com")

# Mirrors schwab_client.PRICE_COLUMNS exactly so both sources write one schema.
PRICE_COLUMNS = [
    "ts", "session_date", "symbol",
    "open", "high", "low", "close", "volume", "vwap", "trade_count", "source",
]

_PRICE_SOURCE = "massive_aggregates"
_FUNDAMENTAL_SOURCE = "massive_reference"

# 429 backoff: free tier is ~5 req/min, so default sleep covers a full window.
# This is the *reactive* fallback; the proactive token bucket below should keep
# us from ever hitting a 429 under normal use.
_RATE_SLEEP_S = 15.0
_MAX_RETRIES = 8

# --- proactive rate limit (avoid 429s instead of reacting to them) ---------- #
# Token bucket paced at the free-tier ceiling. Bursts up to capacity, then emits
# one token per (60 / rate) seconds. Pacing at ~5/min is *faster* than bursting
# into 429s, because the reactive backoff (15s) costs more than the 12s spacing.
# Set MASSIVE_RATE_PER_MIN=0 to disable (e.g. on a paid, uncapped plan).
_RATE_PER_MIN = float(os.environ.get("MASSIVE_RATE_PER_MIN", "5"))
_bucket_lock = threading.Lock()
_tokens = _RATE_PER_MIN
_last_refill = time.monotonic()

# --- grouped-daily disk cache (a closed session's data never changes) ------- #
# One grouped call returns ~12k tickers for a date; caching closed days means a
# repeated/overlapping scan costs zero API calls. Today's (still-open) day is
# never cached. Set MASSIVE_CACHE_DIR to relocate; empty string disables.
_CACHE_DIR = os.environ.get(
    "MASSIVE_CACHE_DIR", str(Path.home() / ".cache" / "massive_grouped")
)


def _throttle() -> None:
    """Block until a rate-limit token is available (no-op if disabled)."""
    global _tokens, _last_refill
    if _RATE_PER_MIN <= 0:
        return
    refill_per_s = _RATE_PER_MIN / 60.0
    with _bucket_lock:
        while True:
            now = time.monotonic()
            _tokens = min(_RATE_PER_MIN, _tokens + (now - _last_refill) * refill_per_s)
            _last_refill = now
            if _tokens >= 1.0:
                _tokens -= 1.0
                return
            time.sleep((1.0 - _tokens) / refill_per_s)


def _api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY not set in environment / .env")
    return key


def _request(path_or_url: str, params: dict | None = None) -> dict:
    """GET a Massive endpoint with 429/5xx backoff; return parsed JSON.

    Accepts either a path (prefixed with ``_BASE``) or an absolute ``next_url``
    for pagination. ``apiKey`` is always injected.
    """
    url = path_or_url if path_or_url.startswith("http") else _BASE + path_or_url
    p = dict(params or {})
    p["apiKey"] = _api_key()

    for attempt in range(_MAX_RETRIES + 1):
        _throttle()  # proactively pace to the free-tier ceiling
        resp = httpx.get(url, params=p, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 and attempt < _MAX_RETRIES:
            retry_after = resp.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else _RATE_SLEEP_S)
            continue
        if resp.status_code >= 500 and attempt < _MAX_RETRIES:
            time.sleep(min(2 ** attempt, 30))
            continue
        # 403 NOT_AUTHORIZED (plan depth), 404, etc. -> surface as error.
        raise RuntimeError(
            f"Massive GET {url} -> {resp.status_code}: {resp.text[:200]}"
        )
    raise RuntimeError(f"Massive GET {url} exhausted retries (last {resp.status_code})")


# --------------------------------------------------------------------------- #
# aggregate -> canonical price frame
# --------------------------------------------------------------------------- #
def _empty_price_frame() -> pd.DataFrame:
    empty = pd.DataFrame(columns=PRICE_COLUMNS)
    empty["ts"] = pd.to_datetime(empty["ts"], utc=True)
    empty["trade_count"] = empty["trade_count"].astype("Int32")
    return empty


def _aggs_to_frame(results: list, symbol: str) -> pd.DataFrame:
    """Convert Massive/Polygon aggregate ``results`` to the canonical frame.

    Bar fields: ``t`` (epoch ms, window start), ``o/h/l/c`` prices, ``v`` volume,
    ``vw`` vwap, ``n`` transaction count.
    """
    if not results:
        return _empty_price_frame()

    df = pd.DataFrame(results)
    ts = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    session_date = ts.dt.tz_convert(_ET).dt.date

    out = pd.DataFrame({
        "ts": ts,
        "session_date": session_date,
        "symbol": symbol,
        "open": df["o"].astype("float64"),
        "high": df["h"].astype("float64"),
        "low": df["l"].astype("float64"),
        "close": df["c"].astype("float64"),
        # volume may be fractional in the feed; round to whole shares.
        "volume": np.rint(df["v"].astype("float64")).astype("int64"),
    })
    out["vwap"] = df["vw"].astype("float64") if "vw" in df.columns else np.float64(np.nan)
    out["trade_count"] = (
        df["n"].astype("Int32") if "n" in df.columns
        else pd.array([pd.NA] * len(out), dtype="Int32")
    )
    out["source"] = _PRICE_SOURCE
    return out[PRICE_COLUMNS]


def _fmt(d) -> str:
    """Coerce a date/datetime/str to a Massive ``YYYY-MM-DD`` range token (ET day)."""
    if isinstance(d, dt.datetime):
        d = d.date()
    if isinstance(d, dt.date):
        return d.isoformat()
    return pd.Timestamp(d).date().isoformat()


def _fetch_aggs(symbol, multiplier, timespan, start, end) -> pd.DataFrame:
    path = (
        f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
        f"{_fmt(start)}/{_fmt(end)}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    frames = []
    payload = _request(path, params)
    while True:
        frames.append(_aggs_to_frame(payload.get("results") or [], symbol))
        next_url = payload.get("next_url")
        if not next_url:
            break
        payload = _request(next_url)  # apiKey re-injected by _request
    if not frames:
        return _empty_price_frame()
    out = pd.concat(frames, ignore_index=True)
    return out[PRICE_COLUMNS]


# --------------------------------------------------------------------------- #
# Public API: price history
# --------------------------------------------------------------------------- #
def get_minute_bars(symbol, start, end, extended_hours: bool = True) -> pd.DataFrame:
    """Per-minute OHLCV bars for ``symbol`` between ``start`` and ``end`` (ET days).

    Massive aggregates already include pre/regular/after-hours bars, so
    ``extended_hours`` is accepted for surface-compatibility but is a no-op.
    """
    return _fetch_aggs(symbol, 1, "minute", start, end)


def get_daily_bars(symbol, start, end) -> pd.DataFrame:
    """Daily OHLCV bars for ``symbol`` between ``start`` and ``end``."""
    return _fetch_aggs(symbol, 1, "day", start, end)


# --------------------------------------------------------------------------- #
# Public API: whole-market grouped daily (one call per date)
# --------------------------------------------------------------------------- #
# Grouped daily columns: per-ticker OHLCV for an entire session in one request.
GROUPED_COLUMNS = [
    "session_date", "symbol", "open", "high", "low", "close", "volume",
]


def get_grouped_daily(date) -> pd.DataFrame:
    """Daily OHLCV for *every* US-stock ticker on ``date`` in a single request.

    Polygon-compatible grouped-aggregates endpoint. Returns columns
    ``GROUPED_COLUMNS`` (one row per ticker). A non-trading day (weekend /
    holiday) yields an empty frame. ``volume`` is rounded to whole shares.

    This is the workhorse for market-wide scans: one GET covers ~12k tickers,
    so a multi-day gap scan costs one request per session rather than one per
    symbol -- and never touches the ~5/min per-symbol rate ceiling per ticker.
    """
    day = _fmt(date)
    sess = pd.Timestamp(day).date()
    # A closed session is immutable -> safe to cache. Today is still open.
    closed = sess < dt.datetime.now(tz=_ET).date()
    cache_path = Path(_CACHE_DIR) / f"{day}.parquet" if _CACHE_DIR else None

    if closed and cache_path is not None and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # corrupt/unreadable cache -> re-fetch

    payload = _request(
        f"/v2/aggs/grouped/locale/us/market/stocks/{day}",
        {"adjusted": "true"},
    )
    results = payload.get("results") or []
    if not results:
        out = pd.DataFrame(columns=GROUPED_COLUMNS)
    else:
        df = pd.DataFrame(results)
        out = pd.DataFrame({
            "session_date": sess,
            "symbol": df["T"].astype("string"),
            "open": df["o"].astype("float64"),
            "high": df["h"].astype("float64"),
            "low": df["l"].astype("float64"),
            "close": df["c"].astype("float64"),
            "volume": np.rint(df["v"].astype("float64")).astype("int64"),
        })[GROUPED_COLUMNS]

    # Cache closed days (even empty holidays) so reruns skip the call entirely.
    if closed and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache_path, index=False)
        except Exception:
            pass

    return out


# --------------------------------------------------------------------------- #
# Public API: fundamentals (float proxy + 52-week range)
# --------------------------------------------------------------------------- #
def _ticker_details(symbol: str) -> dict:
    payload = _request(f"/v3/reference/tickers/{symbol}")
    res = payload.get("results")
    return res if isinstance(res, dict) else {}


def get_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """Float/52wk reference data for ``symbols`` (schwab_client-compatible shape).

    ``float_shares`` uses ``share_class_shares_outstanding`` as a PROXY (Massive
    does not expose true float). 52-week high/low are computed from a trailing
    ~365-day daily pull (max high / min low). Returns columns:
    ``[symbol, float_shares, shares_outstanding, high_52wk, low_52wk, source]``.
    """
    today = dt.datetime.now(tz=_ET).date()
    year_ago = today - dt.timedelta(days=365)

    rows = []
    for sym in symbols:
        details = _ticker_details(sym)
        shares = details.get("share_class_shares_outstanding") or details.get(
            "weighted_shares_outstanding"
        )
        float_proxy = float(shares) if shares is not None else np.nan

        hi = lo = np.nan
        daily = get_daily_bars(sym, year_ago, today)
        if daily is not None and len(daily):
            hi = float(daily["high"].max())
            lo = float(daily["low"].min())

        rows.append({
            "symbol": sym,
            "float_shares": float_proxy,
            "shares_outstanding": shares,
            "high_52wk": hi,
            "low_52wk": lo,
            "source": _FUNDAMENTAL_SOURCE,
        })

    return pd.DataFrame(
        rows,
        columns=["symbol", "float_shares", "shares_outstanding",
                 "high_52wk", "low_52wk", "source"],
    )
