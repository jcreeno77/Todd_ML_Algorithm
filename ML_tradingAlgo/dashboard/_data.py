"""Cached S3 data-readers for the dashboard.

This is the **only** part of the dashboard with unit tests, so all S3/IO logic
lives here and the core functions never import or require Streamlit at runtime.

Testability vs ``st.cache_data``
--------------------------------
Each reader is implemented as a plain, undecorated function prefixed with an
underscore (``_load_events``, ``_coverage_summary`` ...). The public name
(``load_events`` ...) is a thin ``@st.cache_data`` wrapper *iff* Streamlit is
importable, otherwise it falls back to the plain function. Tests import and call
the underscore-prefixed functions directly, so caching never interferes with
direct calling outside a Streamlit runtime.

The live-state JSON under ``live/state/*.json`` is read with the same
``AWS_ENDPOINT_URL``-aware ``s3fs.S3FileSystem`` construction the store uses
(mirrored in :func:`_fs`, kept tiny on purpose).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import s3fs

from ML_tradingAlgo.data import collector, store

__all__ = [
    "load_events",
    "load_minute_bars",
    "coverage_summary",
    "fundamentals_completeness",
    "load_live_state",
    "list_live_symbols",
    # underscore-prefixed cores are exported for direct (test) use:
    "_load_events",
    "_load_minute_bars",
    "_coverage_summary",
    "_fundamentals_completeness",
    "_load_live_state",
    "_list_live_symbols",
]

# Tables tracked for the coverage overview.
_BAR_TABLES = ("bars_minute", "bars_5min", "bars_daily")
_LIVE_SOURCE = "live_tick_agg"
_MINUTE_TABLE = "bars_minute"

# Fundamentals fields whose missing-ness is the "dead-feature" risk.
_FUNDAMENTAL_FIELDS = {
    "short_interest": ("short_interest_ratio", "short_interest"),
    "sector": ("sector_id", "sector"),
    "earnings": ("earnings_date", "earnings"),
}


# --------------------------------------------------------------------------- #
# s3fs construction (mirrors store._fs so live/state JSON is read identically)
# --------------------------------------------------------------------------- #
def _fs() -> s3fs.S3FileSystem:
    """Build a fresh S3 filesystem, honouring AWS_ENDPOINT_URL for mocks."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    client_kwargs = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return s3fs.S3FileSystem(skip_instance_cache=True, client_kwargs=client_kwargs)


def _live_state_dir() -> str:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set")
    prefix = os.environ.get("S3_PREFIX", "").strip("/")
    parts = [bucket]
    if prefix:
        parts.append(prefix)
    parts += ["live", "state"]
    return "/".join(parts)


# --------------------------------------------------------------------------- #
# core readers (undecorated, directly callable in tests)
# --------------------------------------------------------------------------- #
def _load_events(date_range=None) -> pd.DataFrame:
    """Read the events table (deduped) via :func:`collector.read_events`."""
    df = collector.read_events(date_range=date_range)
    if df is None:
        return pd.DataFrame()
    return df


def _load_minute_bars(symbol, session_date, source=None) -> pd.DataFrame:
    """Read a single symbol/session's 1-minute bars, sorted by ``ts``."""
    df = store.read_bars(
        _MINUTE_TABLE,
        symbol=symbol,
        date_range=(session_date, session_date),
        source=source,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _coverage_summary() -> dict:
    """Summarise corpus coverage.

    Returns a dict with::

        n_events          -- total deduped event rows
        n_symbols         -- distinct symbols across events
        events_per_day    -- {session_date_iso: count}
        date_min/date_max -- event session-date span (ISO strings or None)
        bars_per_table    -- {table: row_count} for each tracked bar table
    """
    events = _load_events()
    n_events = int(len(events))

    if n_events and "symbol" in events.columns:
        n_symbols = int(events["symbol"].nunique())
    else:
        n_symbols = 0

    events_per_day: dict = {}
    date_min = date_max = None
    if n_events and "session_date" in events.columns:
        sd = events["session_date"].map(_iso_date)
        counts = sd.value_counts().sort_index()
        events_per_day = {str(k): int(v) for k, v in counts.items()}
        dates = sorted(d for d in sd.tolist() if d is not None)
        if dates:
            date_min, date_max = dates[0], dates[-1]

    bars_per_table = {}
    for table in _BAR_TABLES:
        df = store.read_bars(table)
        bars_per_table[table] = int(len(df)) if df is not None else 0

    return {
        "n_events": n_events,
        "n_symbols": n_symbols,
        "events_per_day": events_per_day,
        "date_min": date_min,
        "date_max": date_max,
        "bars_per_table": bars_per_table,
    }


def _fundamentals_completeness(events_df) -> dict:
    """Completeness of fundamentals fields across the given events frame.

    Surfaces the dead-feature risk: for each of ``short_interest``, ``sector``
    and ``earnings`` reports how many events carry a non-null value vs null.

    Returns ``{n_events, fields: {field: {present, missing, pct_present}}}``.
    A field whose column is entirely absent counts as fully missing.
    """
    if events_df is None or len(events_df) == 0:
        fields = {
            name: {"present": 0, "missing": 0, "pct_present": 0.0}
            for name in _FUNDAMENTAL_FIELDS
        }
        return {"n_events": 0, "fields": fields}

    n = int(len(events_df))
    fields = {}
    for name, candidates in _FUNDAMENTAL_FIELDS.items():
        col = next((c for c in candidates if c in events_df.columns), None)
        if col is None:
            present = 0
        else:
            present = int(events_df[col].notna().sum())
        missing = n - present
        fields[name] = {
            "present": present,
            "missing": missing,
            "pct_present": round(100.0 * present / n, 2) if n else 0.0,
        }
    return {"n_events": n, "fields": fields}


def _load_live_state() -> list:
    """Read all ``live/state/*.json`` snapshots from S3 as a list of dicts.

    Each dict gets an ``instance_id`` derived from the filename if absent.
    Missing/empty directory -> empty list. Malformed JSON is skipped.
    """
    fs = _fs()
    base = _live_state_dir()
    try:
        if not fs.exists(base):
            return []
        paths = fs.glob(f"{base}/*.json")
    except FileNotFoundError:
        return []

    out: list = []
    for path in sorted(paths or []):
        try:
            with fs.open(path, "rb") as f:
                obj = json.loads(f.read().decode("utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(obj, dict):
            continue
        if "instance_id" not in obj:
            fname = path.rsplit("/", 1)[-1]
            obj["instance_id"] = fname[:-5] if fname.endswith(".json") else fname
        out.append(obj)
    return out


def _list_live_symbols(session_date) -> list:
    """Symbols with ``source="live_tick_agg"`` minute bars on ``session_date``."""
    df = store.read_bars(
        _MINUTE_TABLE,
        date_range=(session_date, session_date),
        source=_LIVE_SOURCE,
    )
    if df is None or len(df) == 0 or "symbol" not in df.columns:
        return []
    return sorted(df["symbol"].unique().tolist())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _iso_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return store._to_date(value).isoformat()
    except Exception:
        return str(value)


# --------------------------------------------------------------------------- #
# public @st.cache_data wrappers (fall back to plain funcs without Streamlit)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - thin wrapper selection
    import streamlit as st

    @st.cache_data(ttl=300, show_spinner=False)
    def load_events(date_range=None) -> pd.DataFrame:
        return _load_events(date_range=date_range)

    @st.cache_data(ttl=300, show_spinner=False)
    def load_minute_bars(symbol, session_date, source=None) -> pd.DataFrame:
        return _load_minute_bars(symbol, session_date, source=source)

    @st.cache_data(ttl=60, show_spinner=False)
    def coverage_summary() -> dict:
        return _coverage_summary()

    @st.cache_data(ttl=300, show_spinner=False)
    def fundamentals_completeness(events_df) -> dict:
        return _fundamentals_completeness(events_df)

    @st.cache_data(ttl=5, show_spinner=False)
    def load_live_state() -> list:
        return _load_live_state()

    @st.cache_data(ttl=5, show_spinner=False)
    def list_live_symbols(session_date) -> list:
        return _list_live_symbols(session_date)

except Exception:  # pragma: no cover - no Streamlit available
    load_events = _load_events
    load_minute_bars = _load_minute_bars
    coverage_summary = _coverage_summary
    fundamentals_completeness = _fundamentals_completeness
    load_live_state = _load_live_state
    list_live_symbols = _list_live_symbols
