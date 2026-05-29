"""Pluggable fundamentals data source for the TFT momentum trader.

The model needs three *static* features that Schwab does not provide:

    short_interest_ratio   -- days-to-cover short interest ratio (float)
    sector_id              -- GICS sector as an int 0..10 (categorical)
    days_since_earnings    -- derived downstream from ``earnings_date``

This module supplies them from a pluggable external source (the FMP REST API
or a failure-tolerant Yahoo Finance scrape).

CRITICAL RULE
-------------
When a field is unavailable it is recorded as ``None`` AND its name is added to
the ``_missing`` list with the field never silently defaulting to ``0``.
``0`` is a real GICS sector id and ``0.0`` is a plausible ratio, so a silent
zero would become an invisible *dead* feature in ``compute_static_features``
(``ML_tradingAlgo/tft/features.py``), which consumes these raw values.

Environment
-----------
Read directly via ``os.environ.get`` (this module deliberately does NOT import
the project ``config.py``):

    FUNDAMENTALS_PROVIDER   "fmp" | "yahoo"   (default "yahoo")
    FMP_API_KEY             required by the FMP provider
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import requests

__all__ = [
    "GICS_SECTORS",
    "sector_name_to_id",
    "FundamentalsProvider",
    "FMPProvider",
    "YahooProvider",
    "get_provider",
    "get_fundamentals",
    "days_since_earnings",
    "write_snapshot",
]


# --------------------------------------------------------------------------- #
# GICS sector mapping
# --------------------------------------------------------------------------- #
# The 11 GICS sectors mapped to a stable contiguous int 0..10. The ordering is
# the canonical GICS sector list; ``categorical_cardinalities=[11]`` in the
# model config (see tests/conftest.py) matches this size. Note that ``0`` is a
# *real* sector ("Energy"): an unknown / missing sector must therefore map to
# ``None`` (and be marked missing), never to ``0``.
GICS_SECTORS: dict[str, int] = {
    "Energy": 0,
    "Materials": 1,
    "Industrials": 2,
    "Consumer Discretionary": 3,
    "Consumer Staples": 4,
    "Health Care": 5,
    "Financials": 6,
    "Information Technology": 7,
    "Communication Services": 8,
    "Utilities": 9,
    "Real Estate": 10,
}

# A few common aliases that providers emit for the canonical names above.
_SECTOR_ALIASES: dict[str, str] = {
    "healthcare": "Health Care",
    "technology": "Information Technology",
    "financial services": "Financials",
    "financial": "Financials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
    "communication services": "Communication Services",
    "communications": "Communication Services",
    "industrial goods": "Industrials",
}


def sector_name_to_id(name: Optional[str]) -> Optional[int]:
    """Map a GICS sector *name* to its int id, or ``None`` if unrecognised.

    Returns ``None`` (never ``0``) for empty / unknown input so callers can
    record an explicit miss instead of silently producing a dead feature.
    """
    if not name or not str(name).strip():
        return None
    raw = str(name).strip()
    if raw in GICS_SECTORS:
        return GICS_SECTORS[raw]
    alias = _SECTOR_ALIASES.get(raw.lower())
    if alias is not None:
        return GICS_SECTORS[alias]
    return None


# --------------------------------------------------------------------------- #
# HTTP helper (single choke point so tests can patch it)
# --------------------------------------------------------------------------- #
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HTTP_TIMEOUT = 15
_HTTP_RETRIES = 2  # number of *additional* attempts after the first


def _http_get(url: str) -> str:
    """Perform an HTTP GET and return the response body as text.

    All network access in this module routes through here so that tests can
    patch a single function with canned payloads. A realistic User-Agent is
    sent because Yahoo blocks default library agents.
    """
    resp = requests.get(
        url, headers={"User-Agent": _USER_AGENT}, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.text


def _http_get_resilient(url: str) -> Optional[str]:
    """Call ``_http_get`` with a couple of retries; return ``None`` on failure.

    Never raises -- on persistent failure the caller marks the field missing.
    """
    last_exc = None
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            return _http_get(url)
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            last_exc = exc
            if attempt < _HTTP_RETRIES:
                time.sleep(0.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# date helpers
# --------------------------------------------------------------------------- #
def _to_date(value) -> Optional[dt.date]:
    """Coerce ``value`` (date / datetime / ISO-ish string) to a ``date``."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    if not s:
        return None
    # Try ISO first, then a couple of common human formats (Yahoo style).
    for fmt in (None, "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            if fmt is None:
                return dt.date.fromisoformat(s[:10])
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def days_since_earnings(earnings_date, session_date) -> Optional[float]:
    """Calendar days from ``earnings_date`` to ``session_date``.

    Returns ``None`` if either date is missing/unparseable (never ``0``, since
    ``0`` legitimately means "earnings were today").
    """
    earn = _to_date(earnings_date)
    sess = _to_date(session_date)
    if earn is None or sess is None:
        return None
    return float((sess - earn).days)


def _most_recent_earnings(dates, asof: dt.date) -> Optional[dt.date]:
    """Pick the latest earnings date on or before ``asof`` from ``dates``."""
    candidates = [d for d in (_to_date(x) for x in dates) if d is not None]
    past = [d for d in candidates if d <= asof]
    if past:
        return max(past)
    # No past earnings -> fall back to the earliest known date, if any.
    return min(candidates) if candidates else None


# --------------------------------------------------------------------------- #
# Provider abstraction
# --------------------------------------------------------------------------- #
class FundamentalsProvider(ABC):
    """A source of raw fundamentals fields for a single symbol.

    Implementations return a *raw* dict and never raise on missing data; the
    normalisation/miss-marking happens in :func:`get_fundamentals`.
    """

    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, asof_date) -> dict:
        """Return raw provider fields for ``symbol`` as of ``asof_date``.

        Expected (best-effort) keys:
            sector                -> str | None
            short_interest_ratio  -> float | None
            earnings_date         -> date | None
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# FMP provider
# --------------------------------------------------------------------------- #
class FMPProvider(FundamentalsProvider):
    """Financial Modeling Prep REST API provider.

    Endpoints used (FMP ``/api/v3``):
        /profile/{symbol}          -> company profile; ``sector`` field
        /short-interest/{symbol}   -> ``shortInterestRatio`` field
        /earning_calendar          -> historical earnings dates (``date``)

    Requires ``FMP_API_KEY``.
    """

    name = "fmp"
    BASE = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.environ.get("FMP_API_KEY")

    def _get_json(self, url: str):
        text = _http_get_resilient(url)
        if text is None:
            return None
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None

    def fetch(self, symbol: str, asof_date) -> dict:
        asof = _to_date(asof_date)
        key = self.api_key or ""

        # --- sector (profile) ---
        sector = None
        profile = self._get_json(f"{self.BASE}/profile/{symbol}?apikey={key}")
        if isinstance(profile, list) and profile:
            sector = profile[0].get("sector")
        elif isinstance(profile, dict):
            sector = profile.get("sector")

        # --- short interest ratio ---
        sir = None
        short = self._get_json(f"{self.BASE}/short-interest/{symbol}?apikey={key}")
        if isinstance(short, list) and short:
            sir = short[0].get("shortInterestRatio")
        elif isinstance(short, dict):
            sir = short.get("shortInterestRatio")

        # --- earnings date (most recent on/before asof) ---
        earnings_date = None
        earnings = self._get_json(
            f"{self.BASE}/earning_calendar?symbol={symbol}&apikey={key}"
        )
        if isinstance(earnings, list) and earnings:
            earnings_date = _most_recent_earnings(
                (e.get("date") for e in earnings if isinstance(e, dict)),
                asof or dt.date.today(),
            )

        return {
            "sector": sector,
            "short_interest_ratio": float(sir) if sir is not None else None,
            "earnings_date": earnings_date,
        }


# --------------------------------------------------------------------------- #
# Yahoo provider
# --------------------------------------------------------------------------- #
class YahooProvider(FundamentalsProvider):
    """Yahoo Finance scraping provider (failure tolerant).

    Pages scraped:
        /quote/{symbol}/profile           -> sector
        /quote/{symbol}/key-statistics    -> "Short Ratio"
        /quote/{symbol}                    -> earnings date

    Each page is fetched via :func:`_http_get_resilient`; any failure leaves the
    corresponding field ``None`` (marked missing by :func:`get_fundamentals`).
    """

    name = "yahoo"
    BASE = "https://finance.yahoo.com/quote"

    def _sector(self, symbol: str) -> Optional[str]:
        html = _http_get_resilient(f"{self.BASE}/{symbol}/profile")
        if not html:
            return None
        import re

        m = re.search(
            r'data-test="qsp-sector"[^>]*>([^<]+)<', html, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        # alternate markup: "Sector(s)" label followed by a value span.
        m = re.search(
            r"Sector\(s\)[^<]*</span>\s*<span[^>]*>([^<]+)<", html, re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        return None

    def _short_ratio(self, symbol: str) -> Optional[float]:
        html = _http_get_resilient(f"{self.BASE}/{symbol}/key-statistics")
        if not html:
            return None
        import re

        m = re.search(
            r"Short Ratio[^<]*</td>\s*<td[^>]*>([\d.,]+)<", html, re.IGNORECASE
        )
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    def _earnings_date(self, symbol: str) -> Optional[dt.date]:
        html = _http_get_resilient(f"{self.BASE}/{symbol}")
        if not html:
            return None
        import re

        m = re.search(
            r'data-test="EARNINGS_DATE"[^>]*>([^<]+)<', html, re.IGNORECASE
        )
        if m:
            return _to_date(m.group(1).strip())
        return None

    def fetch(self, symbol: str, asof_date) -> dict:
        return {
            "sector": self._sector(symbol),
            "short_interest_ratio": self._short_ratio(symbol),
            "earnings_date": self._earnings_date(symbol),
        }


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
def get_provider() -> FundamentalsProvider:
    """Return the configured provider (env ``FUNDAMENTALS_PROVIDER``).

    Defaults to Yahoo. Raises ``ValueError`` for an unrecognised value.
    """
    name = os.environ.get("FUNDAMENTALS_PROVIDER", "yahoo").strip().lower()
    if name == "yahoo":
        return YahooProvider()
    if name == "fmp":
        return FMPProvider()
    raise ValueError(
        f"Unknown FUNDAMENTALS_PROVIDER={name!r}; expected 'fmp' or 'yahoo'"
    )


# --------------------------------------------------------------------------- #
# Normalised public API
# --------------------------------------------------------------------------- #
def get_fundamentals(symbol: str, asof_date) -> dict:
    """Fetch and normalise fundamentals for ``symbol`` as of ``asof_date``.

    Returns a dict with the stable shape::

        {
            "symbol": str,
            "short_interest_ratio": float | None,
            "sector_id": int | None,           # GICS 0..10
            "earnings_date": date | None,
            "_missing": list[str],             # field names that were unavailable
            "_source": str,                    # provider name
            "_fetched_at": pd.Timestamp,       # UTC
        }

    Any unavailable field is ``None`` and listed in ``_missing`` -- never ``0``.
    """
    provider = get_provider()
    try:
        raw = provider.fetch(symbol, asof_date)
    except Exception:  # noqa: BLE001 - providers should not raise, but be safe
        raw = {}

    missing: list[str] = []

    # short_interest_ratio
    sir = raw.get("short_interest_ratio")
    if sir is None:
        missing.append("short_interest_ratio")

    # sector_id (mapped from name; unknown -> None + missing, never 0)
    sector_name = raw.get("sector")
    sector_id = sector_name_to_id(sector_name)
    if sector_id is None:
        missing.append("sector_id")

    # earnings_date
    earnings_date = _to_date(raw.get("earnings_date"))
    if earnings_date is None:
        missing.append("earnings_date")

    return {
        "symbol": symbol,
        "short_interest_ratio": sir,
        "sector_id": sector_id,
        "earnings_date": earnings_date,
        "_missing": missing,
        "_source": provider.name,
        "_fetched_at": pd.Timestamp.now(tz="UTC"),
    }


# --------------------------------------------------------------------------- #
# Store write (thin)
# --------------------------------------------------------------------------- #
def write_snapshot(symbol: str, asof_date, schwab_fields: dict) -> None:
    """Merge this module's fundamentals with Schwab fields and persist one row.

    ``schwab_fields`` supplies ``float_shares, high_52wk, low_52wk``. The merged
    row is written to the ``fundamentals`` table partitioned by ``year`` (derived
    from ``asof_date``).

    The store is imported lazily so this module imports cleanly before
    ``data/store.py`` exists, and so tests can monkeypatch the store.
    """
    from ML_tradingAlgo.data.store import write_bars  # lazy: store may not exist yet

    fund = get_fundamentals(symbol, asof_date)
    asof = _to_date(asof_date)

    row = {
        "symbol": symbol,
        "asof_date": asof,
        "year": asof.year if asof is not None else None,
        "short_interest_ratio": fund["short_interest_ratio"],
        "sector_id": fund["sector_id"],
        "earnings_date": fund["earnings_date"],
        # merged Schwab-supplied fields
        "float_shares": schwab_fields.get("float_shares"),
        "high_52wk": schwab_fields.get("high_52wk"),
        "low_52wk": schwab_fields.get("low_52wk"),
        # provenance
        "missing": ",".join(fund["_missing"]),
        "source": fund["_source"],
        "fetched_at": fund["_fetched_at"],
    }

    df = pd.DataFrame([row])
    write_bars(
        df,
        table="fundamentals",
        partition_cols=["year"],
        source="fundamentals_merged",
    )
