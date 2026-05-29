"""Tests for the pluggable fundamentals data source.

These tests never hit the real network. All HTTP is routed through
``fundamentals._http_get`` which is monkeypatched with canned FMP / Yahoo
payloads. The store write is also mocked.

Core invariant under test: when a field is unavailable it is recorded as
``None`` and its name appears in ``_missing`` -- it is NEVER silently
defaulted to ``0`` (because ``0`` is a real GICS sector / a real ratio and a
silent zero becomes an invisible dead feature downstream).
"""

import datetime as dt
import json

import pandas as pd
import pytest

from ML_tradingAlgo.data import fundamentals


# --------------------------------------------------------------------------- #
# Canned payloads
# --------------------------------------------------------------------------- #
FMP_PROFILE = [
    {
        "symbol": "AAPL",
        "sector": "Information Technology",
        "companyName": "Apple Inc.",
    }
]

FMP_SHORT_INTEREST = [
    {
        "symbol": "AAPL",
        "shortInterestRatio": 1.8,
    }
]

FMP_EARNINGS = [
    {"symbol": "AAPL", "date": "2024-05-02"},
    {"symbol": "AAPL", "date": "2024-08-01"},
    {"symbol": "AAPL", "date": "2024-02-01"},
]


YAHOO_PROFILE_HTML = """
<html><body>
<span data-test="qsp-sector">Information Technology</span>
<h3>Apple Inc.</h3>
</body></html>
"""

# Yahoo statistics page: short interest ratio shows as "Short Ratio".
YAHOO_STATS_HTML = """
<html><body>
<table>
<tr><td class="label">Short Ratio (date)</td><td class="value">2.34</td></tr>
</table>
</body></html>
"""

# Yahoo quote/calendar: earnings date.
YAHOO_CALENDAR_HTML = """
<html><body>
<td data-test="EARNINGS_DATE">May 02, 2024</td>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("FUNDAMENTALS_PROVIDER", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)


def _fmp_router(payloads):
    """Return an _http_get replacement routing FMP URLs to canned JSON."""

    def _get(url):
        if "profile" in url:
            return json.dumps(payloads.get("profile", []))
        if "earning" in url:
            return json.dumps(payloads.get("earnings", []))
        if "short" in url.lower():
            return json.dumps(payloads.get("short", []))
        raise AssertionError(f"unexpected FMP url: {url}")

    return _get


def _yahoo_router(pages):
    """Return an _http_get replacement routing Yahoo URLs to canned HTML."""

    def _get(url):
        if "key-statistics" in url or "statistics" in url:
            return pages.get("stats", "")
        if "profile" in url:
            return pages.get("profile", "")
        # default / quote page used for earnings
        return pages.get("calendar", "")

    return _get


# --------------------------------------------------------------------------- #
# GICS mapping
# --------------------------------------------------------------------------- #
def test_gics_mapping_has_11_sectors():
    assert len(fundamentals.GICS_SECTORS) == 11
    # ids must be exactly 0..10
    assert sorted(fundamentals.GICS_SECTORS.values()) == list(range(11))


def test_gics_name_to_id():
    assert fundamentals.sector_name_to_id("Energy") == fundamentals.GICS_SECTORS["Energy"]
    # known sector maps to a valid int
    sid = fundamentals.sector_name_to_id("Information Technology")
    assert isinstance(sid, int)
    assert 0 <= sid <= 10


def test_gics_unknown_sector_returns_none_not_zero():
    assert fundamentals.sector_name_to_id("Nonexistent Sector") is None
    assert fundamentals.sector_name_to_id(None) is None
    assert fundamentals.sector_name_to_id("") is None


def test_gics_zero_is_a_real_sector():
    # sector id 0 must correspond to an actual sector name, proving 0 is real.
    inv = {v: k for k, v in fundamentals.GICS_SECTORS.items()}
    assert 0 in inv
    assert inv[0]  # non-empty name


# --------------------------------------------------------------------------- #
# days_since_earnings
# --------------------------------------------------------------------------- #
def test_days_since_earnings_basic():
    earn = dt.date(2024, 5, 2)
    session = dt.date(2024, 5, 12)
    assert fundamentals.days_since_earnings(earn, session) == 10.0


def test_days_since_earnings_accepts_strings():
    assert fundamentals.days_since_earnings("2024-05-02", "2024-05-12") == 10.0


def test_days_since_earnings_none_returns_none():
    assert fundamentals.days_since_earnings(None, dt.date(2024, 5, 12)) is None
    assert fundamentals.days_since_earnings(dt.date(2024, 5, 2), None) is None


# --------------------------------------------------------------------------- #
# FMP provider
# --------------------------------------------------------------------------- #
def test_fmp_normalize_full(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": FMP_PROFILE,
                "short": FMP_SHORT_INTEREST,
                "earnings": FMP_EARNINGS,
            }
        ),
    )
    prov = fundamentals.FMPProvider()
    raw = prov.fetch("AAPL", dt.date(2024, 5, 12))

    assert raw["sector"] == "Information Technology"
    assert raw["short_interest_ratio"] == 1.8
    # the most recent earnings date on/before asof_date is 2024-05-02
    assert str(raw["earnings_date"]) == "2024-05-02"


def test_fmp_get_fundamentals_normalized(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": FMP_PROFILE,
                "short": FMP_SHORT_INTEREST,
                "earnings": FMP_EARNINGS,
            }
        ),
    )
    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["symbol"] == "AAPL"
    assert out["short_interest_ratio"] == 1.8
    assert out["sector_id"] == fundamentals.GICS_SECTORS["Information Technology"]
    assert str(out["earnings_date"]) == "2024-05-02"
    assert out["_missing"] == []
    assert out["_source"] == "fmp"
    assert "_fetched_at" in out


def test_fmp_missing_short_interest_marked_not_zero(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": FMP_PROFILE,
                "short": [],  # no short-interest data
                "earnings": FMP_EARNINGS,
            }
        ),
    )
    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["short_interest_ratio"] is None
    assert out["short_interest_ratio"] != 0
    assert "short_interest_ratio" in out["_missing"]


def test_fmp_unknown_sector_marked_missing(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    bad_profile = [{"symbol": "AAPL", "sector": "Imaginary Sector"}]
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": bad_profile,
                "short": FMP_SHORT_INTEREST,
                "earnings": FMP_EARNINGS,
            }
        ),
    )
    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["sector_id"] is None
    assert "sector_id" in out["_missing"]


# --------------------------------------------------------------------------- #
# Yahoo provider
# --------------------------------------------------------------------------- #
def test_yahoo_normalize_full(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "yahoo")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _yahoo_router(
            {
                "profile": YAHOO_PROFILE_HTML,
                "stats": YAHOO_STATS_HTML,
                "calendar": YAHOO_CALENDAR_HTML,
            }
        ),
    )
    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["symbol"] == "AAPL"
    assert out["sector_id"] == fundamentals.GICS_SECTORS["Information Technology"]
    assert out["short_interest_ratio"] == 2.34
    assert str(out["earnings_date"]) == "2024-05-02"
    assert out["_source"] == "yahoo"
    assert out["_missing"] == []


def test_yahoo_http_failure_marks_missing_not_zero(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "yahoo")

    def _boom(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(fundamentals, "_http_get", _boom)
    # patch sleep so retries don't slow the test
    monkeypatch.setattr(fundamentals.time, "sleep", lambda *_a, **_k: None)

    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["short_interest_ratio"] is None
    assert out["sector_id"] is None
    assert out["earnings_date"] is None
    assert "short_interest_ratio" in out["_missing"]
    assert "sector_id" in out["_missing"]
    assert "earnings_date" in out["_missing"]
    # the fields are None, never 0
    assert out["short_interest_ratio"] != 0


def test_yahoo_partial_missing_short_interest(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "yahoo")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _yahoo_router(
            {
                "profile": YAHOO_PROFILE_HTML,
                "stats": "<html><body>no short ratio here</body></html>",
                "calendar": YAHOO_CALENDAR_HTML,
            }
        ),
    )
    out = fundamentals.get_fundamentals("AAPL", dt.date(2024, 5, 12))

    assert out["sector_id"] == fundamentals.GICS_SECTORS["Information Technology"]
    assert out["short_interest_ratio"] is None
    assert "short_interest_ratio" in out["_missing"]


# --------------------------------------------------------------------------- #
# provider selection
# --------------------------------------------------------------------------- #
def test_get_provider_default_is_yahoo(monkeypatch):
    monkeypatch.delenv("FUNDAMENTALS_PROVIDER", raising=False)
    prov = fundamentals.get_provider()
    assert isinstance(prov, fundamentals.YahooProvider)


def test_get_provider_fmp(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    prov = fundamentals.get_provider()
    assert isinstance(prov, fundamentals.FMPProvider)


def test_get_provider_case_insensitive(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "FMP")
    assert isinstance(fundamentals.get_provider(), fundamentals.FMPProvider)


def test_get_provider_unknown_raises(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        fundamentals.get_provider()


# --------------------------------------------------------------------------- #
# write_snapshot
# --------------------------------------------------------------------------- #
def test_write_snapshot_merges_and_writes(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": FMP_PROFILE,
                "short": FMP_SHORT_INTEREST,
                "earnings": FMP_EARNINGS,
            }
        ),
    )

    captured = {}

    def fake_write_bars(df, table=None, partition_cols=None, source=None):
        captured["df"] = df
        captured["table"] = table
        captured["partition_cols"] = partition_cols
        captured["source"] = source

    # The store module does not exist yet; create a fake module so the lazy
    # import inside write_snapshot resolves to it.
    import sys
    import types

    fake_store = types.ModuleType("ML_tradingAlgo.data.store")
    fake_store.write_bars = fake_write_bars
    monkeypatch.setitem(sys.modules, "ML_tradingAlgo.data.store", fake_store)

    schwab_fields = {
        "float_shares": 1.5e9,
        "high_52wk": 199.62,
        "low_52wk": 124.17,
    }
    fundamentals.write_snapshot("AAPL", dt.date(2024, 5, 12), schwab_fields)

    assert captured["table"] == "fundamentals"
    assert captured["partition_cols"] == ["year"]
    df = captured["df"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    # merged schwab fields
    assert row["float_shares"] == 1.5e9
    assert row["high_52wk"] == 199.62
    assert row["low_52wk"] == 124.17
    # this module's fields
    assert row["symbol"] == "AAPL"
    assert row["short_interest_ratio"] == 1.8
    assert row["sector_id"] == fundamentals.GICS_SECTORS["Information Technology"]
    # partition column derived from asof_date
    assert row["year"] == 2024


def test_write_snapshot_preserves_missing_as_none(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_PROVIDER", "fmp")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        fundamentals,
        "_http_get",
        _fmp_router(
            {
                "profile": [{"symbol": "AAPL", "sector": "Imaginary"}],
                "short": [],
                "earnings": [],
            }
        ),
    )

    captured = {}

    def fake_write_bars(df, **kwargs):
        captured["df"] = df

    import sys
    import types

    fake_store = types.ModuleType("ML_tradingAlgo.data.store")
    fake_store.write_bars = fake_write_bars
    monkeypatch.setitem(sys.modules, "ML_tradingAlgo.data.store", fake_store)

    fundamentals.write_snapshot(
        "AAPL", dt.date(2024, 5, 12), {"float_shares": 1.0, "high_52wk": 2.0, "low_52wk": 0.5}
    )
    row = captured["df"].iloc[0]
    assert pd.isna(row["short_interest_ratio"])
    assert pd.isna(row["sector_id"])
