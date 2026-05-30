"""Tests for the Schwab market-data API wrapper.

These tests never hit the real Schwab API. The schwab-py client is replaced
with a ``MagicMock`` whose methods return canned ``Response``-like objects that
mimic the JSON shapes Schwab returns.
"""

import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ML_tradingAlgo.data import schwab_client

# Capture the real implementation before any fixture monkeypatches it.
_REAL_GET_CLIENT = schwab_client._get_client


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


# Epoch-millis for known UTC instants (computed independently of the impl).
# 2024-01-02 14:30:00 UTC == 2024-01-02 09:30:00 ET (EST, winter, UTC-5)
TS_WINTER_MS = int(
    dt.datetime(2024, 1, 2, 14, 30, tzinfo=dt.timezone.utc).timestamp() * 1000
)
# 2024-07-01 13:30:00 UTC == 2024-07-01 09:30:00 ET (EDT, summer, UTC-4)
TS_SUMMER_MS = int(
    dt.datetime(2024, 7, 1, 13, 30, tzinfo=dt.timezone.utc).timestamp() * 1000
)
# A premarket bar in winter: 2024-01-02 09:00:00 UTC == 04:00 ET (premarket)
TS_PREMARKET_MS = int(
    dt.datetime(2024, 1, 2, 9, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
)


def _minute_payload(symbol="AAPL"):
    return {
        "symbol": symbol,
        "empty": False,
        "candles": [
            {
                "datetime": TS_WINTER_MS,
                "open": 100.0,
                "high": 101.5,
                "low": 99.5,
                "close": 101.0,
                "volume": 12345,
            },
            {
                "datetime": TS_SUMMER_MS,
                "open": 200.0,
                "high": 202.0,
                "low": 199.0,
                "close": 201.0,
                "volume": 54321,
            },
        ],
    }


@pytest.fixture
def fake_client():
    """A MagicMock schwab-py client with sensible default return values."""
    client = MagicMock()
    client.get_price_history_every_minute.return_value = FakeResponse(_minute_payload())
    client.get_price_history_every_day.return_value = FakeResponse(_minute_payload())
    client.get_instruments.return_value = FakeResponse(
        {
            "AAPL": {
                "fundamental": {
                    "marketCapFloat": 1500.0,  # in millions
                    "sharesOutstanding": 16_000_000_000,
                    "high52": 199.62,
                    "low52": 124.17,
                },
                "symbol": "AAPL",
            }
        }
    )
    client.get_quotes.return_value = FakeResponse(
        {
            "AAPL": {
                "symbol": "AAPL",
                "quote": {
                    "lastPrice": 187.5,
                    "bidPrice": 187.4,
                    "askPrice": 187.6,
                    "totalVolume": 45_000_000,
                    "52WeekHigh": 199.62,
                    "52WeekLow": 124.17,
                },
            }
        }
    )
    client.get_movers.return_value = FakeResponse(
        {
            "screeners": [
                {
                    "symbol": "GAPR",
                    "description": "Gapper Inc",
                    "lastPrice": 4.20,
                    "netChange": 1.40,
                    "netPercentChange": 0.50,  # +50% intraday (fraction)
                    "volume": 8_000_000,
                    "totalVolume": 9_000_000,
                    "trades": 1234,
                    "marketShare": 0.12,
                },
                {
                    "symbol": "RUNR",
                    "description": "Runner Corp",
                    "lastPrice": 12.00,
                    "netChange": 3.00,
                    "netPercentChange": 0.33,
                    "volume": 4_000_000,
                    "totalVolume": 5_000_000,
                    "trades": 567,
                    "marketShare": 0.07,
                },
            ]
        }
    )
    return client


@pytest.fixture(autouse=True)
def patch_client(monkeypatch, fake_client):
    """Force ``_get_client`` to return our mock, and reset the cache."""
    monkeypatch.setattr(schwab_client, "_CLIENT", None, raising=False)
    monkeypatch.setattr(schwab_client, "_get_client", lambda: fake_client)
    # Make backoff sleeps instantaneous.
    monkeypatch.setattr(schwab_client.time, "sleep", lambda *_a, **_k: None)
    return fake_client


PRICE_COLS = [
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


# --------------------------------------------------------------------------- #
# get_minute_bars
# --------------------------------------------------------------------------- #
def test_minute_bars_schema_exact(patch_client):
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert list(df.columns) == PRICE_COLS


def test_minute_bars_epoch_to_utc(patch_client):
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    ts0 = df["ts"].iloc[0]
    assert ts0 == pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    assert ts0.tzinfo is not None
    assert str(df["ts"].dt.tz) == "UTC"


def test_minute_bars_session_date_dst_handling(patch_client):
    """session_date must be the ET trading date, honouring DST."""
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    # Winter bar: 14:30 UTC -> 09:30 ET on 2024-01-02 (EST, UTC-5)
    assert df["session_date"].iloc[0] == dt.date(2024, 1, 2)
    # Summer bar: 13:30 UTC -> 09:30 ET on 2024-07-01 (EDT, UTC-4)
    assert df["session_date"].iloc[1] == dt.date(2024, 7, 1)


def test_minute_bars_session_date_is_date_type(patch_client):
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert isinstance(df["session_date"].iloc[0], dt.date)


def test_minute_bars_dtypes_and_source(patch_client):
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert df["open"].dtype == np.float64
    assert df["high"].dtype == np.float64
    assert df["low"].dtype == np.float64
    assert df["close"].dtype == np.float64
    assert df["volume"].dtype == np.int64
    assert df["vwap"].dtype == np.float64
    assert str(df["trade_count"].dtype) == "Int32"
    assert (df["source"] == "schwab_pricehistory").all()
    assert (df["symbol"] == "AAPL").all()


def test_minute_bars_vwap_nan_when_absent(patch_client):
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert df["vwap"].isna().all()


def test_minute_bars_extended_hours_flag_passed_through(patch_client):
    schwab_client.get_minute_bars(
        "AAPL",
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 7, 2),
        extended_hours=True,
    )
    _, kwargs = patch_client.get_price_history_every_minute.call_args
    assert kwargs["need_extended_hours_data"] is True


def test_minute_bars_extended_hours_false(patch_client):
    schwab_client.get_minute_bars(
        "AAPL",
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 7, 2),
        extended_hours=False,
    )
    _, kwargs = patch_client.get_price_history_every_minute.call_args
    assert kwargs["need_extended_hours_data"] is False


def test_minute_bars_passes_start_and_end(patch_client):
    start = dt.datetime(2024, 1, 1)
    end = dt.datetime(2024, 7, 2)
    schwab_client.get_minute_bars("AAPL", start, end)
    _, kwargs = patch_client.get_price_history_every_minute.call_args
    assert kwargs["start_datetime"] == start
    assert kwargs["end_datetime"] == end


def test_minute_bars_premarket_session_date(patch_client):
    """A 04:00 ET premarket bar still belongs to that ET calendar date."""
    patch_client.get_price_history_every_minute.return_value = FakeResponse(
        {
            "symbol": "AAPL",
            "candles": [
                {
                    "datetime": TS_PREMARKET_MS,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                }
            ],
        }
    )
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)
    )
    # 09:00 UTC -> 04:00 ET on 2024-01-02
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-02 09:00:00", tz="UTC")
    assert df["session_date"].iloc[0] == dt.date(2024, 1, 2)


def test_minute_bars_empty_candles(patch_client):
    patch_client.get_price_history_every_minute.return_value = FakeResponse(
        {"symbol": "AAPL", "empty": True, "candles": []}
    )
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)
    )
    assert list(df.columns) == PRICE_COLS
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# get_daily_bars
# --------------------------------------------------------------------------- #
def test_daily_bars_schema_exact(patch_client):
    df = schwab_client.get_daily_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert list(df.columns) == PRICE_COLS


def test_daily_bars_uses_daily_endpoint(patch_client):
    schwab_client.get_daily_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert patch_client.get_price_history_every_day.called
    assert not patch_client.get_price_history_every_minute.called


def test_daily_bars_utc_and_session_date(patch_client):
    df = schwab_client.get_daily_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["session_date"].iloc[1] == dt.date(2024, 7, 1)


# --------------------------------------------------------------------------- #
# get_fundamentals
# --------------------------------------------------------------------------- #
FUND_COLS = [
    "symbol",
    "float_shares",
    "shares_outstanding",
    "high_52wk",
    "low_52wk",
    "source",
]


def test_fundamentals_schema_exact(patch_client):
    df = schwab_client.get_fundamentals(["AAPL"])
    assert list(df.columns) == FUND_COLS


def test_fundamentals_float_scaling(patch_client):
    """marketCapFloat (millions) must be multiplied by 1e6, per legacy logic."""
    df = schwab_client.get_fundamentals(["AAPL"])
    row = df.iloc[0]
    assert row["float_shares"] == 1500.0 * 1e6
    assert row["shares_outstanding"] == 16_000_000_000
    assert row["high_52wk"] == 199.62
    assert row["low_52wk"] == 124.17
    assert row["source"] == "schwab_fundamental"
    assert row["symbol"] == "AAPL"


def test_fundamentals_uses_fundamental_projection(patch_client):
    from schwab.client import Client

    schwab_client.get_fundamentals(["AAPL"])
    _, kwargs = patch_client.get_instruments.call_args
    args, _ = patch_client.get_instruments.call_args
    # projection may be passed positionally or by keyword
    proj = kwargs.get("projection")
    if proj is None and len(args) > 1:
        proj = args[1]
    assert proj == Client.Instrument.Projection.FUNDAMENTAL


def test_fundamentals_multiple_symbols(patch_client):
    patch_client.get_instruments.return_value = FakeResponse(
        {
            "AAPL": {
                "fundamental": {
                    "marketCapFloat": 1000.0,
                    "sharesOutstanding": 1,
                    "high52": 10,
                    "low52": 1,
                }
            },
            "MSFT": {
                "fundamental": {
                    "marketCapFloat": 2000.0,
                    "sharesOutstanding": 2,
                    "high52": 20,
                    "low52": 2,
                }
            },
        }
    )
    df = schwab_client.get_fundamentals(["AAPL", "MSFT"])
    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    assert df.set_index("symbol").loc["MSFT", "float_shares"] == 2000.0 * 1e6


# --------------------------------------------------------------------------- #
# get_quote_snapshot
# --------------------------------------------------------------------------- #
QUOTE_COLS = ["symbol", "last_price", "bid", "ask", "total_volume", "ts"]


def test_quote_snapshot_schema_exact(patch_client):
    df = schwab_client.get_quote_snapshot(["AAPL"])
    assert list(df.columns) == QUOTE_COLS


def test_quote_snapshot_values(patch_client):
    df = schwab_client.get_quote_snapshot(["AAPL"])
    row = df.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["last_price"] == 187.5
    assert row["bid"] == 187.4
    assert row["ask"] == 187.6
    assert row["total_volume"] == 45_000_000


def test_quote_snapshot_ts_is_utc(patch_client):
    df = schwab_client.get_quote_snapshot(["AAPL"])
    ts = df["ts"].iloc[0]
    assert ts.tzinfo is not None
    assert str(df["ts"].dt.tz) == "UTC"


# --------------------------------------------------------------------------- #
# get_live_quote
# --------------------------------------------------------------------------- #
def test_live_quote_maps_fields(patch_client):
    q = schwab_client.get_live_quote("AAPL")
    assert q == {
        "last_price": 187.5,
        "total_volume": 45_000_000,
        "high_52wk": 199.62,
        "low_52wk": 124.17,
    }


def test_live_quote_tolerates_missing_52wk(patch_client):
    patch_client.get_quotes.return_value = FakeResponse(
        {
            "AAPL": {
                "symbol": "AAPL",
                "quote": {
                    "lastPrice": 187.5,
                    "totalVolume": 45_000_000,
                    # 52WeekHigh present, 52WeekLow absent
                    "52WeekHigh": 199.62,
                },
            }
        }
    )
    q = schwab_client.get_live_quote("AAPL")
    assert q["last_price"] == 187.5
    assert q["total_volume"] == 45_000_000
    assert q["high_52wk"] == 199.62
    assert q["low_52wk"] is None


# --------------------------------------------------------------------------- #
# get_prior_close
# --------------------------------------------------------------------------- #
def test_prior_close_returns_last_daily_close(patch_client, monkeypatch):
    frame = pd.DataFrame(
        {
            "session_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "symbol": ["AAPL", "AAPL"],
            "close": [101.0, 205.5],
        }
    )
    monkeypatch.setattr(schwab_client, "get_daily_bars", lambda *a, **k: frame)
    assert schwab_client.get_prior_close("AAPL") == 205.5
    assert isinstance(schwab_client.get_prior_close("AAPL"), float)


# --------------------------------------------------------------------------- #
# get_movers
# --------------------------------------------------------------------------- #
MOVERS_COLS = [
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


def test_movers_schema_exact(patch_client):
    df = schwab_client.get_movers()
    assert list(df.columns) == MOVERS_COLS


def test_movers_values_and_source(patch_client):
    df = schwab_client.get_movers().set_index("symbol")
    top = df.loc["GAPR"]
    assert top["last_price"] == 4.20
    assert top["net_change"] == 1.40
    assert top["net_pct_change"] == 0.50
    assert top["volume"] == 8_000_000
    assert top["total_volume"] == 9_000_000
    assert top["trades"] == 1234
    assert top["market_share"] == 0.12
    assert top["source"] == "schwab_movers"


def test_movers_ts_is_utc(patch_client):
    df = schwab_client.get_movers()
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["ts"].iloc[0].tzinfo is not None


def test_movers_default_index_and_sort_resolved_to_enums(patch_client):
    """String defaults must be resolved to the Schwab enum members."""
    from schwab.client import Client

    schwab_client.get_movers()
    args, kwargs = patch_client.get_movers.call_args
    assert args[0] == Client.Movers.Index.EQUITY_ALL
    assert kwargs["sort_order"] == Client.Movers.SortOrder.PERCENT_CHANGE_UP
    # frequency omitted (None) -> not forwarded, Schwab uses its default
    assert "frequency" not in kwargs


def test_movers_accepts_enum_members_directly(patch_client):
    from schwab.client import Client

    schwab_client.get_movers(
        index=Client.Movers.Index.NASDAQ,
        sort_order=Client.Movers.SortOrder.VOLUME,
        frequency=Client.Movers.Frequency.FIVE,
    )
    args, kwargs = patch_client.get_movers.call_args
    assert args[0] == Client.Movers.Index.NASDAQ
    assert kwargs["sort_order"] == Client.Movers.SortOrder.VOLUME
    assert kwargs["frequency"] == Client.Movers.Frequency.FIVE


def test_movers_frequency_int_resolved_to_enum(patch_client):
    from schwab.client import Client

    schwab_client.get_movers(frequency=5)
    _, kwargs = patch_client.get_movers.call_args
    assert kwargs["frequency"] == Client.Movers.Frequency.FIVE


def test_movers_legacy_field_fallbacks(patch_client):
    """Older TDA-style keys (last / change / netPercentChangeInDouble) parse."""
    patch_client.get_movers.return_value = FakeResponse(
        {
            "screeners": [
                {
                    "symbol": "OLD",
                    "description": "Legacy Shape",
                    "last": 7.5,
                    "change": 2.5,
                    "netPercentChangeInDouble": 0.25,
                    "volume": 100,
                    "totalVolume": 200,
                }
            ]
        }
    )
    row = schwab_client.get_movers().iloc[0]
    assert row["last_price"] == 7.5
    assert row["net_change"] == 2.5
    assert row["net_pct_change"] == 0.25


def test_movers_empty_payload(patch_client):
    patch_client.get_movers.return_value = FakeResponse({})
    df = schwab_client.get_movers()
    assert list(df.columns) == MOVERS_COLS
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# 429 retry / backoff
# --------------------------------------------------------------------------- #
def test_429_retry_then_success(patch_client, monkeypatch):
    """A 429 followed by a 200 should be retried and ultimately succeed."""
    sleeps = []
    monkeypatch.setattr(schwab_client.time, "sleep", lambda s: sleeps.append(s))

    good = FakeResponse(_minute_payload())
    patch_client.get_price_history_every_minute.side_effect = [
        FakeResponse({}, status_code=429),
        good,
    ]
    df = schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert len(df) == 2
    assert patch_client.get_price_history_every_minute.call_count == 2
    assert sleeps == [1]  # first backoff is 1 second


def test_429_backoff_is_exponential(patch_client, monkeypatch):
    sleeps = []
    monkeypatch.setattr(schwab_client.time, "sleep", lambda s: sleeps.append(s))

    good = FakeResponse(_minute_payload())
    patch_client.get_price_history_every_minute.side_effect = [
        FakeResponse({}, status_code=429),
        FakeResponse({}, status_code=429),
        FakeResponse({}, status_code=429),
        good,
    ]
    schwab_client.get_minute_bars(
        "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
    )
    assert sleeps == [1, 2, 4]


def test_429_exhausts_retries_and_raises(patch_client, monkeypatch):
    monkeypatch.setattr(schwab_client.time, "sleep", lambda *_a: None)
    patch_client.get_price_history_every_minute.side_effect = [
        FakeResponse({}, status_code=429) for _ in range(10)
    ]
    with pytest.raises(Exception):
        schwab_client.get_minute_bars(
            "AAPL", dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 2)
        )


# --------------------------------------------------------------------------- #
# _get_client caching / auth wiring
# --------------------------------------------------------------------------- #
def test_get_client_reads_env_and_caches(monkeypatch):
    # The autouse ``patch_client`` fixture replaces ``_get_client`` with a
    # lambda; restore the real implementation so we exercise the auth wiring.
    monkeypatch.setattr(schwab_client, "_get_client", _REAL_GET_CLIENT)
    monkeypatch.setattr(schwab_client, "_CLIENT", None, raising=False)
    monkeypatch.setenv("SCHWAB_APP_KEY", "key123")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret456")
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "/tmp/token.json")

    calls = []

    def fake_from_token_file(token_path, api_key, app_secret, *a, **k):
        calls.append((token_path, api_key, app_secret))
        return MagicMock()

    monkeypatch.setattr(
        schwab_client.schwab.auth,
        "client_from_token_file",
        fake_from_token_file,
    )

    c1 = schwab_client._get_client()
    c2 = schwab_client._get_client()
    assert c1 is c2  # cached
    assert len(calls) == 1
    assert calls[0] == ("/tmp/token.json", "key123", "secret456")
