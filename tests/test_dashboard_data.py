"""Tests for the dashboard data layer (``ML_tradingAlgo.dashboard._data``).

Moto server-mode fixture (copied from ``tests/test_store.py``) -- s3fs/pyarrow
open their own aiobotocore sessions that ``@mock_aws`` does not reliably patch,
so we run a real ``ThreadedMotoServer`` and point boto3 + s3fs at it via
``AWS_ENDPOINT_URL``.

These tests exercise ONLY ``_data.py`` (the Streamlit UI is verified manually).
To avoid the ``@st.cache_data`` wrappers caching across tests / requiring a
Streamlit runtime, the tests call the **underscore-prefixed** core functions
(``_load_events``, ``_coverage_summary`` ...) directly.
"""

from __future__ import annotations

import datetime as dt
import json

import boto3
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.dashboard import _data
from ML_tradingAlgo.data import store

BUCKET = "test-dash-bucket"
PREFIX = "warehouse"


@pytest.fixture(scope="module")
def moto_server():
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    yield endpoint
    server.stop()


@pytest.fixture
def s3_env(moto_server, monkeypatch):
    endpoint = moto_server
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("S3_PREFIX", PREFIX)

    s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
    try:
        s3.create_bucket(Bucket=BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Clear leftovers from a previous test in the module.
    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=obj["Key"])

    import s3fs

    s3fs.S3FileSystem.clear_instance_cache()
    yield endpoint


# --------------------------------------------------------------------------- #
# seed helpers
# --------------------------------------------------------------------------- #
def _seed_events():
    """Two sessions, three events with distinct symbols.

    The store dedupes ``read_bars`` output on ``(symbol, ts, source)`` and the
    events table has no ``ts`` column, so two rows with the same symbol+source
    would be collapsed regardless of session. Production nightly runs use one
    asof-session per run; here we use distinct symbols so all three rows
    survive (mirroring real distinct gap-up names) and the fundamentals math is
    fully exercised (one all-None row, one partial-None row).
    """
    rows = [
        {
            "symbol": "AAA",
            "session_date": dt.date(2026, 1, 2),
            "prior_close": 1.0,
            "open_price": 1.3,
            "gap_pct": 0.30,
            "rvol_at_open": 5.0,
            "float_shares": 10_000_000,
            "short_interest_ratio": 2.5,
            "sector_id": 7,
            "earnings_date": "2026-02-01",
            "passed_filters": True,
            "filter_reasons": "",
            "detected_at": pd.Timestamp("2026-01-02 21:00", tz="UTC"),
        },
        {
            "symbol": "BBB",
            "session_date": dt.date(2026, 1, 2),
            "prior_close": 2.0,
            "open_price": 2.7,
            "gap_pct": 0.35,
            "rvol_at_open": 4.0,
            "float_shares": 20_000_000,
            # all fundamentals missing -> dead-feature risk
            "short_interest_ratio": None,
            "sector_id": None,
            "earnings_date": None,
            "passed_filters": False,
            "filter_reasons": "premarket_unavailable",
            "detected_at": pd.Timestamp("2026-01-02 21:00", tz="UTC"),
        },
        {
            "symbol": "CCC",
            "session_date": dt.date(2026, 1, 3),
            "prior_close": 1.3,
            "open_price": 1.7,
            "gap_pct": 0.31,
            "rvol_at_open": 6.0,
            "float_shares": 10_000_000,
            "short_interest_ratio": 3.1,
            "sector_id": None,  # sector missing for this one
            "earnings_date": "2026-02-01",
            "passed_filters": True,
            "filter_reasons": "",
            "detected_at": pd.Timestamp("2026-01-03 21:00", tz="UTC"),
        },
    ]
    df = pd.DataFrame(rows)
    store.write_bars(df, table="events", partition_cols=["session_date"], source="collector")


def _minute_df(symbol, session_date, n=4, source="schwab_pricehistory"):
    base = pd.Timestamp(f"{session_date} 14:30", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "session_date": [session_date] * n,
            "open": [1.0 + i for i in range(n)],
            "high": [1.5 + i for i in range(n)],
            "low": [0.8 + i for i in range(n)],
            "close": [1.2 + i for i in range(n)],
            "volume": [1000 * (i + 1) for i in range(n)],
            "source": [source] * n,
        }
    )


def _write_live_state(endpoint, instance_id, obj):
    import s3fs

    fs = s3fs.S3FileSystem(
        skip_instance_cache=True, client_kwargs={"endpoint_url": endpoint}
    )
    path = f"{BUCKET}/{PREFIX}/live/state/{instance_id}.json"
    with fs.open(path, "wb") as f:
        f.write(json.dumps(obj, default=str).encode("utf-8"))


# --------------------------------------------------------------------------- #
# load_events
# --------------------------------------------------------------------------- #
def test_load_events_returns_rows(s3_env):
    _seed_events()
    df = _data._load_events()
    assert len(df) == 3
    assert set(df["symbol"]) == {"AAA", "BBB", "CCC"}


def test_load_events_date_range(s3_env):
    _seed_events()
    df = _data._load_events(date_range=(dt.date(2026, 1, 2), dt.date(2026, 1, 2)))
    assert len(df) == 2
    assert all(_data.store._to_date(d) == dt.date(2026, 1, 2) for d in df["session_date"])


def test_load_events_empty(s3_env):
    df = _data._load_events()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# coverage_summary
# --------------------------------------------------------------------------- #
def test_coverage_summary_counts(s3_env):
    _seed_events()
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2)),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2), n=2),
        table="bars_daily",
        partition_cols=["symbol", "session_date"],
        source="schwab",
    )

    cov = _data._coverage_summary()
    assert cov["n_events"] == 3
    assert cov["n_symbols"] == 3
    assert cov["events_per_day"]["2026-01-02"] == 2
    assert cov["events_per_day"]["2026-01-03"] == 1
    assert cov["date_min"] == "2026-01-02"
    assert cov["date_max"] == "2026-01-03"
    assert cov["bars_per_table"]["bars_minute"] == 4
    assert cov["bars_per_table"]["bars_daily"] == 2
    assert cov["bars_per_table"]["bars_5min"] == 0


def test_coverage_summary_empty(s3_env):
    cov = _data._coverage_summary()
    assert cov["n_events"] == 0
    assert cov["n_symbols"] == 0
    assert cov["events_per_day"] == {}
    assert cov["date_min"] is None
    assert all(v == 0 for v in cov["bars_per_table"].values())


# --------------------------------------------------------------------------- #
# fundamentals_completeness (incl. None fields)
# --------------------------------------------------------------------------- #
def test_fundamentals_completeness_math(s3_env):
    _seed_events()
    events = _data._load_events()
    comp = _data._fundamentals_completeness(events)

    assert comp["n_events"] == 3
    # short_interest: AAA(2.5), CCC(3.1) present; BBB None -> 2 present / 1 missing
    assert comp["fields"]["short_interest"]["present"] == 2
    assert comp["fields"]["short_interest"]["missing"] == 1
    # sector: only AAA has it (BBB None, CCC None) -> 1 present / 2 missing
    assert comp["fields"]["sector"]["present"] == 1
    assert comp["fields"]["sector"]["missing"] == 2
    # earnings: AAA + CCC present; BBB None -> 2 present / 1 missing
    assert comp["fields"]["earnings"]["present"] == 2
    assert comp["fields"]["earnings"]["missing"] == 1
    assert comp["fields"]["short_interest"]["pct_present"] == pytest.approx(66.67, abs=0.01)


def test_fundamentals_completeness_empty():
    comp = _data._fundamentals_completeness(pd.DataFrame())
    assert comp["n_events"] == 0
    for f in ("short_interest", "sector", "earnings"):
        assert comp["fields"][f]["present"] == 0
        assert comp["fields"][f]["missing"] == 0


def test_fundamentals_completeness_missing_columns():
    # No fundamentals columns at all -> everything counts as missing.
    df = pd.DataFrame({"symbol": ["X", "Y"], "session_date": [dt.date(2026, 1, 2)] * 2})
    comp = _data._fundamentals_completeness(df)
    assert comp["n_events"] == 2
    for f in ("short_interest", "sector", "earnings"):
        assert comp["fields"][f]["present"] == 0
        assert comp["fields"][f]["missing"] == 2


# --------------------------------------------------------------------------- #
# load_minute_bars
# --------------------------------------------------------------------------- #
def test_load_minute_bars_returns_rows(s3_env):
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2)),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    store.write_bars(
        _minute_df("BBB", dt.date(2026, 1, 2)),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    out = _data._load_minute_bars("AAA", dt.date(2026, 1, 2))
    assert len(out) == 4
    assert set(out["symbol"]) == {"AAA"}
    # sorted by ts ascending
    assert out["ts"].is_monotonic_increasing


def test_load_minute_bars_source_filter(s3_env):
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2), source="schwab_pricehistory"),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2), n=2, source="live_tick_agg"),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    live = _data._load_minute_bars("AAA", dt.date(2026, 1, 2), source="live_tick_agg")
    assert set(live["source"]) == {"live_tick_agg"}
    assert len(live) == 2


def test_load_minute_bars_empty(s3_env):
    out = _data._load_minute_bars("ZZZ", dt.date(2026, 1, 2))
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# load_live_state
# --------------------------------------------------------------------------- #
def test_load_live_state_parses_json(s3_env):
    state1 = {
        "instance_id": "1",
        "watched_symbols": ["AAA", "BBB"],
        "positions": [{"symbol": "AAA", "qty": 100, "entry": 1.3}],
        "last_signal": {"symbol": "AAA", "action": "BUY"},
        "last_price": {"AAA": 1.45},
        "pnl": 12.5,
        "updated_at": "2026-01-02T21:00:00+00:00",
    }
    state2 = {
        "watched_symbols": ["CCC"],
        "pnl": -3.0,
        "updated_at": "2026-01-02T21:01:00+00:00",
    }
    _write_live_state(s3_env, "1", state1)
    _write_live_state(s3_env, "2", state2)

    states = _data._load_live_state()
    assert len(states) == 2
    by_id = {s["instance_id"]: s for s in states}
    assert by_id["1"]["pnl"] == 12.5
    assert by_id["1"]["watched_symbols"] == ["AAA", "BBB"]
    # instance_id derived from filename when absent from the JSON body
    assert by_id["2"]["watched_symbols"] == ["CCC"]


def test_load_live_state_empty(s3_env):
    assert _data._load_live_state() == []


def test_load_live_state_skips_malformed(s3_env):
    import s3fs

    fs = s3fs.S3FileSystem(
        skip_instance_cache=True, client_kwargs={"endpoint_url": s3_env}
    )
    with fs.open(f"{BUCKET}/{PREFIX}/live/state/bad.json", "wb") as f:
        f.write(b"{not valid json")
    _write_live_state(s3_env, "1", {"instance_id": "1", "pnl": 1.0})

    states = _data._load_live_state()
    assert len(states) == 1
    assert states[0]["instance_id"] == "1"


# --------------------------------------------------------------------------- #
# list_live_symbols
# --------------------------------------------------------------------------- #
def test_list_live_symbols_filters_by_source(s3_env):
    store.write_bars(
        _minute_df("AAA", dt.date(2026, 1, 2), source="live_tick_agg"),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    store.write_bars(
        _minute_df("BBB", dt.date(2026, 1, 2), source="live_tick_agg"),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )
    # CCC only has authoritative bars -> must NOT appear
    store.write_bars(
        _minute_df("CCC", dt.date(2026, 1, 2), source="schwab_pricehistory"),
        table="bars_minute",
        partition_cols=["symbol", "session_date"],
    )

    syms = _data._list_live_symbols(dt.date(2026, 1, 2))
    assert syms == ["AAA", "BBB"]


def test_list_live_symbols_empty(s3_env):
    assert _data._list_live_symbols(dt.date(2026, 1, 2)) == []
