"""Tests for the universe tracking layer (``ML_tradingAlgo.data.gainers``).

Uses moto server mode (matching ``tests/test_store.py``) so s3fs/pyarrow talk to
a real HTTP mock S3 endpoint rather than relying on botocore decorator patching.
"""

from __future__ import annotations

import datetime as dt

import boto3
import pytest
from moto.server import ThreadedMotoServer

from ML_tradingAlgo.data import gainers

BUCKET = "test-universe-bucket"
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

    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=obj["Key"])

    import s3fs

    s3fs.S3FileSystem.clear_instance_cache()
    yield endpoint


# --------------------------------------------------------------------------- #
# record -> load round-trip
# --------------------------------------------------------------------------- #
def test_record_load_roundtrip(s3_env):
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 2)})
    out = gainers.load_universe(dt.date(2026, 1, 1))
    assert out == ["AAA"]


def test_record_defaults_observed_by_live_scanner(s3_env):
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 2)})
    from ML_tradingAlgo.data import store

    df = store.read_bars("universe")
    assert set(df["observed_by"]) == {"live_scanner"}


def test_record_uses_ctx_observed_by(s3_env):
    gainers.record_observation(
        "AAA", {"session_date": dt.date(2026, 1, 2), "observed_by": "backfill"}
    )
    from ML_tradingAlgo.data import store

    df = store.read_bars("universe")
    assert set(df["observed_by"]) == {"backfill"}


def test_record_carries_gap_and_volume(s3_env):
    gainers.record_observation(
        "AAA",
        {
            "session_date": dt.date(2026, 1, 2),
            "premarket_gap_pct": 32.5,
            "premarket_volume": 1_000_000,
        },
    )
    from ML_tradingAlgo.data import store

    df = store.read_bars("universe")
    assert df.iloc[0]["premarket_gap_pct"] == 32.5
    assert df.iloc[0]["premarket_volume"] == 1_000_000


def test_record_none_ctx_does_not_raise(s3_env):
    gainers.record_observation("AAA")
    out = gainers.load_universe(dt.date(2000, 1, 1))
    assert out == ["AAA"]


# --------------------------------------------------------------------------- #
# dedupe to unique symbols
# --------------------------------------------------------------------------- #
def test_multiple_observations_dedupe_to_unique_symbols(s3_env):
    for _ in range(3):
        gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 2)})
    gainers.record_observation("BBB", {"session_date": dt.date(2026, 1, 2)})
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 3)})

    out = gainers.load_universe(dt.date(2026, 1, 1))
    assert out == ["AAA", "BBB"]


def test_load_returns_sorted(s3_env):
    gainers.record_observation("ZZZ", {"session_date": dt.date(2026, 1, 2)})
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 2)})
    gainers.record_observation("MMM", {"session_date": dt.date(2026, 1, 2)})

    out = gainers.load_universe(dt.date(2026, 1, 1))
    assert out == ["AAA", "MMM", "ZZZ"]


# --------------------------------------------------------------------------- #
# since_date filtering
# --------------------------------------------------------------------------- #
def test_since_date_excludes_older_sessions(s3_env):
    gainers.record_observation("OLD", {"session_date": dt.date(2026, 1, 1)})
    gainers.record_observation("NEW", {"session_date": dt.date(2026, 1, 10)})

    out = gainers.load_universe(dt.date(2026, 1, 5))
    assert out == ["NEW"]


def test_load_empty_returns_empty_list(s3_env):
    out = gainers.load_universe(dt.date(2026, 1, 1))
    assert out == []


# --------------------------------------------------------------------------- #
# seed_from_manual
# --------------------------------------------------------------------------- #
def test_seed_from_manual_writes_manual_rows(s3_env):
    gainers.seed_from_manual(["AAA", "BBB"], session_date=dt.date(2026, 1, 2))

    out = gainers.load_universe(dt.date(2026, 1, 1))
    assert out == ["AAA", "BBB"]

    from ML_tradingAlgo.data import store

    df = store.read_bars("universe")
    assert set(df["observed_by"]) == {"manual"}


def test_seed_from_manual_default_session_date_today(s3_env):
    gainers.seed_from_manual(["CCC"])
    # today's ET date should be >= a far-past since_date
    out = gainers.load_universe(dt.date(2000, 1, 1))
    assert "CCC" in out


# --------------------------------------------------------------------------- #
# since_date type normalization (date / datetime / str)
# --------------------------------------------------------------------------- #
def test_since_date_accepts_date(s3_env):
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 5)})
    assert gainers.load_universe(dt.date(2026, 1, 1)) == ["AAA"]


def test_since_date_accepts_datetime(s3_env):
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 5)})
    assert gainers.load_universe(dt.datetime(2026, 1, 1, 9, 30)) == ["AAA"]


def test_since_date_accepts_iso_string(s3_env):
    gainers.record_observation("AAA", {"session_date": dt.date(2026, 1, 5)})
    assert gainers.load_universe("2026-01-01") == ["AAA"]


def test_since_date_string_filters_correctly(s3_env):
    gainers.record_observation("OLD", {"session_date": dt.date(2026, 1, 1)})
    gainers.record_observation("NEW", {"session_date": dt.date(2026, 1, 10)})
    assert gainers.load_universe("2026-01-05") == ["NEW"]
