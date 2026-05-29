"""Tests for the Schwab trade-execution module.

These tests never hit the real Schwab API. The schwab-py client is replaced
with a ``MagicMock`` whose methods return canned ``Response``-like objects, and
the order builders / ``Utils`` are monkeypatched at module level so the key
behaviours don't depend on the real ``schwab`` package being importable.

Mirrors the mocking pattern in ``test_schwab_client.py``.
"""

from unittest.mock import MagicMock

import pytest

from ML_tradingAlgo.data import schwab_trader


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_client():
    """A MagicMock schwab-py client with sensible default return values."""
    client = MagicMock()
    client.get_account_numbers.return_value = FakeResponse(
        [
            {"accountNumber": "111111111", "hashValue": "HASH_ONE"},
            {"accountNumber": "222222222", "hashValue": "HASH_TWO"},
        ]
    )
    client.place_order.return_value = FakeResponse(status_code=201)
    return client


@pytest.fixture(autouse=True)
def patch_trader(monkeypatch, fake_client):
    """Force ``_get_client`` to return our mock, reset caches, no real network."""
    monkeypatch.setattr(schwab_trader, "_CLIENT", None, raising=False)
    monkeypatch.setattr(schwab_trader, "_get_client", lambda: fake_client)
    monkeypatch.setattr(schwab_trader, "_ACCOUNT_HASH", None, raising=False)
    monkeypatch.setattr(schwab_trader.time, "sleep", lambda *_a, **_k: None)
    # Builders return identifiable marker objects so we can assert on them.
    monkeypatch.setattr(
        schwab_trader,
        "equity_buy_market",
        lambda symbol, qty: {"side": "BUY", "symbol": symbol, "qty": qty},
    )
    monkeypatch.setattr(
        schwab_trader,
        "equity_sell_market",
        lambda symbol, qty: {"side": "SELL", "symbol": symbol, "qty": qty},
    )
    # Utils(client, hash).extract_order_id(r) -> a fixed id.
    utils_obj = MagicMock()
    utils_obj.extract_order_id.return_value = "ORDER123"
    monkeypatch.setattr(
        schwab_trader, "Utils", lambda *_a, **_k: utils_obj
    )
    return fake_client


@pytest.fixture
def captured_notifications(monkeypatch):
    """Capture every notify() call as (message, level) tuples."""
    calls = []
    monkeypatch.setattr(
        schwab_trader, "notify", lambda msg, level="info": calls.append((msg, level))
    )
    return calls


# --------------------------------------------------------------------------- #
# get_account_hash
# --------------------------------------------------------------------------- #
def test_account_hash_matches_brokerage_account_id(patch_trader, monkeypatch):
    monkeypatch.setenv("BROKERAGE_ACCOUNT_ID", "222222222")
    assert schwab_trader.get_account_hash() == "HASH_TWO"


def test_account_hash_falls_back_to_only_account_when_env_unset(
    patch_trader, monkeypatch
):
    monkeypatch.delenv("BROKERAGE_ACCOUNT_ID", raising=False)
    patch_trader.get_account_numbers.return_value = FakeResponse(
        [{"accountNumber": "999999999", "hashValue": "ONLY_HASH"}]
    )
    assert schwab_trader.get_account_hash() == "ONLY_HASH"


def test_account_hash_falls_back_on_no_match(patch_trader, monkeypatch):
    monkeypatch.setenv("BROKERAGE_ACCOUNT_ID", "does-not-exist")
    # Falls back to the first account.
    assert schwab_trader.get_account_hash() == "HASH_ONE"


def test_account_hash_is_cached(patch_trader, monkeypatch):
    monkeypatch.delenv("BROKERAGE_ACCOUNT_ID", raising=False)
    first = schwab_trader.get_account_hash()
    second = schwab_trader.get_account_hash()
    assert first == second
    assert patch_trader.get_account_numbers.call_count == 1


# --------------------------------------------------------------------------- #
# _trading_enabled
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "YES"])
def test_trading_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("TRADING_ENABLED", value)
    assert schwab_trader._trading_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "", "nope"])
def test_trading_enabled_falsy(monkeypatch, value):
    monkeypatch.setenv("TRADING_ENABLED", value)
    assert schwab_trader._trading_enabled() is False


def test_trading_enabled_default_false(monkeypatch):
    monkeypatch.delenv("TRADING_ENABLED", raising=False)
    assert schwab_trader._trading_enabled() is False


# --------------------------------------------------------------------------- #
# Dry-run (trading disabled)
# --------------------------------------------------------------------------- #
def test_buy_dry_run_returns_none_and_notifies(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.delenv("TRADING_ENABLED", raising=False)
    result = schwab_trader.buy_market("AAPL", 10)
    assert result is None
    assert not patch_trader.place_order.called
    assert not patch_trader.get_account_numbers.called
    assert any("[DRY-RUN]" in msg and "BUY" in msg for msg, _ in captured_notifications)


def test_sell_dry_run_returns_none_and_notifies(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.delenv("TRADING_ENABLED", raising=False)
    result = schwab_trader.sell_market("AAPL", 10)
    assert result is None
    assert not patch_trader.place_order.called
    assert not patch_trader.get_account_numbers.called
    assert any(
        "[DRY-RUN]" in msg and "SELL" in msg for msg, _ in captured_notifications
    )


# --------------------------------------------------------------------------- #
# Live (trading enabled)
# --------------------------------------------------------------------------- #
def test_buy_live_places_order_and_returns_id(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("BROKERAGE_ACCOUNT_ID", "222222222")
    # Legacy callers pass floats (e.g. buy_quantity / 2).
    order_id = schwab_trader.buy_market("AAPL", 10.0)

    assert order_id == "ORDER123"
    assert patch_trader.place_order.called
    account_hash, order = patch_trader.place_order.call_args.args
    assert account_hash == "HASH_TWO"
    assert order == {"side": "BUY", "symbol": "AAPL", "qty": 10}
    assert isinstance(order["qty"], int)


def test_sell_live_places_order_and_returns_id(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("BROKERAGE_ACCOUNT_ID", "111111111")
    order_id = schwab_trader.sell_market("MSFT", 4.0)

    assert order_id == "ORDER123"
    account_hash, order = patch_trader.place_order.call_args.args
    assert account_hash == "HASH_ONE"
    assert order == {"side": "SELL", "symbol": "MSFT", "qty": 4}
    assert isinstance(order["qty"], int)


def test_live_order_coerces_float_quantity_to_int(patch_trader, monkeypatch):
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.delenv("BROKERAGE_ACCOUNT_ID", raising=False)
    schwab_trader.buy_market("AAPL", 7.9)
    _, order = patch_trader.place_order.call_args.args
    assert order["qty"] == 7  # int() truncation


def test_live_order_non_ok_response_raises_and_notifies(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.delenv("BROKERAGE_ACCOUNT_ID", raising=False)
    patch_trader.place_order.return_value = FakeResponse(status_code=400)

    with pytest.raises(Exception):
        schwab_trader.buy_market("AAPL", 10)

    assert any(level == "error" for _, level in captured_notifications)


def test_live_order_success_notifies(
    patch_trader, monkeypatch, captured_notifications
):
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.delenv("BROKERAGE_ACCOUNT_ID", raising=False)
    schwab_trader.buy_market("AAPL", 10)
    assert any("ORDER123" in msg for msg, _ in captured_notifications)
