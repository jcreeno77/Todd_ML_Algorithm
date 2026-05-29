"""Trade execution over the ``schwab-py`` library.

This is the Schwab replacement for the dead TD Ameritrade order path. It exposes
a tiny, stable surface for the live trading loop::

    get_account_hash()            -> str   (encrypted account id for order calls)
    buy_market(symbol, quantity)  -> str | None  (order id, or None in dry-run)
    sell_market(symbol, quantity) -> str | None

Safety / behaviour
------------------
* **Dry-run by default.** Orders are only sent when ``TRADING_ENABLED`` is truthy
  (``true``/``1``/``yes``, case-insensitive). Otherwise a ``[DRY-RUN]`` alert is
  emitted and no network call is made.
* Every network call is wrapped in :func:`schwab_client._request_with_retry` so
  HTTP 429s are retried with exponential backoff.
* All order / dry-run / error events route through :func:`notify` so the alert
  channel stays pluggable.
* The encrypted account hash is resolved once and cached at module level. Raw
  account numbers and full hashes are never logged (masked to the last 4 chars).

The schwab-py order builders and :class:`schwab.utils.Utils` are imported lazily
(inside the functions) and exposed as module-level names so that the package is
not a hard import dependency and tests can monkeypatch them.
"""

from __future__ import annotations

import os
import time

from .notify import notify
from .schwab_client import _get_client, _request_with_retry

__all__ = ["get_account_hash", "buy_market", "sell_market"]

# Module-level names for the lazily-imported schwab symbols. They start as None
# and are populated on first use; tests monkeypatch them directly.
equity_buy_market = None
equity_sell_market = None
Utils = None

# Cached encrypted account hash (resolved once per process). Tests reset to None.
_ACCOUNT_HASH = None

# Mirror schwab_client's cached-client global name so the test fixtures that
# reset ``_CLIENT`` have something to target; the real client lives in
# schwab_client, this is only a convenience alias kept None here.
_CLIENT = None


def _load_schwab_symbols() -> None:
    """Lazily import the schwab order builders / Utils into module globals."""
    global equity_buy_market, equity_sell_market, Utils
    if equity_buy_market is None:
        from schwab.orders.equities import equity_buy_market as _buy
        from schwab.orders.equities import equity_sell_market as _sell

        equity_buy_market = _buy
        equity_sell_market = _sell
    if Utils is None:
        from schwab.utils import Utils as _Utils

        Utils = _Utils


def _mask(value: str) -> str:
    """Mask a sensitive identifier, revealing only the last 4 characters."""
    text = str(value)
    return ("*" * max(len(text) - 4, 0)) + text[-4:]


def _trading_enabled() -> bool:
    """True only when ``TRADING_ENABLED`` is a recognised truthy value."""
    return os.environ.get("TRADING_ENABLED", "false").strip().lower() in {
        "true",
        "1",
        "yes",
    }


def get_account_hash() -> str:
    """Resolve (and cache) the encrypted account hash for order placement.

    Picks the account whose ``accountNumber`` matches ``BROKERAGE_ACCOUNT_ID``;
    falls back to the first/sole account when the env var is unset or no entry
    matches. The result is cached in :data:`_ACCOUNT_HASH`.
    """
    global _ACCOUNT_HASH
    if _ACCOUNT_HASH is not None:
        return _ACCOUNT_HASH

    client = _get_client()
    response = _request_with_retry(client.get_account_numbers)
    accounts = response.json() or []
    if not accounts:
        raise RuntimeError("Schwab returned no linked accounts")

    wanted = os.environ.get("BROKERAGE_ACCOUNT_ID")
    chosen = None
    if wanted:
        for entry in accounts:
            if entry.get("accountNumber") == wanted:
                chosen = entry
                break
    if chosen is None:
        chosen = accounts[0]

    _ACCOUNT_HASH = chosen["hashValue"]
    return _ACCOUNT_HASH


def _place_market(side: str, symbol: str, quantity) -> str | None:
    """Shared buy/sell market-order logic. ``side`` is ``"BUY"`` or ``"SELL"``."""
    qty = int(quantity)

    if not _trading_enabled():
        notify(f"[DRY-RUN] would {side} {qty} {symbol}")
        return None

    _load_schwab_symbols()
    client = _get_client()
    account_hash = get_account_hash()

    builder = equity_buy_market if side == "BUY" else equity_sell_market
    order = builder(symbol, qty)

    response = _request_with_retry(client.place_order, account_hash, order)
    if getattr(response, "status_code", None) not in (200, 201):
        notify(
            f"{side} order for {qty} {symbol} failed "
            f"(HTTP {getattr(response, 'status_code', '?')})",
            level="error",
        )
        response.raise_for_status()
        # raise_for_status may not raise on every odd status; guarantee a failure.
        raise RuntimeError(
            f"Schwab {side} order rejected (HTTP {getattr(response, 'status_code', '?')})"
        )

    order_id = Utils(client, account_hash).extract_order_id(response)
    notify(f"{side} {qty} {symbol} placed (order {order_id})")
    return order_id


def buy_market(symbol: str, quantity) -> str | None:
    """Place a market BUY for ``quantity`` shares of ``symbol``.

    Returns the order id, or ``None`` when trading is disabled (dry-run).
    """
    return _place_market("BUY", symbol, quantity)


def sell_market(symbol: str, quantity) -> str | None:
    """Place a market SELL for ``quantity`` shares of ``symbol``.

    Returns the order id, or ``None`` when trading is disabled (dry-run).
    """
    return _place_market("SELL", symbol, quantity)
