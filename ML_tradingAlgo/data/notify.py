"""Channel-agnostic alert notifier — the Twilio/WhatsApp replacement.

The project deprecated Twilio (see ``memory``/``docs/notifications.md``). This
module is the single place all alerts (trade signals, token-expiry warnings,
errors) flow through, so the delivery channel can change without touching any
call site.

Channels
--------
* **logging (default)** — with no configuration, every alert is logged. Zero
  external dependency; nothing breaks if alerting is unconfigured.
* **Discord webhook (opt-in)** — set ``ALERT_WEBHOOK_URL`` to a Discord webhook
  URL and alerts are additionally POSTed there as ``{"content": <message>}``.
  This is the planned production channel; see ``docs/notifications.md``.

Alerting must never crash the caller: a webhook failure degrades to a logged
warning. The public ``notify(message, level="info")`` signature is intentionally
callable with a single positional argument so it can be passed directly as the
``notify`` callback used by :mod:`ML_tradingAlgo.data.token_health`.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Levels accepted for the standard-library logger; anything else falls back to
# "info" so a bad caller can't suppress an alert.
_LEVELS = {"debug", "info", "warning", "error", "critical"}

_WEBHOOK_TIMEOUT = 10.0  # seconds — alerting is best-effort, never block long


def _webhook_url() -> str | None:
    """Configured webhook URL, or None to use the logging channel only."""
    return os.environ.get("ALERT_WEBHOOK_URL") or None


def _http_post(url, json=None, timeout=None):  # pragma: no cover - thin shim
    """Indirection point so tests can patch the network call."""
    import requests

    return requests.post(url, json=json, timeout=timeout)


def notify(message: str, level: str = "info") -> str:
    """Send an alert through the configured channel.

    Always logs the message (local record). If ``ALERT_WEBHOOK_URL`` is set,
    additionally POSTs a Discord-shaped ``{"content": message}`` payload.

    Returns the channel that ultimately handled the alert: ``"webhook"`` when the
    POST succeeded, otherwise ``"log"`` (including when a webhook attempt failed).
    """
    lvl = level if level in _LEVELS else "info"
    getattr(logger, lvl)(message)

    url = _webhook_url()
    if not url:
        return "log"

    try:
        resp = _http_post(url, json={"content": message}, timeout=_WEBHOOK_TIMEOUT)
        if resp is not None:
            resp.raise_for_status()
        return "webhook"
    except Exception as exc:  # never let alerting crash the caller
        logger.warning("alert webhook delivery failed (%s); message was: %s", exc, message)
        return "log"
