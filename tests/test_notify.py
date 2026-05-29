"""Tests for the channel-agnostic alert notifier (Twilio replacement)."""
import logging

import pytest

from ML_tradingAlgo.data import notify as notify_mod
from ML_tradingAlgo.data.notify import notify


def test_notify_logs_by_default(caplog, monkeypatch):
    """With no webhook configured, notify logs and reports the 'log' channel."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    with caplog.at_level(logging.INFO):
        channel = notify("Todd just bought AAPL")
    assert channel == "log"
    assert "Todd just bought AAPL" in caplog.text


def test_notify_posts_to_webhook_when_configured(monkeypatch):
    """When ALERT_WEBHOOK_URL is set, notify POSTs a Discord-shaped payload."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.test/webhook/abc")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _OkResponse()

    monkeypatch.setattr(notify_mod, "_http_post", fake_post)
    channel = notify("signal fired")
    assert channel == "webhook"
    assert captured["url"] == "https://discord.test/webhook/abc"
    assert captured["json"] == {"content": "signal fired"}
    assert captured["timeout"] is not None


def test_notify_webhook_failure_degrades_to_log(caplog, monkeypatch):
    """A webhook error must never propagate — it degrades to a logged warning."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.test/webhook/abc")

    def boom(url, json=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(notify_mod, "_http_post", boom)
    with caplog.at_level(logging.WARNING):
        channel = notify("important alert")
    assert channel == "log"
    assert "important alert" in caplog.text
    assert "webhook" in caplog.text.lower()


def test_notify_single_arg_matches_token_health_callback(monkeypatch):
    """token_health calls notify(message) with one positional arg — must work."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    # Should not raise; this is the exact call shape token_health uses.
    assert notify("Schwab token expires in 1.0d") == "log"


def test_invalid_level_falls_back_to_info(caplog, monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    with caplog.at_level(logging.INFO):
        channel = notify("msg", level="not-a-level")
    assert channel == "log"
    assert "msg" in caplog.text


class _OkResponse:
    status_code = 200

    def raise_for_status(self):
        return None
