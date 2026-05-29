"""Schwab refresh-token freshness check.

Schwab's refresh token has a hard 7-day life and CANNOT be renewed
programmatically — re-running the interactive login (schwab_auth) is the only
way to get a new one. A long-running scanner therefore needs a heads-up before
the token dies mid-session.

schwab-py writes the token file as ``{"creation_timestamp": <epoch_int>,
"token": {...}}``; the refresh token expires 7 days after that timestamp. This
module reads it and reports status, calling a pluggable ``notify`` callback when
re-auth is due. It is channel-agnostic: the default notifier just logs, so no
external alerting dependency (e.g. Twilio) is required — wire whatever channel
you like by passing ``notify``.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

EXPIRY_DAYS = 7.0          # Schwab refresh-token hard limit
DEFAULT_WARN_AFTER = 6.0   # warn on day 6 so there's time to re-auth


def _token_path(token_path: str | None) -> str | None:
    return token_path or os.environ.get("SCHWAB_TOKEN_PATH")


def token_age_days(token_path: str | None = None, now: float | None = None) -> float | None:
    """Age of the stored refresh token in days, or None if no token file."""
    path = _token_path(token_path)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        created = data.get("creation_timestamp")
        if created is None:
            return None
        now = time.time() if now is None else now
        return max(0.0, (now - float(created)) / 86400.0)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def check_token_freshness(
    token_path: str | None = None,
    warn_after_days: float = DEFAULT_WARN_AFTER,
    expiry_days: float = EXPIRY_DAYS,
    notify=None,
    now: float | None = None,
) -> dict:
    """Report refresh-token status and notify when re-auth is due.

    Returns a dict: ``{status, age_days, days_remaining, message}`` where status
    is one of ``missing``, ``ok``, ``warn``, ``expired``. ``notify(message)`` is
    called for ``warn`` and ``expired`` (defaults to ``logger.warning``).
    """
    notify = notify or logger.warning
    age = token_age_days(token_path, now=now)

    if age is None:
        msg = "No Schwab token found — run: python3 -m ML_tradingAlgo.data.schwab_auth"
        result = {"status": "missing", "age_days": None, "days_remaining": None, "message": msg}
        notify(msg)
        return result

    remaining = expiry_days - age
    if age >= expiry_days:
        status, msg = "expired", (
            f"Schwab refresh token EXPIRED ({age:.1f}d old). Re-auth now: "
            "python3 -m ML_tradingAlgo.data.schwab_auth"
        )
        notify(msg)
    elif age >= warn_after_days:
        status, msg = "warn", (
            f"Schwab refresh token expires in {remaining:.1f}d ({age:.1f}d old). "
            "Re-auth soon: python3 -m ML_tradingAlgo.data.schwab_auth"
        )
        notify(msg)
    else:
        status, msg = "ok", f"Schwab token healthy ({age:.1f}d old, {remaining:.1f}d left)."

    return {"status": status, "age_days": age, "days_remaining": remaining, "message": msg}


if __name__ == "__main__":  # pragma: no cover
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = check_token_freshness()
    print(res["message"])
