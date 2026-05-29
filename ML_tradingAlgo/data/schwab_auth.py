"""One-time Schwab OAuth bootstrap — generates the token file the pipeline uses.

The Schwab login flow is INTERACTIVE: it opens (or asks you to visit) a browser,
you log in + approve, and Schwab redirects to the callback URL. This script
captures that and writes the token to SCHWAB_TOKEN_PATH. Run it ONCE; after that
schwab_client.get_minute_bars/etc. reuse the token (schwab-py auto-refreshes it).

Run it yourself (it needs your interactive login) via the `!` prefix in Claude
Code or directly in a terminal:

    cd ML_tradingAlgo && python3 -m ML_tradingAlgo.data.schwab_auth
    # headless / remote (paste the redirect URL by hand):
    python3 -m ML_tradingAlgo.data.schwab_auth --manual

Requires SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL, SCHWAB_TOKEN_PATH
in ML_tradingAlgo/.env. The callback URL must match the one registered on your
Schwab Developer Portal app exactly (default here: https://127.0.0.1:8182).
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

from schwab import auth


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    manual = "--manual" in argv

    api_key = os.environ.get("SCHWAB_APP_KEY")
    app_secret = os.environ.get("SCHWAB_APP_SECRET")
    callback_url = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
    token_path = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json")

    missing = [k for k, v in {
        "SCHWAB_APP_KEY": api_key,
        "SCHWAB_APP_SECRET": app_secret,
    }.items() if not v]
    if missing:
        print(f"Missing env vars: {', '.join(missing)} (set them in ML_tradingAlgo/.env)")
        return 1

    print(f"Callback URL : {callback_url}")
    print(f"Token path   : {token_path}")
    print(f"Flow         : {'manual (paste URL)' if manual else 'browser login'}\n")

    try:
        if manual:
            # Prints an auth URL; you log in, then paste the full redirect URL back.
            auth.client_from_manual_flow(api_key, app_secret, callback_url, token_path)
        else:
            # Opens a browser and runs a local loopback server to catch the redirect.
            auth.client_from_login_flow(api_key, app_secret, callback_url, token_path)
    except Exception as exc:
        print(f"\nAuth failed: {exc!r}")
        print("Tips: ensure the callback URL matches your Schwab app exactly; try "
              "--manual if no browser/loopback is available.")
        return 1

    if os.path.exists(token_path):
        print(f"\nToken written to {token_path}. You can now run the pre-flight:")
        print("    python3 -m ML_tradingAlgo.data._preflight")
        return 0
    print("\nFlow completed but no token file found — check the path/permissions.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
