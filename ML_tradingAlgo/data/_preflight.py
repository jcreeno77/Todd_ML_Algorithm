"""One-shot live pre-flight for the TFT data pipeline.

Validates the REAL external integrations that unit tests (which mock Schwab and
S3) cannot cover. Run this once you have:
  - Rotated Schwab creds + a completed OAuth token at SCHWAB_TOKEN_PATH
  - An S3 bucket with R/W IAM access, and S3_BUCKET set
all present in ML_tradingAlgo/.env.

Usage:
    cd ML_tradingAlgo && python3 -m ML_tradingAlgo.data._preflight
    # or from the worktree root:
    python3 -m ML_tradingAlgo.data._preflight

Each check is independent and best-effort; a failure is reported, not fatal,
so one run surfaces every problem at once. Nothing here is destructive beyond a
single tiny object it writes and then deletes from the `_preflight` table.
"""
from __future__ import annotations

import os
import sys
import datetime as dt

# Load .env so the pipeline's os.environ reads pick up local secrets. The store
# and clients read env directly and do NOT call load_dotenv themselves.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv always installed
    pass

import pandas as pd

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_env() -> bool:
    """Confirm the required env vars are present before hitting any service."""
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"]
    schwab = ["SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "SCHWAB_TOKEN_PATH"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        _fail(f"missing required env vars: {', '.join(missing)}")
        return False
    _ok("required S3 env vars present")
    miss_schwab = [k for k in schwab if not os.environ.get(k)]
    if miss_schwab:
        _warn(f"Schwab env vars unset (Schwab checks will be skipped): {', '.join(miss_schwab)}")
    return True


def check_s3_roundtrip() -> bool:
    """write_bars -> read_bars -> delete against the real bucket."""
    try:
        from ML_tradingAlgo.data import store
        df = pd.DataFrame({
            "symbol": ["PREFLIGHT"],
            "ts": [pd.Timestamp.now(tz="UTC")],
            "session_date": [dt.date.today()],
            "value": [1.0],
        })
        store.write_bars(df, table="_preflight", partition_cols=["symbol"], source="preflight")
        out = store.read_bars("_preflight", symbol="PREFLIGHT")
        if len(out) >= 1:
            _ok(f"S3 round-trip OK (bucket={os.environ.get('S3_BUCKET')}, read {len(out)} row)")
        else:
            _fail("S3 wrote but read returned no rows")
            return False
        # Best-effort cleanup of the _preflight prefix.
        try:
            fs = store._fs()
            base = store._base_path("_preflight")
            if fs.exists(base):
                fs.rm(base, recursive=True)
                _ok("cleaned up _preflight test data")
        except Exception as exc:
            _warn(f"could not clean up _preflight data: {exc}")
        return True
    except Exception as exc:
        _fail(f"S3 round-trip failed: {exc!r}")
        return False


def check_schwab_bars() -> bool:
    """Pull recent minute bars (extended hours) and report premarket coverage."""
    if not os.environ.get("SCHWAB_APP_KEY"):
        _warn("skipping Schwab bars check (no creds)")
        return True
    try:
        from ML_tradingAlgo.data import schwab_client
        end = dt.datetime.now()
        start = end - dt.timedelta(days=5)
        df = schwab_client.get_minute_bars("AAPL", start, end, extended_hours=True)
        if df is None or len(df) == 0:
            _fail("Schwab returned no minute bars for AAPL")
            return False
        _ok(f"Schwab minute bars OK ({len(df)} bars for AAPL)")
        # Premarket = bars before 09:30 ET. Critical for premarket_* features.
        try:
            et = df["ts"].dt.tz_convert("America/New_York")
            premarket = df[(et.dt.hour < 9) | ((et.dt.hour == 9) & (et.dt.minute < 30))]
            if len(premarket) > 0:
                _ok(f"premarket bars ARE returned ({len(premarket)} pre-09:30 ET) — premarket features viable")
            else:
                _warn("NO premarket bars returned — premarket_high/low/volume + rvol_at_open will degrade")
        except Exception as exc:
            _warn(f"could not assess premarket coverage: {exc}")
        return True
    except Exception as exc:
        _fail(f"Schwab bars check failed: {exc!r}")
        return False


def check_fundamentals() -> bool:
    """Confirm the chosen provider returns short interest / sector / earnings."""
    try:
        from ML_tradingAlgo.data import fundamentals
        provider = os.environ.get("FUNDAMENTALS_PROVIDER", "yahoo")
        data = fundamentals.get_fundamentals("AAPL", dt.date.today())
        missing = data.get("_missing", [])
        got = {k: data.get(k) for k in ("short_interest_ratio", "sector_id", "earnings_date")}
        if not missing:
            _ok(f"fundamentals ({provider}) returned all fields: {got}")
        else:
            _warn(f"fundamentals ({provider}) missing {missing}; got {got}")
        return True
    except Exception as exc:
        _fail(f"fundamentals check failed: {exc!r}")
        return False


def check_movers() -> bool:
    """Confirm Schwab /movers is index-bound (locks the universe design)."""
    if not os.environ.get("SCHWAB_APP_KEY"):
        _warn("skipping Schwab movers check (no creds)")
        return True
    try:
        from ML_tradingAlgo.data import schwab_client
        client = schwab_client._get_client()
        # schwab-py exposes get_movers(index, ...); indices are $DJI/$COMPX/$SPX only.
        resp = client.get_movers("$SPX")
        n = len(resp.json().get("screeners", resp.json())) if hasattr(resp, "json") else "?"
        _warn(f"/movers reachable but index-bound ($SPX returned ~{n}); universe stays live-scanner-driven")
        return True
    except Exception as exc:
        _warn(f"/movers check inconclusive ({exc!r}) — design already assumes it's unusable")
        return True


def main() -> int:
    print("=== TFT data-pipeline live pre-flight ===\n")
    if not check_env():
        print("\nAborting: fix env vars first.")
        return 1
    results = [
        check_s3_roundtrip(),
        check_schwab_bars(),
        check_fundamentals(),
        check_movers(),
    ]
    print()
    if all(results):
        _ok("pre-flight complete — core integrations reachable")
        return 0
    _fail("pre-flight had failures — see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
