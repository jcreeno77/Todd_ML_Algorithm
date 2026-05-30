# Massive.com Market Data

Polygon-compatible REST source behind `ML_tradingAlgo/data/massive_client.py`, a
drop-in for `schwab_client` selected via `DATA_PROVIDER=massive`. Auth is a plain
`MASSIVE_API_KEY` (no OAuth). Free tier serves ~18 months of 1-minute history.

## Avoiding rate-limit throttles

The free tier allows ~5 requests/minute and returns **HTTP 429** when exceeded.
Three measures keep us under it (all free, no plan upgrade):

1. **Proactive token bucket** (`massive_client._throttle`) — paces every request to
   the ceiling instead of bursting into 429s. This is *faster* than reacting,
   because the 429 backoff (15s) costs more than the ~12s spacing it replaces.
   Tunable via `MASSIVE_RATE_PER_MIN` (default `5`; set `0` to disable on a paid,
   uncapped plan). The old `Retry-After` backoff stays as a fallback.

2. **Grouped-daily disk cache** (`MASSIVE_CACHE_DIR`, default `~/.cache/massive_grouped`)
   — a *closed* session's data never changes, so each day's whole-market grouped
   frame (~12k tickers) is cached as parquet on first fetch. Repeated/overlapping
   scans then cost **zero** API calls. Today's still-open session is never cached;
   empty holidays are cached too (so reruns skip them).

3. **Skip non-trading days** — the scanner (`scan_gappers`) skips Sat/Sun before
   calling, rather than spending a quota'd request on a day that returns empty.

### Other levers (not yet implemented)
- One `/snapshot/.../gainers` call replaces a full market scan **for the current
  day** only.
- Polygon-style **flat files** (bulk daily dumps, no per-call limit) are the real
  zero-throttle path for historical bulk — typically a **paid** tier, so out of
  scope under the project's cost discipline without sign-off.

## Known caveat — float proxy
`float_shares` uses `share_class_shares_outstanding` (current, not as-of-date) as a
PROXY; Massive exposes no true public float. This weakens the `<50M float` screen;
flagged for later tightening.
