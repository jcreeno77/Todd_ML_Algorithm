"""Live-loop tee: a non-blocking sidecar that mirrors the live trading hot path
into the S3 Parquet warehouse without ever blocking the trading loop.

This module is consumed by ``live_data_gather_unified.py``. The live loop builds
1-minute and 5-minute candles as plain 5-element sequences
``[open, high, low, close, volume]``; on each completed candle it calls
:meth:`LiveTee.put_candle`, which is **strictly non-blocking** (a single
``queue.put_nowait``). A daemon thread drains the queue, batches candles and
writes them to the warehouse via :func:`ML_tradingAlgo.data.store.write_bars`.

Design / safety
---------------
* ``put_candle`` runs on the trading thread and must NEVER block or raise.
  It only enqueues. If the queue is full the candle is dropped and logged --
  losing a mirrored bar is always preferable to stalling order execution.
* All S3 / warehouse I/O happens on the daemon thread.
* New-symbol observations are forwarded to
  :func:`ML_tradingAlgo.data.gainers.record_observation` from the daemon thread.
* ``update_state`` writes a tiny JSON snapshot to ``live/state/{id}.json`` for
  the live dashboard. Best-effort; failures are logged, never raised.

Tables
------
* resolution ``"1min"`` -> table ``"bars_minute"``
* anything else        -> table ``"bars_5min"``

All rows are written with ``source="live_tick_agg"``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import queue
import threading
from zoneinfo import ZoneInfo

import pandas as pd

from ML_tradingAlgo.data import gainers
from ML_tradingAlgo.data.store import write_bars

__all__ = ["LiveTee"]

_LOG = logging.getLogger(__name__)

_SOURCE = "live_tick_agg"
_TABLE_1MIN = "bars_minute"
_TABLE_5MIN = "bars_5min"
_PARTITION_COLS = ["symbol", "session_date"]
_ET = ZoneInfo("America/New_York")

# Sentinel pushed onto the queue to wake the worker for an unconditional flush /
# shutdown. Kept module-private so it can never collide with a real candle.
_STOP = object()
_FLUSH = object()


def _table_for(resolution: str) -> str:
    return _TABLE_1MIN if resolution == "1min" else _TABLE_5MIN


def _et_date(ts: pd.Timestamp) -> dt.date:
    """ET session date for a UTC timestamp."""
    return ts.tz_convert(_ET).date()


class LiveTee:
    """Async, non-blocking mirror of the live candle stream into the warehouse."""

    def __init__(
        self,
        instance_id,
        flush_every: int = 20,
        flush_interval: float = 10.0,
        max_queue: int = 10000,
    ):
        self.instance_id = instance_id
        self.flush_every = max(1, int(flush_every))
        self.flush_interval = float(flush_interval)
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        # symbols already forwarded to gainers this session (daemon-thread only)
        self._seen_symbols: set[str] = set()

    # ------------------------------------------------------------------ #
    # producer side (trading thread) -- MUST be non-blocking
    # ------------------------------------------------------------------ #
    def put_candle(
        self,
        symbol: str,
        candle,
        resolution: str,
        session_date=None,
    ) -> None:
        """Enqueue a completed candle. Non-blocking; never raises.

        ``candle`` is a 5-element sequence ``[open, high, low, close, volume]``.
        On a full queue the candle is dropped (and logged) rather than blocking
        the trading loop.
        """
        item = (symbol, list(candle), resolution, session_date, pd.Timestamp.now(tz="UTC"))
        try:
            self._q.put_nowait(item)
        except queue.Full:
            _LOG.warning("LiveTee queue full; dropping %s %s candle", symbol, resolution)
        except Exception as exc:  # pragma: no cover - defensive; must never raise
            _LOG.warning("LiveTee.put_candle failed (dropped): %s", exc)

    def update_state(self, state: dict) -> None:
        """Enqueue a dashboard state snapshot for the daemon thread to persist.

        Non-blocking. The actual S3 write happens on the worker thread.
        """
        snapshot = dict(state)
        snapshot["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        try:
            self._q.put_nowait(("__state__", snapshot))
        except queue.Full:
            _LOG.warning("LiveTee queue full; dropping state update")
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("LiveTee.update_state failed (dropped): %s", exc)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"LiveTee-{self.instance_id}", daemon=True
        )
        self._thread.start()

    def stop(self, drain: bool = True) -> None:
        """Signal the worker to stop and join it, flushing any remainder."""
        if not self._running:
            return
        self._running = False
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            # Worker will still notice _running flipped via timeout polling.
            pass
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def flush(self) -> None:
        """Request an out-of-band flush of the current batch (best-effort)."""
        try:
            self._q.put_nowait(_FLUSH)
        except queue.Full:
            pass

    # ------------------------------------------------------------------ #
    # consumer side (daemon thread)
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        batch: list[dict] = []
        last_flush = pd.Timestamp.now(tz="UTC")

        while True:
            timeout = self.flush_interval
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = _FLUSH

            stop = False
            if item is _STOP:
                stop = True
            elif item is _FLUSH:
                pass
            elif isinstance(item, tuple) and item and item[0] == "__state__":
                self._write_state(item[1])
            else:
                self._handle_candle(item, batch)

            now = pd.Timestamp.now(tz="UTC")
            time_due = (now - last_flush).total_seconds() >= self.flush_interval
            size_due = len(batch) >= self.flush_every
            if batch and (stop or time_due or size_due):
                self._flush_batch(batch)
                batch = []
                last_flush = now
            elif not batch:
                last_flush = now

            if stop:
                # Final drain: pull anything still queued, then flush remainder.
                self._drain_remaining(batch)
                if batch:
                    self._flush_batch(batch)
                    batch = []
                break

    def _drain_remaining(self, batch: list[dict]) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP or item is _FLUSH:
                continue
            if isinstance(item, tuple) and item and item[0] == "__state__":
                self._write_state(item[1])
                continue
            self._handle_candle(item, batch)

    def _handle_candle(self, item, batch: list[dict]) -> None:
        try:
            symbol, candle, resolution, session_date, ts = item
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("LiveTee got malformed queue item (skipped): %s", exc)
            return

        sd = self._resolve_session_date(session_date, ts)
        self._maybe_record_gainer(symbol, sd)

        o, h, low, c, v = (list(candle) + [None] * 5)[:5]
        batch.append(
            {
                "symbol": symbol,
                "ts": ts,
                "session_date": sd,
                "resolution": resolution,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": v,
                "_table": _table_for(resolution),
            }
        )

    def _resolve_session_date(self, session_date, ts: pd.Timestamp) -> dt.date:
        if session_date is not None:
            if isinstance(session_date, dt.datetime):
                return session_date.date()
            if isinstance(session_date, pd.Timestamp):
                return session_date.date()
            if isinstance(session_date, dt.date):
                return session_date
            return pd.Timestamp(session_date).date()
        return _et_date(ts)

    def _maybe_record_gainer(self, symbol: str, session_date: dt.date) -> None:
        if symbol in self._seen_symbols:
            return
        self._seen_symbols.add(symbol)
        try:
            gainers.record_observation(
                symbol,
                {"observed_by": "live_scanner", "session_date": session_date},
            )
        except Exception as exc:
            _LOG.warning("LiveTee gainers.record_observation failed for %s: %s", symbol, exc)

    def _flush_batch(self, batch: list[dict]) -> None:
        if not batch:
            return
        df = pd.DataFrame(batch)
        # Drop the routing column before writing; write each table separately.
        for table, group in df.groupby("_table", sort=False):
            out = group.drop(columns=["_table"])
            try:
                write_bars(
                    out,
                    table=table,
                    partition_cols=_PARTITION_COLS,
                    source=_SOURCE,
                )
            except Exception as exc:
                _LOG.warning("LiveTee flush to %s failed (%d rows dropped): %s", table, len(out), exc)

    # ------------------------------------------------------------------ #
    # dashboard state -> S3
    # ------------------------------------------------------------------ #
    def _state_path(self) -> str:
        bucket = os.environ.get("S3_BUCKET")
        if not bucket:
            raise ValueError("S3_BUCKET environment variable is not set")
        prefix = os.environ.get("S3_PREFIX", "").strip("/")
        parts = [bucket]
        if prefix:
            parts.append(prefix)
        parts += ["live", "state", f"{self.instance_id}.json"]
        return "/".join(parts)

    def _write_state(self, snapshot: dict) -> None:
        try:
            import s3fs

            endpoint = os.environ.get("AWS_ENDPOINT_URL")
            client_kwargs = {}
            if endpoint:
                client_kwargs["endpoint_url"] = endpoint
            fs = s3fs.S3FileSystem(skip_instance_cache=True, client_kwargs=client_kwargs)
            path = self._state_path()
            with fs.open(path, "wb") as f:
                f.write(json.dumps(snapshot, default=str).encode("utf-8"))
        except Exception as exc:
            _LOG.warning("LiveTee.update_state write failed: %s", exc)
