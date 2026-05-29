"""Training-label generator for the TFT momentum-trading model.

Produces two labels per gap-up event, consumed by the dual-head TFT model:

* ``y_win``   -- continuation probability target (BCE head), in ``[0, 1]``.
* ``y_offset``-- ATR-normalized optimal-entry offset (MSE head), unbounded.

The intelligence lives in :func:`compute_labels`, a pure function with no I/O.
:func:`build_labeled_dataset` is a thin wrapper that joins the persisted
``events/gap_ups`` table with ``bars_minute`` from the store and emits one
labeled row per event.

----------------------------------------------------------------------------
Label definitions (single source of truth)
----------------------------------------------------------------------------

Entry price
    The ``close`` of the entry bar (``bars.iloc[entry_bar_idx]``).

Lookahead window
    The next ``lookahead_bars`` bars AFTER the entry bar (entry bar excluded),
    truncated at the end of available data.

Triple-barrier for ``y_win``
    * TP level = ``entry * (1 + tp_pct/100)``;  hit when a bar ``high >= TP``.
    * SL level = ``entry * (1 - sl_pct/100)``;  hit when a bar ``low  <= SL``.
    * Walk bars in order. The FIRST barrier touched decides the outcome:
        - TP first  -> ``y_win = 1.0``
        - SL first  -> ``y_win = 0.0``
    * Tie (both TP and SL within the SAME bar): treated as SL-first
      (conservative) -> ``y_win = 0.0``.
    * Neither barrier touched within the window -> a proportional, binarized
      score based on the final return::

          final_return = (last_close - entry) / entry
          y_win = 0.5 + 0.5 * clip(final_return / (tp_pct / 100), -1, 1)

      so a flat finish maps to 0.5, a finish at +tp_pct to 1.0, and a finish
      at -tp_pct to 0.0, clipped beyond.

``y_offset`` (ATR-normalized best-entry distance)
    ``y_offset = (entry - min_low) / ATR``
    where ``min_low`` is the lowest ``low`` over the lookahead window and
    ``ATR`` is the average true range computed over the PRE-entry bars
    (bars ``0 .. entry_bar_idx-1``):

        * Use a standard 14-period ATR (mean of the true range over the last
          14 pre-entry bars) when at least 14 pre-entry bars exist;
        * otherwise use the mean true range over all available pre-entry bars.

    True range for bar ``i`` is the standard
    ``max(high-low, |high-prev_close|, |low-prev_close|)``; the first
    pre-entry bar (no prior close) uses ``high - low``.

    A positive ``y_offset`` means price dipped below the entry close (a better
    entry was available); negative means price never traded below entry.
    Guard: if ``ATR == 0`` the offset is undefined and returns ``0.0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure label computation
# ---------------------------------------------------------------------------

def _average_true_range(pre_bars: pd.DataFrame, period: int = 14) -> float:
    """Average true range over the pre-entry bars.

    Uses the last ``period`` bars when available, else all of them. The first
    bar (no prior close) contributes ``high - low``.
    """
    if len(pre_bars) == 0:
        return 0.0

    highs = pre_bars["high"].to_numpy(dtype=float)
    lows = pre_bars["low"].to_numpy(dtype=float)
    closes = pre_bars["close"].to_numpy(dtype=float)

    prev_close = np.empty_like(closes)
    prev_close[0] = np.nan
    prev_close[1:] = closes[:-1]

    hl = highs - lows
    hc = np.abs(highs - prev_close)
    lc = np.abs(lows - prev_close)

    true_range = np.where(
        np.isnan(prev_close),
        hl,
        np.maximum(hl, np.maximum(hc, lc)),
    )

    if len(true_range) >= period:
        true_range = true_range[-period:]

    return float(np.mean(true_range))


def compute_labels(
    bars: pd.DataFrame,
    entry_bar_idx: int,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
) -> tuple[float, float]:
    """Compute ``(y_win, y_offset)`` for one entry.

    Parameters
    ----------
    bars
        OHLC DataFrame with at least ``high``, ``low``, ``close`` columns,
        ordered chronologically (positional indexing via ``.iloc``).
    entry_bar_idx
        Positional index of the entry bar; entry price is its ``close``.
    tp_pct, sl_pct
        Take-profit / stop-loss thresholds in percent.
    lookahead_bars
        Number of bars after the entry bar to evaluate.

    Returns
    -------
    (y_win, y_offset)
        See module docstring for exact definitions.
    """
    n = len(bars)
    if entry_bar_idx < 0 or entry_bar_idx >= n:
        raise IndexError(f"entry_bar_idx {entry_bar_idx} out of range for {n} bars")

    entry = float(bars.iloc[entry_bar_idx]["close"])

    tp_frac = tp_pct / 100.0
    sl_frac = sl_pct / 100.0
    tp_level = entry * (1.0 + tp_frac)
    sl_level = entry * (1.0 - sl_frac)

    # Lookahead window: bars strictly after the entry bar, truncated at end.
    start = entry_bar_idx + 1
    stop = min(start + lookahead_bars, n)
    window = bars.iloc[start:stop]

    # ----- y_win via triple barrier -----
    y_win: float | None = None
    if len(window) > 0:
        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        for hi, lo in zip(highs, lows):
            sl_hit = lo <= sl_level
            tp_hit = hi >= tp_level
            # Conservative tie-break: if both barriers fall in the same bar,
            # assume SL was reached first.
            if sl_hit:
                y_win = 0.0
                break
            if tp_hit:
                y_win = 1.0
                break

    if y_win is None:
        # Neither barrier touched (or empty window) -> proportional score.
        if len(window) > 0:
            last_close = float(window.iloc[-1]["close"])
        else:
            last_close = entry
        final_return = (last_close - entry) / entry if entry != 0 else 0.0
        ratio = final_return / tp_frac if tp_frac != 0 else 0.0
        y_win = 0.5 + 0.5 * float(np.clip(ratio, -1.0, 1.0))

    # ----- y_offset -----
    pre_bars = bars.iloc[:entry_bar_idx]
    atr = _average_true_range(pre_bars)
    if atr == 0 or len(window) == 0:
        y_offset = 0.0
    else:
        min_low = float(window["low"].min())
        y_offset = (entry - min_low) / atr

    return float(y_win), float(y_offset)


# ---------------------------------------------------------------------------
# Store-backed wrapper
# ---------------------------------------------------------------------------

# Imported lazily inside build_labeled_dataset so tests can monkeypatch this
# module attribute even before the store module exists. Declared here so
# ``monkeypatch.setattr(labeler, "read_bars", ...)`` always has a target.
read_bars = None  # type: ignore[assignment]

# Regular trading hours open (US equities), Eastern Time.
MARKET_OPEN_TIME = pd.Timestamp("09:30").time()


def _ensure_read_bars():
    """Return the store ``read_bars`` callable, importing lazily if needed."""
    global read_bars
    if read_bars is None:
        from ML_tradingAlgo.data.store import read_bars as _rb  # lazy import

        read_bars = _rb
    return read_bars


def _first_regular_hours_idx(bars: pd.DataFrame) -> int:
    """Positional index of the first bar at/after 09:30 ET (the entry bar).

    Falls back to index 0 when the index is not a DatetimeIndex or no bar is
    at/after the open.
    """
    idx = bars.index
    if isinstance(idx, pd.DatetimeIndex):
        times = idx.tz_convert("America/New_York").time if idx.tz is not None else idx.time
        mask = np.array([t >= MARKET_OPEN_TIME for t in times])
        positions = np.nonzero(mask)[0]
        if len(positions) > 0:
            return int(positions[0])
    return 0


def build_labeled_dataset(
    session_date_range: tuple,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
) -> pd.DataFrame:
    """Build a labeled dataset by joining gap-up events with minute bars.

    Reads the ``events`` table for the given session date range, then for each
    event loads its ``bars_minute`` session, determines the entry bar (first
    regular-hours bar at/after 09:30 ET), and computes ``y_win``/``y_offset``.

    Returns one row per event: all static event fields plus ``y_win`` and
    ``y_offset``. Events whose bars are missing/empty are skipped.
    """
    rb = _ensure_read_bars()

    events = rb("events", date_range=session_date_range)
    if events is None or len(events) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, event in events.iterrows():
        symbol = event["symbol"]
        bars = rb("bars_minute", symbol=symbol, date_range=session_date_range)
        if bars is None or len(bars) == 0:
            continue

        entry_idx = _first_regular_hours_idx(bars)
        y_win, y_offset = compute_labels(
            bars,
            entry_idx,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            lookahead_bars=lookahead_bars,
        )

        row = event.to_dict()
        row["y_win"] = y_win
        row["y_offset"] = y_offset
        rows.append(row)

    return pd.DataFrame(rows)
