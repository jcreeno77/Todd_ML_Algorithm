"""Assembly bridge: labeled gap-up events -> model-ready stacked tensor arrays.

This is the highest-risk module in the TFT pipeline because it is the single
point where *features* and *labels* are aligned to the same entry bar. A drift
of even one bar between the feature window and the label lookahead would
silently corrupt every training example. The invariant below is therefore the
load-bearing contract of this module.

ONE-PASS ALIGNMENT INVARIANT
----------------------------
For a single session we read ``bars_minute`` once, sort by ``ts`` and install a
``DatetimeIndex`` (so ``labeler._first_regular_hours_idx`` can actually find the
09:30 ET open instead of silently falling back to index 0). Then, from that one
frame and one ``entry_idx``:

    * FEATURES come from ``bars.iloc[:entry_idx + 1]`` -- the model sees the
      pre-entry bars plus the entry bar itself (inclusive slice).
    * LABELS come from ``labeler.compute_labels(bars, entry_idx, ...)`` on the
      SAME frame, looking strictly forward from ``entry_idx``.

Because both halves reference the identical ``bars`` object and the identical
``entry_idx``, features and labels can never drift apart. Test #8
(``test_alignment_guard``) pins this by recomputing the labels independently.

DatetimeIndex caveat
--------------------
``store.read_bars`` returns ``ts`` as a *column*, not an index. We must do::

    bars = bars.sort_values("ts").set_index(pd.DatetimeIndex(bars["ts"]))

before entry-bar detection, or ``_first_regular_hours_idx`` returns 0 and the
entry silently misaligns to the first premarket bar.

min_bars guard
--------------
``build_feature_matrix`` does NOT pad. If fewer than ``sequence_length + 5``
bars are available up to and including the entry bar, the event is skipped with
reason ``"short_session"``.

Dependency injection
--------------------
``read_bars``, ``fundamentals_lookup`` and ``spy_lookup`` are accepted as
callables (defaulting to the real store / fundamentals functions when ``None``)
so tests can inject synthetic data and never touch S3 or the network.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ML_tradingAlgo.data import labeler as _labeler
from ML_tradingAlgo.data import fundamentals as _fundamentals
from ML_tradingAlgo.tft import features as _features
from ML_tradingAlgo.tft.augment import _OHLC


_DAYS_PER_MONTH = 30.44

# OHLCV columns a bar_transform sees / writes back, positionally.
_OHLCV = _OHLC + ["volume"]


# --------------------------------------------------------------------------- #
# default dependency resolvers (lazy so importing this module is cheap / safe)
# --------------------------------------------------------------------------- #
def _default_read_bars():
    from ML_tradingAlgo.data.store import read_bars as _rb

    return _rb


# --------------------------------------------------------------------------- #
# 1-min -> 5-min resampling
# --------------------------------------------------------------------------- #
def resample_1min_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min OHLCV bars into 5-min bars.

    open=first, high=max, low=min, close=last, volume=sum, grouped by the
    5-minute floor of each bar's timestamp.

    The timestamp source is, in order of preference: a ``DatetimeIndex`` on the
    frame, else a ``ts`` column. Returns a frame with columns
    ``[open, high, low, close, volume]`` indexed by the 5-min bucket start.
    """
    if isinstance(bars_1min.index, pd.DatetimeIndex):
        ts = bars_1min.index
    elif "ts" in bars_1min.columns:
        ts = pd.DatetimeIndex(pd.to_datetime(bars_1min["ts"]))
    else:
        raise ValueError("resample_1min_to_5min needs a DatetimeIndex or a 'ts' column")

    bucket = ts.floor("5min")
    grouped = bars_1min.groupby(bucket, sort=True)
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(),
    })
    out.index.name = None
    return out[["open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------- #
# static_data assembly
# --------------------------------------------------------------------------- #
def build_static_data(
    event_row: dict,
    bars_1min: pd.DataFrame,
    fundamentals_row: dict | None,
    spy_bars=None,
    vix_level=None,
    sector_etf_return=None,
    avg_daily_volume_20d=None,
    current_price: float | None = None,
) -> dict:
    """Assemble the static_data dict that ``build_feature_matrix`` consumes.

    Sourcing (see module / task spec):
        * ``float_shares`` from ``event_row`` (caller skips on missing/0).
        * ``short_interest_ratio`` from fundamentals else 0.0.
        * ``gap_percentage`` from ``event_row['gap_pct']`` -- a FRACTION (e.g.
          0.35), passed through as-is to match live usage.
        * ``sector_id`` via ``fundamentals.sector_name_to_id`` (unknown/None ->
          0; cardinality is exactly 11, no spare bucket).
        * ``days_since_earnings`` via the fundamentals helper else 0.0.
        * ``high_52wk`` / ``low_52wk`` from fundamentals if present, else
          ``current_price`` (neutral ratios).
        * ``premarket_high`` / ``premarket_low`` from ``event_row`` else
          ``prior_close``.
        * ``current_price`` = close of the entry bar (passed in by the caller).
        * ``prior_close`` from ``event_row``.
        * ``avg_daily_volume_20d`` passed value else proxy
          ``bars_1min['volume'].sum()``.
        * ``spy_bars`` / ``vix_level`` / ``sector_etf_return`` passed through
          (None is fine; features.py neutralizes them).

    Guarantees every continuous field is finite (no None/NaN) and ``sector_id``
    is an int in 0..10.
    """
    fund = fundamentals_row or {}

    prior_close = _finite(event_row.get("prior_close"), default=0.0)

    if current_price is None:
        # Fall back to the last close available if the caller did not supply one.
        current_price = float(bars_1min["close"].iloc[-1]) if len(bars_1min) else prior_close
    current_price = _finite(current_price, default=prior_close)

    float_shares = _finite(event_row.get("float_shares"), default=0.0)

    short_interest_ratio = _finite(fund.get("short_interest_ratio"), default=0.0)

    # gap_pct is a fraction; pass through as-is.
    gap_percentage = _finite(event_row.get("gap_pct"), default=0.0)

    # sector_id: prefer a precomputed int id, else map a sector name. Unknown -> 0.
    sector_id = fund.get("sector_id")
    if sector_id is None:
        sector_id = _fundamentals.sector_name_to_id(fund.get("sector"))
    if sector_id is None:
        sector_id = 0
    sector_id = int(sector_id)
    if sector_id < 0 or sector_id > 10:
        sector_id = 0

    # days_since_earnings: prefer a precomputed value, else derive from earnings_date.
    dse = fund.get("days_since_earnings")
    if dse is None and fund.get("earnings_date") is not None:
        dse = _fundamentals.days_since_earnings(
            fund.get("earnings_date"), event_row.get("session_date")
        )
    days_since_earnings = _finite(dse, default=0.0)

    high_52wk = _finite(fund.get("high_52wk"), default=current_price)
    low_52wk = _finite(fund.get("low_52wk"), default=current_price)

    premarket_high = _finite(event_row.get("premarket_high"), default=prior_close)
    premarket_low = _finite(event_row.get("premarket_low"), default=prior_close)

    if avg_daily_volume_20d is None:
        avg_daily_volume_20d = float(bars_1min["volume"].sum()) if len(bars_1min) else 0.0
    avg_daily_volume_20d = _finite(avg_daily_volume_20d, default=0.0)

    return {
        "float_shares": float_shares,
        "short_interest_ratio": short_interest_ratio,
        "gap_percentage": gap_percentage,
        "sector_id": sector_id,
        "days_since_earnings": days_since_earnings,
        "high_52wk": high_52wk,
        "low_52wk": low_52wk,
        "premarket_high": premarket_high,
        "premarket_low": premarket_low,
        "current_price": current_price,
        "prior_close": prior_close,
        "avg_daily_volume_20d": avg_daily_volume_20d,
        # pass-through market context (None tolerated downstream)
        "spy_bars": spy_bars,
        "vix_level": vix_level,
        "sector_etf_return": sector_etf_return,
        # new Task-5 fields
        "session_date": event_row.get("session_date"),
        "prior_day_high": _finite(event_row.get("prior_day_high"), default=prior_close),
        "prior_day_range": _finite(event_row.get("prior_day_range"), default=0.0),
        "day_of_run": int(_finite(event_row.get("day_of_run"), default=1) or 1),
        "intraday_volume_profile": event_row.get("intraday_volume_profile"),
    }


def _finite(value, default: float) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``."""
    if value is None:
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return f


# --------------------------------------------------------------------------- #
# single-event assembly
# --------------------------------------------------------------------------- #
def assemble_event(
    event_row,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
    sequence_length: int = 30,
    min_bars: int = 35,
    read_bars=None,
    fundamentals_lookup=None,
    spy_bars=None,
    bar_transform=None,
) -> dict | None:
    """Assemble ONE labeled event into model-ready arrays.

    Returns a dict::

        {
            "temporal": np.ndarray (sequence_length, 69) float32,
            "static_continuous": np.ndarray (11,) float32,
            "static_categorical": np.ndarray (1,) int64,
            "y_win": float,
            "y_offset": float,
            "session_date": <session date>,
            "symbol": str,
        }

    or ``None`` (with the ``reason`` recorded by the caller) when the event is
    unusable: missing/zero float, missing bars, or too few bars (< min_bars up
    to and including the entry bar).

    On its own ``assemble_event`` cannot record a skip reason in a shared list,
    so it returns ``None``; ``assemble_dataset`` re-derives the reason. Callers
    needing the reason should use :func:`_assemble_event_with_reason`.

    ``bar_transform`` is an optional callable ``(df: DataFrame) -> DataFrame``
    that receives the raw 1-min OHLCV bars (with DatetimeIndex already set)
    BEFORE entry detection and feature engineering. It must return a
    length-preserving frame with the same index. When ``None`` (default) the
    bars are used unmodified.
    """
    result = _assemble_event_with_reason(
        event_row,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        lookahead_bars=lookahead_bars,
        sequence_length=sequence_length,
        min_bars=min_bars,
        read_bars=read_bars,
        fundamentals_lookup=fundamentals_lookup,
        spy_bars=spy_bars,
        bar_transform=bar_transform,
    )
    return result[0]


def _assemble_event_with_reason(
    event_row,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
    sequence_length: int = 30,
    min_bars: int = 35,
    read_bars=None,
    fundamentals_lookup=None,
    spy_bars=None,
    bar_transform=None,
) -> tuple[dict | None, str | None]:
    """Like :func:`assemble_event` but returns ``(result, skip_reason)``.

    ``bar_transform`` — optional callable applied to the raw OHLCV bars after
    the DatetimeIndex is set and BEFORE entry detection / feature engineering.
    Must be length-preserving; alignment is positional. A length-changing
    transform is rejected as a failed assembly (reason
    ``"bar_transform_length_mismatch"``) rather than crashing the run.

    This reads ``bars_minute`` from the store exactly ONCE, then delegates the
    post-load assembly to :func:`_assemble_from_bars`. Callers that already hold
    the loaded frame (e.g. the augment loop) call ``_assemble_from_bars``
    directly to avoid re-reading S3.
    """
    if hasattr(event_row, "to_dict"):
        event_row = event_row.to_dict()

    symbol = event_row.get("symbol")
    session_date = event_row.get("session_date")

    # float guard up front: candle pressure divides by float.
    float_shares = event_row.get("float_shares")
    if float_shares is None or _finite(float_shares, default=0.0) == 0.0:
        return None, "no_float"

    rb = read_bars if read_bars is not None else _default_read_bars()

    date_range = (session_date, session_date)
    bars = rb("bars_minute", symbol=symbol, date_range=date_range)
    if bars is None or len(bars) == 0:
        return None, "no_bars"

    return _assemble_from_bars(
        event_row,
        bars,
        bar_transform=bar_transform,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        lookahead_bars=lookahead_bars,
        sequence_length=sequence_length,
        min_bars=min_bars,
        fundamentals_lookup=fundamentals_lookup,
        spy_bars=spy_bars,
    )


def _assemble_from_bars(
    event_row,
    bars,
    *,
    bar_transform=None,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
    sequence_length: int = 30,
    min_bars: int = 35,
    fundamentals_lookup=None,
    spy_bars=None,
) -> tuple[dict | None, str | None]:
    """Post-load assembly: given already-read ``bars`` for one event, build a
    sample. Does NOT touch the store, so it can be re-run for every augmented
    variant of an event from a single S3 read.

    ``bars`` is defensively copied at the top so the caller's frame is never
    mutated and each ``bar_transform`` starts from clean bars.

    ``bar_transform`` — optional callable applied to the raw OHLCV bars after
    the DatetimeIndex is set and BEFORE entry detection / feature engineering.
    It must return a length-preserving frame; columns are written back BY
    POSITION. A length mismatch returns ``(None, "bar_transform_length_mismatch")``
    so the dataset loop's None-skip handles it gracefully.
    """
    if hasattr(event_row, "to_dict"):
        event_row = event_row.to_dict()

    # Never mutate the caller's frame; each variant starts from clean bars.
    bars = bars.copy()

    symbol = event_row.get("symbol")
    session_date = event_row.get("session_date")

    # --- one-pass: sort by ts and install a DatetimeIndex --------------------
    if "ts" not in bars.columns:
        return None, "no_ts"
    bars = bars.sort_values("ts")
    bars = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["ts"])))

    # --- optional bar-level augmentation (applied to raw bars BEFORE features)
    if bar_transform is not None:
        transformed = bar_transform(bars[_OHLCV])
        # Enforce the length-preserving contract; alignment is positional.
        if len(transformed) != len(bars):
            return None, "bar_transform_length_mismatch"
        for col in _OHLCV:
            bars[col] = transformed[col].to_numpy()

    entry_idx = _labeler._first_regular_hours_idx(bars)

    # min_bars guard: bars up to & including entry must be >= sequence_length + 5.
    bars_through_entry = entry_idx + 1
    if bars_through_entry < max(min_bars, sequence_length + 5):
        return None, "short_session"

    # --- features from bars.iloc[:entry_idx + 1] -----------------------------
    feat_bars = bars.iloc[: entry_idx + 1]
    current_price = float(bars.iloc[entry_idx]["close"])

    fundamentals_row = None
    if fundamentals_lookup is not None:
        fundamentals_row = fundamentals_lookup(symbol, session_date)

    static_data = build_static_data(
        event_row,
        feat_bars,
        fundamentals_row,
        spy_bars=spy_bars,
        current_price=current_price,
    )

    bars_5min = resample_1min_to_5min(feat_bars)

    temporal, static_cont, static_cat = _features.build_feature_matrix(
        feat_bars,
        bars_5min,
        static_data,
        sequence_length=sequence_length,
    )

    # --- labels from the SAME bars + entry_idx, looking forward --------------
    y_win, y_offset = _labeler.compute_labels(
        bars,
        entry_idx,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        lookahead_bars=lookahead_bars,
    )

    result = {
        "temporal": np.asarray(temporal, dtype=np.float32),
        "static_continuous": np.asarray(static_cont, dtype=np.float32),
        "static_categorical": np.asarray(static_cat, dtype=np.int64),
        "y_win": float(y_win),
        "y_offset": float(y_offset),
        "session_date": session_date,
        "symbol": symbol,
    }
    return result, None


# --------------------------------------------------------------------------- #
# dataset assembly
# --------------------------------------------------------------------------- #
def assemble_dataset(
    session_date_range,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
    sequence_length: int = 30,
    min_bars: int = 35,
    lambda_decay: float = 0.0578,
    asof=None,
    read_bars=None,
    fundamentals_lookup=None,
    spy_lookup=None,
    augment: bool = False,
    n_augment: int = 0,
    bar_transforms=None,
    augment_rng=None,
) -> dict:
    """Enumerate events, assemble each, stack into arrays, weight by recency.

    Events come from ``labeler.build_labeled_dataset`` (which itself reads the
    ``events`` table via the same ``read_bars`` we forward here). Each event is
    assembled with :func:`assemble_event`; failures are recorded in ``skipped``.

    Recency sample weights: ``w = exp(-lambda_decay * age_months)`` where
    ``age_months = (asof - session_date).days / 30.44``. ``asof`` defaults to the
    max session_date among the *kept* events, so the newest event has weight 1.0.

    Augmentation (train-only):
        When ``augment=True`` and ``n_augment > 0``, each successfully assembled
        original sample is followed immediately by up to ``n_augment`` augmented
        copies produced by re-running the post-load assembly
        (:func:`_assemble_from_bars`) on the SAME bars with a ``bar_transform``
        chosen from ``bar_transforms``. Bars are read from the store exactly ONCE
        per event and reused for the original and every augmented variant, so
        ``n_augment`` does not multiply S3 read volume. Augmented samples are
        tagged with ``is_augmented=True`` and ``parent_index`` pointing to the
        position of their original sample in the stacked arrays. Original samples
        have ``is_augmented=False`` and ``parent_index`` equal to their own
        position. A variant whose transform fails the length contract is skipped
        (no row, no dangling parent slot).

        ``bar_transforms`` — list of callables (one is chosen per augmentation).
        When falsy/empty, augmentation is SKIPPED for that event (treated as
        ``n_augment=0``) rather than emitting degenerate identity duplicates.

        ``augment_rng`` — ``np.random.RandomState`` governing only transform
        SELECTION among ``bar_transforms`` (a single default generator is created
        once for the whole run when ``None``). Each transform owns its own
        perturbation RNG (e.g. ``lambda df: jitter_bars(df, rng=...)``); for fully
        reproducible augmentation the caller must seed the RNGs inside the
        transforms in addition to passing ``augment_rng`` for selection.

    Returns::

        {
            "temporal": (N, 30, 69) float32,
            "static_continuous": (N, 11) float32,
            "static_categorical": (N, 1) int64,
            "y_win": (N,) float32,
            "y_offset": (N,) float32,
            "sample_weights": (N,) float32,
            "session_dates": list,
            "symbols": list,
            "skipped": list[(symbol, session_date, reason)],
            "is_augmented": (N,) bool,
            "parent_index": (N,) int64,
        }
    """
    rb = read_bars if read_bars is not None else _default_read_bars()

    # build_labeled_dataset reads events + bars; point its read_bars at ours so
    # tests never hit S3. Restore afterwards.
    prev_rb = _labeler.read_bars
    _labeler.read_bars = rb
    try:
        events = _labeler.build_labeled_dataset(
            session_date_range,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            lookahead_bars=lookahead_bars,
        )
    finally:
        _labeler.read_bars = prev_rb

    temporal_list: list[np.ndarray] = []
    static_cont_list: list[np.ndarray] = []
    static_cat_list: list[np.ndarray] = []
    y_win_list: list[float] = []
    y_offset_list: list[float] = []
    session_dates: list = []
    symbols: list = []
    skipped: list[tuple] = []
    is_aug_list: list[bool] = []
    parent_list: list[int] = []

    if events is None or len(events) == 0:
        return _empty_dataset(sequence_length)

    # Default selection RNG constructed ONCE so a single (unseeded) generator
    # spans all events rather than being re-seeded per event.
    _select_rng = augment_rng if augment_rng is not None else np.random.RandomState()

    # day_of_run: count consecutive prior session_dates for the same symbol.
    _run_map: dict = {}
    if events is not None and len(events):
        ev = events.copy()
        ev["_d"] = [_to_date(x) for x in ev["session_date"]]
        for sym, grp in ev.groupby("symbol"):
            dates = sorted(d for d in grp["_d"] if d is not None)
            run = 0
            prev = None
            for d in dates:
                if prev is not None and 0 < (d - prev).days <= 4:
                    run += 1
                else:
                    run = 1
                _run_map[(sym, d)] = run
                prev = d

    for _, event in events.iterrows():
        event_row = event.to_dict()
        symbol = event_row.get("symbol")
        session_date = event_row.get("session_date")
        event_row["day_of_run"] = _run_map.get(
            (symbol, _to_date(session_date)), 1
        )

        spy_bars = None
        if spy_lookup is not None:
            spy_bars = spy_lookup(session_date)

        # float guard up front so we never spend an S3 read on a no-float event.
        float_shares = event_row.get("float_shares")
        if float_shares is None or _finite(float_shares, default=0.0) == 0.0:
            skipped.append((symbol, session_date, "no_float"))
            continue

        # --- ONE S3 read per event; reused for original + every augment ------
        base_bars = rb(
            "bars_minute", symbol=symbol, date_range=(session_date, session_date)
        )
        if base_bars is None or len(base_bars) == 0:
            skipped.append((symbol, session_date, "no_bars"))
            continue

        result, reason = _assemble_from_bars(
            event_row,
            base_bars,
            bar_transform=None,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            lookahead_bars=lookahead_bars,
            sequence_length=sequence_length,
            min_bars=min_bars,
            fundamentals_lookup=fundamentals_lookup,
            spy_bars=spy_bars,
        )

        if result is None:
            skipped.append((symbol, session_date, reason))
            continue

        # Record the index this original will occupy in the stacked arrays.
        parent_pos = len(temporal_list)

        temporal_list.append(result["temporal"])
        static_cont_list.append(result["static_continuous"])
        static_cat_list.append(result["static_categorical"])
        y_win_list.append(result["y_win"])
        y_offset_list.append(result["y_offset"])
        session_dates.append(result["session_date"])
        symbols.append(result["symbol"])
        is_aug_list.append(False)
        parent_list.append(parent_pos)

        # --- augmentation (train-only) --------------------------------------
        # Empty/missing bar_transforms -> skip augmentation rather than emit
        # degenerate identity duplicates.
        if augment and n_augment > 0 and bar_transforms:
            for _ in range(n_augment):
                # Choose a transform randomly when multiple are provided.
                if len(bar_transforms) == 1:
                    tf = bar_transforms[0]
                else:
                    tf = bar_transforms[int(_select_rng.randint(0, len(bar_transforms)))]

                # Reuse the bars already read for the ORIGINAL — no extra S3 read.
                aug_result, _ = _assemble_from_bars(
                    event_row,
                    base_bars,
                    bar_transform=tf,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    lookahead_bars=lookahead_bars,
                    sequence_length=sequence_length,
                    min_bars=min_bars,
                    fundamentals_lookup=fundamentals_lookup,
                    spy_bars=spy_bars,
                )
                if aug_result is None:
                    continue
                temporal_list.append(aug_result["temporal"])
                static_cont_list.append(aug_result["static_continuous"])
                static_cat_list.append(aug_result["static_categorical"])
                y_win_list.append(aug_result["y_win"])
                y_offset_list.append(aug_result["y_offset"])
                session_dates.append(aug_result["session_date"])
                symbols.append(aug_result["symbol"])
                is_aug_list.append(True)
                parent_list.append(parent_pos)

    if not temporal_list:
        out = _empty_dataset(sequence_length)
        out["skipped"] = skipped
        return out

    sample_weights = _recency_weights(session_dates, lambda_decay=lambda_decay, asof=asof)

    return {
        "temporal": np.stack(temporal_list).astype(np.float32),
        "static_continuous": np.stack(static_cont_list).astype(np.float32),
        "static_categorical": np.stack(static_cat_list).astype(np.int64),
        "y_win": np.asarray(y_win_list, dtype=np.float32),
        "y_offset": np.asarray(y_offset_list, dtype=np.float32),
        "sample_weights": sample_weights.astype(np.float32),
        "session_dates": session_dates,
        "symbols": symbols,
        "skipped": skipped,
        "is_augmented": np.asarray(is_aug_list, dtype=bool),
        "parent_index": np.asarray(parent_list, dtype=np.int64),
    }


def _recency_weights(session_dates, lambda_decay: float, asof=None) -> np.ndarray:
    """``w = exp(-lambda_decay * age_months)``; newest event -> weight 1.0."""
    dates = [_to_date(d) for d in session_dates]
    if asof is None:
        asof = max(d for d in dates if d is not None)
    else:
        asof = _to_date(asof)

    weights = np.empty(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        if d is None:
            weights[i] = 1.0
            continue
        age_days = (asof - d).days
        age_months = age_days / _DAYS_PER_MONTH
        weights[i] = np.exp(-lambda_decay * age_months)
    return weights


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    s = str(value).strip()
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _empty_dataset(sequence_length: int) -> dict:
    return {
        "temporal": np.empty((0, sequence_length, 69), dtype=np.float32),
        "static_continuous": np.empty((0, 11), dtype=np.float32),
        "static_categorical": np.empty((0, 1), dtype=np.int64),
        "y_win": np.empty((0,), dtype=np.float32),
        "y_offset": np.empty((0,), dtype=np.float32),
        "sample_weights": np.empty((0,), dtype=np.float32),
        "session_dates": [],
        "symbols": [],
        "skipped": [],
        "is_augmented": np.empty((0,), dtype=bool),
        "parent_index": np.empty((0,), dtype=np.int64),
    }
