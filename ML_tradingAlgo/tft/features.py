"""Feature engineering pipeline for TFT momentum trader.

Computes all 55 features (8 static + 38 one-min temporal + 9 five-min aggregate)
from raw OHLCV bar data. Single code path for training and inference.

Legacy candle pressure formula preserved: (((close-low)-(high-close))/open*1000) * (vol/float*100)
The * 1000 scaling is CRITICAL for consistency with legacy training/inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

TEMPORAL_1MIN_FEATURE_NAMES: list[str] = [
    # Price Action (8)
    "open_vwap_ratio",
    "high_vwap_ratio",
    "low_vwap_ratio",
    "close_vwap_ratio",
    "atr_normalized_range",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # Volume Profile (8)
    "log_volume",
    "relative_volume_20d",
    "volume_ema5_ratio",
    "volume_ema10_ratio",
    "volume_ema20_ratio",
    "cumulative_volume_ratio",
    "volume_price_trend",
    "obv_slope",
    # VWAP Dynamics (5)
    "vwap_distance_atr",
    "vwap_slope_5bar",
    "vwap_touches_10bar",
    "bars_since_vwap_cross",
    "vwap_reclaim_flag",
    # Momentum (8)
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "macd_hist_accel",
    "roc_5",
    "roc_10",
    "stoch_k",
    "williams_r",
    # Market Context (4)
    "spy_return",
    "spy_rsi",
    "sector_etf_return",
    "vix_level",
    # Other (2)
    "mfi",
    "consecutive_candle_count",
    # Legacy Core Signal (3)
    "candle_pressure_weighted",
    "candle_pressure_unweighted",
    "candle_pressure_squared",
]

TEMPORAL_5MIN_FEATURE_NAMES: list[str] = [
    "5min_candle_pressure_weighted",
    "5min_candle_pressure_unweighted",
    "5min_candle_pressure_squared",
    "5min_ohlc_range_normalized",
    "5min_relative_volume",
    "5min_vwap_distance",
    "5min_rsi_14",
    "5min_macd_histogram",
    "5min_obv_slope",
]

STATIC_CONTINUOUS_FEATURE_NAMES: list[str] = [
    "float_shares_log",
    "short_interest_ratio",
    "gap_percentage",
    "days_since_earnings",
    "high_52wk_ratio",
    "low_52wk_ratio",
    "premarket_range",
]


# ---------------------------------------------------------------------------
# Helper: technical indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal, histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Session VWAP (cumulative from start of data)."""
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume * direction).cumsum()


def _obv_slope(close: pd.Series, volume: pd.Series, window: int = 5) -> pd.Series:
    """OBV slope over rolling window."""
    obv = _obv(close, volume)
    return obv.diff(window) / window


def _stochastic_k(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic %K."""
    lowest = low.rolling(period, min_periods=1).min()
    highest = high.rolling(period, min_periods=1).max()
    denom = (highest - lowest).replace(0, np.nan)
    return ((close - lowest) / denom * 100).fillna(50)


def _williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    denom = (highest - lowest).replace(0, np.nan)
    return (((highest - close) / denom) * -100).fillna(-50)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    typical = (high + low + close) / 3
    raw_mf = typical * volume
    delta = typical.diff()
    pos_mf = (raw_mf * (delta > 0)).rolling(period, min_periods=1).sum()
    neg_mf = (raw_mf * (delta < 0)).rolling(period, min_periods=1).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


# ---------------------------------------------------------------------------
# Legacy candle pressure
# ---------------------------------------------------------------------------

def compute_legacy_candle_pressure(
    close: float,
    low: float,
    high: float,
    open_: float,
    volume: float,
    float_shares: float,
) -> dict[str, float]:
    """Compute legacy candle pressure with * 1000 scaling.

    Formula: (((close-low) - (high-close)) / open) * 1000
    Weighted: unweighted * (volume / float_shares * 100)
    Squared: weighted ** 2
    """
    open_safe = open_ if open_ != 0 else 1e-8
    unweighted = (((close - low) - (high - close)) / open_safe) * 1000
    float_safe = float_shares if float_shares != 0 else 1e-8
    weighted = unweighted * (volume / float_safe * 100)
    squared = weighted ** 2
    return {"weighted": weighted, "unweighted": unweighted, "squared": squared}


def _compute_legacy_candle_pressure_series(
    df: pd.DataFrame, float_shares: float
) -> pd.DataFrame:
    """Vectorized legacy candle pressure for a DataFrame of bars."""
    open_safe = df["open"].replace(0, np.nan).fillna(1e-8)
    float_safe = float_shares if float_shares != 0 else 1e-8
    unweighted = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / open_safe) * 1000
    weighted = unweighted * (df["volume"] / float_safe * 100)
    squared = weighted ** 2
    return pd.DataFrame({
        "weighted": weighted,
        "unweighted": unweighted,
        "squared": squared,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Temporal 1-min features (38)
# ---------------------------------------------------------------------------

def compute_temporal_features_1min(
    bars: pd.DataFrame,
    float_shares: float,
    avg_volume_20d: float,
    spy_bars: Optional[pd.DataFrame],
    vix_level: Optional[float],
    sector_etf_return: Optional[float],
) -> np.ndarray:
    """Compute 38 temporal features per 1-min bar.

    Args:
        bars: DataFrame with columns [open, high, low, close, volume].
        float_shares: Outstanding float shares.
        avg_volume_20d: 20-day average daily volume.
        spy_bars: Optional SPY 1-min bars (same length) for market context.
        vix_level: Current VIX level (scalar).
        sector_etf_return: Sector ETF session return (scalar).

    Returns:
        np.ndarray of shape (n_bars, 38).
    """
    df = bars.copy().reset_index(drop=True)
    n = len(df)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ATR
    atr = _atr(h, l, c)
    atr_safe = atr.replace(0, np.nan).fillna(method="bfill").fillna(0.01)

    # VWAP
    vwap = _vwap(h, l, c, v)
    vwap_safe = vwap.replace(0, np.nan).fillna(method="bfill").fillna(c)

    # --- Price Action (8) ---
    open_vwap = o / vwap_safe
    high_vwap = h / vwap_safe
    low_vwap = l / vwap_safe
    close_vwap = c / vwap_safe

    total_range = (h - l).replace(0, np.nan).fillna(1e-8)
    atr_norm_range = total_range / atr_safe

    body = (c - o).abs()
    body_ratio = body / total_range
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range

    # --- Volume Profile (8) ---
    log_vol = np.log(v.replace(0, 1).astype(float))
    avg_vol_per_min = avg_volume_20d / 390 if avg_volume_20d > 0 else 1
    rel_vol = v / avg_vol_per_min

    vol_ema5 = _ema(v.astype(float), 5).replace(0, np.nan).fillna(1)
    vol_ema10 = _ema(v.astype(float), 10).replace(0, np.nan).fillna(1)
    vol_ema20 = _ema(v.astype(float), 20).replace(0, np.nan).fillna(1)
    vol_ema5_ratio = v / vol_ema5
    vol_ema10_ratio = v / vol_ema10
    vol_ema20_ratio = v / vol_ema20

    # Cumulative volume ratio: actual cumulative / expected cumulative
    cum_vol = v.cumsum().astype(float)
    bar_idx = pd.Series(np.arange(1, n + 1), dtype=float)
    expected_cum = bar_idx * avg_vol_per_min
    cum_vol_ratio = cum_vol / expected_cum.replace(0, 1)

    # VPT: cumulative sum of vol * pct_change
    pct_change = c.pct_change().fillna(0)
    vpt = (v * pct_change).cumsum()

    # OBV slope
    obv_slope = _obv_slope(c, v, window=5).fillna(0)

    # --- VWAP Dynamics (5) ---
    vwap_dist = (c - vwap_safe) / atr_safe
    vwap_slope = vwap_safe.diff(5) / 5
    vwap_slope = vwap_slope.fillna(0)

    # VWAP touches: count bars in last 10 where price crossed VWAP
    above_vwap = c >= vwap_safe
    cross = above_vwap.astype(int).diff().abs().fillna(0)
    vwap_touches = cross.rolling(10, min_periods=1).sum()

    # Bars since last VWAP cross
    cross_indices = cross.values
    bars_since_cross = pd.Series(np.zeros(n), dtype=float)
    last_cross = -1
    for i in range(n):
        if cross_indices[i] > 0:
            last_cross = i
        bars_since_cross.iloc[i] = (i - last_cross) if last_cross >= 0 else float(i)

    # VWAP reclaim flag: crossed above from below (was below, now above)
    prev_above = above_vwap.shift(1).fillna(False)
    reclaim = (~prev_above & above_vwap).astype(float)

    # --- Momentum (8) ---
    rsi_14 = _rsi(c, 14)
    macd_line, _, macd_hist = _macd(c)
    macd_hist_accel = macd_hist.diff().fillna(0)
    roc_5 = c.pct_change(5).fillna(0) * 100
    roc_10 = c.pct_change(10).fillna(0) * 100
    stoch_k = _stochastic_k(c, h, l)
    will_r = _williams_r(c, h, l)

    # --- Market Context (4) ---
    if spy_bars is not None and len(spy_bars) >= n:
        spy_c = spy_bars["close"].iloc[:n].reset_index(drop=True)
        spy_ret = spy_c.pct_change().fillna(0).cumsum() * 100
        spy_rsi_val = _rsi(spy_c, 14)
    else:
        spy_ret = pd.Series(np.zeros(n))
        spy_rsi_val = pd.Series(np.full(n, 50.0))

    sector_ret = pd.Series(np.full(n, sector_etf_return if sector_etf_return is not None else 0.0))
    vix_val = pd.Series(np.full(n, vix_level if vix_level is not None else 0.0))

    # --- Other (2) ---
    mfi = _mfi(h, l, c, v)

    # Consecutive green/red: positive = green streak, negative = red streak
    is_green = c > o
    consecutive = pd.Series(np.zeros(n), dtype=float)
    count = 0
    for i in range(n):
        if is_green.iloc[i]:
            count = max(count, 0) + 1
        else:
            count = min(count, 0) - 1
        consecutive.iloc[i] = count

    # --- Legacy Core Signal (3) ---
    legacy = _compute_legacy_candle_pressure_series(df, float_shares)

    # Assemble all 38 features
    features = pd.DataFrame({
        TEMPORAL_1MIN_FEATURE_NAMES[0]: open_vwap,
        TEMPORAL_1MIN_FEATURE_NAMES[1]: high_vwap,
        TEMPORAL_1MIN_FEATURE_NAMES[2]: low_vwap,
        TEMPORAL_1MIN_FEATURE_NAMES[3]: close_vwap,
        TEMPORAL_1MIN_FEATURE_NAMES[4]: atr_norm_range,
        TEMPORAL_1MIN_FEATURE_NAMES[5]: body_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[6]: upper_wick_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[7]: lower_wick_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[8]: log_vol,
        TEMPORAL_1MIN_FEATURE_NAMES[9]: rel_vol,
        TEMPORAL_1MIN_FEATURE_NAMES[10]: vol_ema5_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[11]: vol_ema10_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[12]: vol_ema20_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[13]: cum_vol_ratio,
        TEMPORAL_1MIN_FEATURE_NAMES[14]: vpt,
        TEMPORAL_1MIN_FEATURE_NAMES[15]: obv_slope,
        TEMPORAL_1MIN_FEATURE_NAMES[16]: vwap_dist,
        TEMPORAL_1MIN_FEATURE_NAMES[17]: vwap_slope,
        TEMPORAL_1MIN_FEATURE_NAMES[18]: vwap_touches,
        TEMPORAL_1MIN_FEATURE_NAMES[19]: bars_since_cross,
        TEMPORAL_1MIN_FEATURE_NAMES[20]: reclaim,
        TEMPORAL_1MIN_FEATURE_NAMES[21]: rsi_14,
        TEMPORAL_1MIN_FEATURE_NAMES[22]: macd_line,
        TEMPORAL_1MIN_FEATURE_NAMES[23]: macd_hist,
        TEMPORAL_1MIN_FEATURE_NAMES[24]: macd_hist_accel,
        TEMPORAL_1MIN_FEATURE_NAMES[25]: roc_5,
        TEMPORAL_1MIN_FEATURE_NAMES[26]: roc_10,
        TEMPORAL_1MIN_FEATURE_NAMES[27]: stoch_k,
        TEMPORAL_1MIN_FEATURE_NAMES[28]: will_r,
        TEMPORAL_1MIN_FEATURE_NAMES[29]: spy_ret,
        TEMPORAL_1MIN_FEATURE_NAMES[30]: spy_rsi_val,
        TEMPORAL_1MIN_FEATURE_NAMES[31]: sector_ret,
        TEMPORAL_1MIN_FEATURE_NAMES[32]: vix_val,
        TEMPORAL_1MIN_FEATURE_NAMES[33]: mfi,
        TEMPORAL_1MIN_FEATURE_NAMES[34]: consecutive,
        TEMPORAL_1MIN_FEATURE_NAMES[35]: legacy["weighted"],
        TEMPORAL_1MIN_FEATURE_NAMES[36]: legacy["unweighted"],
        TEMPORAL_1MIN_FEATURE_NAMES[37]: legacy["squared"],
    })

    # Fill any remaining NaN from warmup periods
    features = features.fillna(method="bfill").fillna(0)

    return features.values


# ---------------------------------------------------------------------------
# Temporal 5-min features (9)
# ---------------------------------------------------------------------------

def compute_temporal_features_5min(
    bars: pd.DataFrame,
    float_shares: float,
    avg_volume_5min_20d: float,
) -> np.ndarray:
    """Compute 9 temporal features per 5-min bar.

    Args:
        bars: DataFrame with columns [open, high, low, close, volume].
        float_shares: Outstanding float shares.
        avg_volume_5min_20d: 20-day average 5-min bar volume.

    Returns:
        np.ndarray of shape (n_bars, 9).
    """
    df = bars.copy().reset_index(drop=True)
    n = len(df)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Legacy candle pressure (3)
    legacy = _compute_legacy_candle_pressure_series(df, float_shares)

    # ATR
    atr = _atr(h, l, c)
    atr_safe = atr.replace(0, np.nan).fillna(method="bfill").fillna(0.01)

    # OHLC range normalized by ATR (1)
    ohlc_range = (h - l) / atr_safe

    # Relative volume (1)
    avg_safe = avg_volume_5min_20d if avg_volume_5min_20d > 0 else 1
    rel_vol = v / avg_safe

    # VWAP distance (1)
    vwap = _vwap(h, l, c, v)
    vwap_safe = vwap.replace(0, np.nan).fillna(method="bfill").fillna(c)
    vwap_dist = (c - vwap_safe) / atr_safe

    # RSI(14) (1)
    rsi_14 = _rsi(c, 14)

    # MACD histogram (1)
    _, _, macd_hist = _macd(c)

    # OBV slope (1)
    obv_slope = _obv_slope(c, v, window=5).fillna(0)

    features = pd.DataFrame({
        TEMPORAL_5MIN_FEATURE_NAMES[0]: legacy["weighted"],
        TEMPORAL_5MIN_FEATURE_NAMES[1]: legacy["unweighted"],
        TEMPORAL_5MIN_FEATURE_NAMES[2]: legacy["squared"],
        TEMPORAL_5MIN_FEATURE_NAMES[3]: ohlc_range,
        TEMPORAL_5MIN_FEATURE_NAMES[4]: rel_vol,
        TEMPORAL_5MIN_FEATURE_NAMES[5]: vwap_dist,
        TEMPORAL_5MIN_FEATURE_NAMES[6]: rsi_14,
        TEMPORAL_5MIN_FEATURE_NAMES[7]: macd_hist,
        TEMPORAL_5MIN_FEATURE_NAMES[8]: obv_slope,
    })

    features = features.fillna(method="bfill").fillna(0)

    return features.values


# ---------------------------------------------------------------------------
# Static features (8: 7 continuous + 1 categorical)
# ---------------------------------------------------------------------------

def compute_static_features(
    float_shares: float,
    short_interest_ratio: float,
    gap_percentage: float,
    sector_id: int,
    days_since_earnings: float,
    high_52wk: float,
    low_52wk: float,
    premarket_high: float,
    premarket_low: float,
    current_price: float,
    prior_close: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute static features.

    Returns:
        Tuple of (continuous_features [7], categorical_features [1]).
    """
    # 1. float_shares (log-scaled)
    float_log = np.log(max(float_shares, 1))

    # 2. short_interest_ratio (raw)
    si_ratio = short_interest_ratio

    # 3. gap_percentage (raw)
    gap_pct = gap_percentage

    # 4. days_since_earnings (raw)
    days_earn = float(days_since_earnings)

    # 5. 52wk high ratio: 1 - close/high52
    high_52_safe = high_52wk if high_52wk != 0 else 1e-8
    high_52wk_ratio = 1 - current_price / high_52_safe

    # 6. 52wk low ratio: 1 - low52/close
    close_safe = current_price if current_price != 0 else 1e-8
    low_52wk_ratio = 1 - low_52wk / close_safe

    # 7. premarket range: high_ratio + low_ratio
    prior_safe = prior_close if prior_close != 0 else 1e-8
    pm_high_ratio = premarket_high / prior_safe - 1
    pm_low_ratio = premarket_low / prior_safe - 1
    premarket_range = pm_high_ratio + pm_low_ratio

    continuous = np.array([
        float_log,
        si_ratio,
        gap_pct,
        days_earn,
        high_52wk_ratio,
        low_52wk_ratio,
        premarket_range,
    ], dtype=np.float64)

    categorical = np.array([sector_id], dtype=np.int64)

    return continuous, categorical


# ---------------------------------------------------------------------------
# Full feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    bars_1min: pd.DataFrame,
    bars_5min: pd.DataFrame,
    static_data: dict,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the complete feature matrix for one stock/session.

    Args:
        bars_1min: 1-minute OHLCV bars (needs >= sequence_length + warmup).
        bars_5min: 5-minute OHLCV bars.
        static_data: Dict with static feature inputs.
        sequence_length: Number of trailing bars to return.

    Returns:
        Tuple of:
          - temporal: (sequence_length, 47) — 38 1-min + 9 5-min features
          - static_continuous: (7,)
          - static_categorical: (1,)
    """
    # Compute 1-min temporal features (38)
    features_1min = compute_temporal_features_1min(
        bars=bars_1min,
        float_shares=static_data["float_shares"],
        avg_volume_20d=static_data["avg_daily_volume_20d"],
        spy_bars=static_data.get("spy_bars"),
        vix_level=static_data.get("vix_level"),
        sector_etf_return=static_data.get("sector_etf_return"),
    )

    # Compute 5-min temporal features (9)
    avg_5min_vol = static_data["avg_daily_volume_20d"] / 78  # 78 five-min bars per session
    features_5min = compute_temporal_features_5min(
        bars=bars_5min,
        float_shares=static_data["float_shares"],
        avg_volume_5min_20d=avg_5min_vol,
    )

    # Align 5-min features to 1-min: each 5-min bar covers 5 1-min bars
    # Forward-fill: repeat each 5-min feature row for 5 1-min bars
    n_1min = len(bars_1min)
    n_5min = len(bars_5min)

    aligned_5min = np.zeros((n_1min, 9))
    for i in range(n_1min):
        # Map 1-min index to 5-min index (every 5 bars)
        idx_5min = min(i // 5, n_5min - 1)
        aligned_5min[i] = features_5min[idx_5min]

    # Concatenate: 38 1-min + 9 5-min = 47 temporal features
    temporal = np.concatenate([features_1min, aligned_5min], axis=1)

    # Take last sequence_length bars
    if temporal.shape[0] > sequence_length:
        temporal = temporal[-sequence_length:]

    # Static features
    static_cont, static_cat = compute_static_features(**static_data)

    return temporal, static_cont, static_cat
