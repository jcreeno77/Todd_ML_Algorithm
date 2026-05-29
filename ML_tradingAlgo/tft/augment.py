"""Bar-level data augmentation for the TFT momentum pipeline.

Transforms operate on a 1-min OHLCV DataFrame and are **length- and
index-preserving** so that downstream feature engineering and triple-barrier
labeling can be re-run on the perturbed bars without changing the entry-bar
index. Reversal is deliberately NOT provided (empirically harmful for causal
momentum series). ``mixup_batch`` operates on already-batched tensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = ["open", "high", "low", "close"]


def _rng(rng):
    return rng if rng is not None else np.random.RandomState()


def _repair_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Re-impose high >= max(o,c) and low <= min(o,c)."""
    hi = df[["open", "close", "high"]].max(axis=1)
    lo = df[["open", "close", "low"]].min(axis=1)
    df["high"] = hi
    df["low"] = lo
    df["volume"] = df["volume"].clip(lower=0)
    return df


def jitter_bars(bars: pd.DataFrame, sigma: float = 0.01, rng=None) -> pd.DataFrame:
    """Multiplicative Gaussian noise on OHLC (and volume), per the financial
    augmentation literature (sigma ~ 0.01)."""
    rng = _rng(rng)
    out = bars.copy()
    noise = 1.0 + rng.randn(len(bars), len(_OHLC)) * sigma
    out[_OHLC] = bars[_OHLC].to_numpy() * noise
    vol_noise = 1.0 + rng.randn(len(bars)) * sigma
    out["volume"] = (bars["volume"].to_numpy() * np.clip(vol_noise, 0.0, None))
    return _repair_ohlc(out)


def window_slice_bars(bars: pd.DataFrame, frac: float = 0.6, rng=None) -> pd.DataFrame:
    """"Magnify": take a contiguous sub-window of length int(n*frac) and
    interpolate each column back to the original length."""
    rng = _rng(rng)
    n = len(bars)
    w = max(2, int(round(n * frac)))
    start = int(rng.randint(0, max(1, n - w + 1)))
    sub = bars.iloc[start:start + w]
    src = np.linspace(0.0, 1.0, len(sub))
    dst = np.linspace(0.0, 1.0, n)
    out = bars.copy()
    for col in _OHLC + ["volume"]:
        out[col] = np.interp(dst, src, sub[col].to_numpy(dtype=float))
    return _repair_ohlc(out)


def time_warp_bars(bars: pd.DataFrame, n_knots: int = 4, sigma: float = 0.2, rng=None) -> pd.DataFrame:
    """Monotonic time-axis warp via random knot offsets + linear resampling."""
    rng = _rng(rng)
    n = len(bars)
    knots = np.linspace(0.0, 1.0, n_knots + 2)
    offsets = np.concatenate([[0.0], rng.randn(n_knots) * sigma / n_knots, [0.0]])
    warped_knots = np.clip(knots + offsets, 0.0, 1.0)
    warped_knots = np.maximum.accumulate(warped_knots)  # enforce monotonicity
    base = np.linspace(0.0, 1.0, n)
    warped_time = np.interp(base, knots, warped_knots)
    out = bars.copy()
    for col in _OHLC + ["volume"]:
        out[col] = np.interp(warped_time, base, bars[col].to_numpy(dtype=float))
    return _repair_ohlc(out)


import torch  # noqa: E402  (kept local-friendly; torch only needed for mixup)


def mixup_batch(batch: dict, alpha: float = 0.3, rng=None) -> dict:
    """Mixup for the dual-head TFT.

    Interpolates ``temporal``, ``static_continuous``, ``y_win``, ``y_offset``
    and ``weight`` between the batch and a shuffled copy of itself using a
    single lambda ~ Beta(alpha, alpha). ``static_categorical`` is taken from the
    dominant sample (lambda >= 0.5 -> original order, else the shuffled order).
    ``alpha <= 0`` returns the batch unchanged.
    """
    if alpha is None or alpha <= 0:
        return batch
    rng = _rng(rng)
    lam = float(rng.beta(alpha, alpha))
    bsz = batch["temporal"].shape[0]
    perm = torch.as_tensor(rng.permutation(bsz), dtype=torch.long)

    out = dict(batch)
    for key in ("temporal", "static_continuous", "y_win", "y_offset", "weight"):
        if key in batch:
            out[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]
    if "static_categorical" in batch:
        out["static_categorical"] = (
            batch["static_categorical"] if lam >= 0.5 else batch["static_categorical"][perm]
        )
    return out
