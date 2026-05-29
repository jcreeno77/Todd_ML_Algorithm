import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bars():
    idx = pd.date_range("2026-05-29 09:30", periods=40, freq="1min", tz="US/Eastern")
    base = 5.0 + np.cumsum(np.random.RandomState(0).randn(40) * 0.02)
    return pd.DataFrame({
        "open": base, "high": base + 0.03, "low": base - 0.03,
        "close": base + 0.01, "volume": np.full(40, 10000.0),
    }, index=idx)


def _ohlc_valid(df):
    return (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all() and \
           (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()


class TestBarTransforms:
    def test_jitter_preserves_shape_index_and_ohlc(self, bars):
        from ML_tradingAlgo.tft.augment import jitter_bars
        out = jitter_bars(bars, sigma=0.01, rng=np.random.RandomState(1))
        assert list(out.index) == list(bars.index)
        assert out.shape == bars.shape
        assert _ohlc_valid(out)
        assert not np.allclose(out["close"].to_numpy(), bars["close"].to_numpy())

    def test_window_slice_preserves_length(self, bars):
        from ML_tradingAlgo.tft.augment import window_slice_bars
        out = window_slice_bars(bars, frac=0.6, rng=np.random.RandomState(2))
        assert len(out) == len(bars)
        assert list(out.index) == list(bars.index)
        assert _ohlc_valid(out)

    def test_time_warp_preserves_length(self, bars):
        from ML_tradingAlgo.tft.augment import time_warp_bars
        out = time_warp_bars(bars, n_knots=4, sigma=0.2, rng=np.random.RandomState(3))
        assert len(out) == len(bars)
        assert list(out.index) == list(bars.index)
        assert _ohlc_valid(out)

    def test_no_time_reverse_helper_exists(self):
        # Guard: reversal is empirically harmful; it must NOT be offered.
        import ML_tradingAlgo.tft.augment as aug
        assert not hasattr(aug, "reverse_bars")


class TestMixup:
    def _batch(self):
        import torch
        return {
            "temporal": torch.randn(8, 30, 69),
            "static_continuous": torch.randn(8, 11),
            "static_categorical": torch.randint(0, 11, (8, 1)),
            "y_win": torch.randint(0, 2, (8, 1)).float(),
            "y_offset": torch.randn(8, 1),
            "weight": torch.ones(8, 1),
        }

    def test_mixup_preserves_shapes_and_categorical_dtype(self):
        from ML_tradingAlgo.tft.augment import mixup_batch
        import torch
        out = mixup_batch(self._batch(), alpha=0.3, rng=np.random.RandomState(0))
        assert out["temporal"].shape == (8, 30, 69)
        assert out["static_continuous"].shape == (8, 11)
        assert out["static_categorical"].dtype == torch.int64
        assert out["y_win"].shape == (8, 1)

    def test_mixup_alpha_zero_is_identity(self):
        from ML_tradingAlgo.tft.augment import mixup_batch
        import torch
        b = self._batch()
        out = mixup_batch(b, alpha=0.0, rng=np.random.RandomState(0))
        assert torch.allclose(out["temporal"], b["temporal"])
        assert torch.allclose(out["y_win"], b["y_win"])
