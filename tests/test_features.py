# tests/test_features.py
import pytest
import numpy as np
import pandas as pd
from ML_tradingAlgo.tft.features import (
    compute_legacy_candle_pressure,
    compute_temporal_features_1min,
    compute_temporal_features_5min,
    compute_static_features,
    build_feature_matrix,
    TEMPORAL_1MIN_FEATURE_NAMES,
    TEMPORAL_5MIN_FEATURE_NAMES,
    STATIC_CONTINUOUS_FEATURE_NAMES,
)


@pytest.fixture
def sample_1min_bars():
    """30 one-minute OHLCV bars."""
    np.random.seed(42)
    n = 60  # need 60 bars for indicator warmup
    base_price = 5.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.03),
        "low": prices - np.abs(np.random.randn(n) * 0.03),
        "close": prices + np.random.randn(n) * 0.02,
        "volume": np.random.randint(10000, 100000, n),
    })


@pytest.fixture
def sample_5min_bars():
    """12 five-minute OHLCV bars (covers 60 mins)."""
    np.random.seed(42)
    n = 12
    base_price = 5.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.05),
        "low": prices - np.abs(np.random.randn(n) * 0.05),
        "close": prices + np.random.randn(n) * 0.03,
        "volume": np.random.randint(50000, 500000, n),
    })


@pytest.fixture
def sample_static_data():
    return {
        "float_shares": 2_000_000,
        "short_interest_ratio": 0.15,
        "gap_percentage": 0.35,
        "sector_id": 3,
        "days_since_earnings": 45,
        "high_52wk": 8.0,
        "low_52wk": 1.5,
        "premarket_high": 6.0,
        "premarket_low": 4.5,
        "current_price": 5.5,
        "prior_close": 4.0,
        "spy_bars": None,
        "vix_level": 20.0,
        "sector_etf_return": 0.01,
        "avg_daily_volume_20d": 500_000,
    }


class TestLegacyCandlePressure:
    def test_scaling_preserved(self):
        """The * 1000 scaling factor must be present."""
        feature = compute_legacy_candle_pressure(
            close=5.1, low=4.9, high=5.2, open_=5.0, volume=100000, float_shares=2000000
        )
        expected_unweighted = (((5.1 - 4.9) - (5.2 - 5.1)) / 5.0) * 1000
        expected_weighted = expected_unweighted * (100000 / 2000000 * 100)
        assert abs(feature["weighted"] - expected_weighted) < 1e-6
        assert abs(feature["unweighted"] - expected_unweighted) < 1e-6
        assert abs(feature["squared"] - expected_weighted ** 2) < 1e-6


class TestTemporalFeatures:
    def test_1min_feature_count(self, sample_1min_bars, sample_static_data):
        features = compute_temporal_features_1min(
            bars=sample_1min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_20d=sample_static_data["avg_daily_volume_20d"],
            spy_bars=sample_static_data["spy_bars"],
            vix_level=sample_static_data["vix_level"],
            sector_etf_return=sample_static_data["sector_etf_return"],
        )
        assert features.shape[1] == 38, f"Expected 38 1-min features, got {features.shape[1]}"
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 38

    def test_5min_feature_count(self, sample_5min_bars, sample_static_data):
        features = compute_temporal_features_5min(
            bars=sample_5min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_5min_20d=sample_static_data["avg_daily_volume_20d"] / 78,
        )
        assert features.shape[1] == 9, f"Expected 9 5-min features, got {features.shape[1]}"
        assert len(TEMPORAL_5MIN_FEATURE_NAMES) == 9

    def test_no_nan_in_output(self, sample_1min_bars, sample_static_data):
        features = compute_temporal_features_1min(
            bars=sample_1min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_20d=sample_static_data["avg_daily_volume_20d"],
            spy_bars=sample_static_data["spy_bars"],
            vix_level=sample_static_data["vix_level"],
            sector_etf_return=sample_static_data["sector_etf_return"],
        )
        assert not np.isnan(features[-30:]).any(), "NaN found in last 30 bars of features"


class TestStaticFeatures:
    def test_static_feature_count(self, sample_static_data):
        continuous, categorical = compute_static_features(**sample_static_data)
        assert continuous.shape == (7,)
        assert categorical.shape == (1,)
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 7

    def test_float_is_log_scaled(self, sample_static_data):
        continuous, _ = compute_static_features(**sample_static_data)
        expected = np.log(sample_static_data["float_shares"])
        assert abs(continuous[0] - expected) < 1e-6


class TestBuildFeatureMatrix:
    def test_full_matrix_shape(self, sample_1min_bars, sample_5min_bars, sample_static_data):
        temporal, static_cont, static_cat = build_feature_matrix(
            bars_1min=sample_1min_bars,
            bars_5min=sample_5min_bars,
            static_data=sample_static_data,
            sequence_length=30,
        )
        assert temporal.shape == (30, 47), f"Expected (30, 47), got {temporal.shape}"
        assert static_cont.shape == (7,)
        assert static_cat.shape == (1,)
