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
        assert features.shape[1] == 40, f"Expected 40 1-min features, got {features.shape[1]}"
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 40

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
        assert continuous.shape == (9,)
        assert categorical.shape == (1,)
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 9

    def test_float_is_log_scaled(self, sample_static_data):
        continuous, _ = compute_static_features(**sample_static_data)
        expected = np.log(sample_static_data["float_shares"])
        assert abs(continuous[0] - expected) < 1e-6


class TestTimeFeatures:
    def test_tod_features_present_and_bounded(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=2_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        assert feats.shape[1] == len(TEMPORAL_1MIN_FEATURE_NAMES)
        i_sin = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_sin")
        i_cos = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_cos")
        assert np.all(np.abs(feats[:, i_sin]) <= 1.0 + 1e-6)
        assert np.all(np.abs(feats[:, i_cos]) <= 1.0 + 1e-6)

    def test_tod_uses_datetime_index_when_present(self):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        idx = pd.date_range("2026-05-29 09:30", periods=20, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({
            "open": np.linspace(5, 5.2, 20), "high": np.linspace(5.05, 5.25, 20),
            "low": np.linspace(4.95, 5.15, 20), "close": np.linspace(5, 5.2, 20),
            "volume": np.full(20, 10000),
        }, index=idx)
        feats = compute_temporal_features_1min(
            bars, float_shares=2_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i_sin = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_sin")
        # First bar at 09:30 -> minute 0 -> sin(0) == 0.
        assert abs(feats[0, i_sin]) < 1e-6

    def test_dow_features_present(self, sample_static_data):
        from ML_tradingAlgo.tft.features import (
            compute_static_features, STATIC_CONTINUOUS_FEATURE_NAMES,
        )
        data = dict(sample_static_data)
        data["session_date"] = "2026-05-29"  # a Friday (weekday 4)
        cont, cat = compute_static_features(**data)
        assert cont.shape[0] == len(STATIC_CONTINUOUS_FEATURE_NAMES)
        i_sin = STATIC_CONTINUOUS_FEATURE_NAMES.index("dow_sin")
        assert -1.0 - 1e-6 <= cont[i_sin] <= 1.0 + 1e-6


class TestBuildFeatureMatrix:
    def test_full_matrix_shape(self, sample_1min_bars, sample_5min_bars, sample_static_data):
        temporal, static_cont, static_cat = build_feature_matrix(
            bars_1min=sample_1min_bars,
            bars_5min=sample_5min_bars,
            static_data=sample_static_data,
            sequence_length=30,
        )
        assert temporal.shape == (30, 49), f"Expected (30, 49), got {temporal.shape}"
        assert static_cont.shape == (9,)
        assert static_cat.shape == (1,)
