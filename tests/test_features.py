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
        assert features.shape[1] == 60, f"Expected 60 1-min features, got {features.shape[1]}"
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 60

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
        assert continuous.shape == (11,)
        assert categorical.shape == (1,)
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 11

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
        assert temporal.shape == (30, 69), f"Expected (30, 69), got {temporal.shape}"
        assert static_cont.shape == (11,)
        assert static_cat.shape == (1,)


class TestVolumeFeatures:
    def test_float_rotation_monotonic(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("float_rotation")
        col = feats[:, i]
        assert np.all(np.diff(col) >= -1e-6)  # cumulative -> non-decreasing
        assert col[-1] > 0

    def test_log_dollar_volume_present(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("log_dollar_volume")
        assert np.all(feats[:, i] > 0)

    def test_intraday_rvol_profile_and_fallback(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("intraday_rvol")
        # fallback path (no profile) must still produce finite values
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        assert np.all(np.isfinite(feats[:, i]))
        # profile path: a profile of all-1.0 expected volume -> rvol == volume
        prof = np.ones(390)
        feats2 = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            intraday_volume_profile=prof,
        )
        assert np.all(np.isfinite(feats2[:, i]))
        # all-1.0 expected volume -> rvol equals raw volume exactly
        assert np.allclose(feats2[:, i], sample_1min_bars["volume"].to_numpy())

    def test_build_intraday_volume_profile(self):
        from ML_tradingAlgo.tft.features import build_intraday_volume_profile
        idx = pd.date_range("2026-05-29 09:30", periods=10, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({"volume": np.arange(10) + 1.0}, index=idx)
        prof = build_intraday_volume_profile([bars])
        assert len(prof) == 390
        assert prof[0] == 1.0  # minute 0 -> first bar volume


class TestPriceLevelFeatures:
    NEW = [
        "pm_high_dist_atr", "pm_low_dist_atr", "broke_pm_high",
        "round_number_dist_atr", "or_high_dist_atr", "or_break_flag",
        "anchored_vwap_dist_atr", "prior_close_dist_atr",
        "prior_high_dist_atr", "gap_fill_progress", "ema_overextension_atr",
    ]

    def test_all_present_and_finite(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            premarket_high=5.3, premarket_low=4.7, prior_close=4.5,
            prior_day_high=5.0,
        )
        for name in self.NEW:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name

    def test_broke_pm_high_is_binary(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            premarket_high=5.0,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("broke_pm_high")
        assert set(np.unique(feats[:, i])).issubset({0.0, 1.0})

    def test_neutral_when_inputs_missing(self, sample_1min_bars):
        # No premarket/prior inputs -> features must still be finite (neutral).
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        for name in ["pm_high_dist_atr", "prior_close_dist_atr", "gap_fill_progress"]:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name

    def test_anchored_vwap_differs_from_cumulative_with_premarket(self):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        # 10 premarket bars (09:00-09:09) at a low price, then 20 regular-hours
        # bars (09:30+) at a higher price. Anchored VWAP must ignore premarket.
        pre = pd.date_range("2026-05-29 09:00", periods=10, freq="1min", tz="US/Eastern")
        reg = pd.date_range("2026-05-29 09:30", periods=20, freq="1min", tz="US/Eastern")
        idx = pre.append(reg)
        price = np.concatenate([np.full(10, 4.0), np.full(20, 6.0)])
        bars = pd.DataFrame({
            "open": price, "high": price + 0.05, "low": price - 0.05,
            "close": price, "volume": np.full(30, 10000.0),
        }, index=idx)
        feats = compute_temporal_features_1min(
            bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i_anc = TEMPORAL_1MIN_FEATURE_NAMES.index("anchored_vwap_dist_atr")
        i_cum = TEMPORAL_1MIN_FEATURE_NAMES.index("vwap_distance_atr")
        # On the last bar the two VWAPs must differ (anchored excludes the cheap
        # premarket bars, so its VWAP is higher and the distance smaller).
        assert not np.isclose(feats[-1, i_anc], feats[-1, i_cum])


class TestStructureFeatures:
    def test_structure_features_present(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        for name in ["pullback_depth_atr", "higher_low_count", "new_hod_flag",
                     "bars_since_hod", "price_accel", "volume_accel"]:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name

    def test_pullback_nonnegative_and_hod_binary(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i_pb = TEMPORAL_1MIN_FEATURE_NAMES.index("pullback_depth_atr")
        i_hod = TEMPORAL_1MIN_FEATURE_NAMES.index("new_hod_flag")
        assert np.all(feats[:, i_pb] >= -1e-6)
        assert set(np.unique(feats[:, i_hod])).issubset({0.0, 1.0})

    def test_total_1min_feature_count_is_60(self):
        from ML_tradingAlgo.tft.features import TEMPORAL_1MIN_FEATURE_NAMES
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 60

    def test_bars_since_hod_resets_and_increments(self):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        # Highs: rise for 3 bars (each a new HOD), then 3 bars below the HOD,
        # then a new HOD again. Use a DatetimeIndex so the function is happy.
        idx = pd.date_range("2026-05-29 09:30", periods=7, freq="1min", tz="US/Eastern")
        highs = np.array([10.0, 11.0, 12.0, 11.5, 11.0, 11.8, 13.0])
        bars = pd.DataFrame({
            "open": highs - 0.2, "high": highs, "low": highs - 0.5,
            "close": highs - 0.1, "volume": np.full(7, 10000.0),
        }, index=idx)
        feats = compute_temporal_features_1min(
            bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("bars_since_hod")
        # bars 0,1,2 are new HODs (->0), bar3 (+1), bar4 (+2), bar5 (+3),
        # bar6 is a new HOD (->0).
        np.testing.assert_array_equal(feats[:, i], [0, 0, 0, 1, 2, 3, 0])


class TestFeatureMatrixDims:
    def test_build_feature_matrix_shapes(self, sample_1min_bars, sample_5min_bars, sample_static_data):
        from ML_tradingAlgo.tft.features import build_feature_matrix
        data = dict(sample_static_data)
        data.update({"session_date": "2026-05-29", "prior_day_high": 6.0,
                     "prior_day_range": 1.5, "day_of_run": 1,
                     "intraday_volume_profile": None})
        temporal, cont, cat = build_feature_matrix(
            sample_1min_bars, sample_5min_bars, data, sequence_length=30,
        )
        assert temporal.shape == (30, 69)
        assert cont.shape == (11,)
        assert cat.shape == (1,)

    def test_model_forward_with_new_dims(self):
        import torch
        from ML_tradingAlgo.tft.model import TemporalFusionTransformer
        cfg = {
            "hidden_size": 32, "lstm_layers": 1, "attention_heads": 2,
            "dropout": 0.1, "num_temporal_features": 69,
            "num_static_continuous": 11, "num_static_categorical": 1,
            "categorical_cardinalities": [11], "categorical_embedding_dim": 8,
            "sequence_length": 30,
        }
        m = TemporalFusionTransformer(**cfg)
        p, off, attn = m(torch.randn(2, 30, 69), torch.randn(2, 11),
                         torch.randint(0, 11, (2, 1)))
        assert p.shape[0] == 2 and off.shape[0] == 2


class TestNewStaticFeatures:
    def test_static_count_is_11(self):
        from ML_tradingAlgo.tft.features import STATIC_CONTINUOUS_FEATURE_NAMES
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 11

    def test_day_of_run_and_gap_vs_range(self, sample_static_data):
        from ML_tradingAlgo.tft.features import (
            compute_static_features, STATIC_CONTINUOUS_FEATURE_NAMES,
        )
        data = dict(sample_static_data)
        data.update({"day_of_run": 2, "prior_day_range": 1.0,
                     "prior_day_high": 5.0, "session_date": "2026-05-29"})
        cont, _ = compute_static_features(**data)
        i_run = STATIC_CONTINUOUS_FEATURE_NAMES.index("day_of_run")
        i_gap = STATIC_CONTINUOUS_FEATURE_NAMES.index("gap_vs_prior_range")
        assert cont[i_run] == 2.0
        # gap_vs_prior_range = (current_price - prior_close)/prior_day_range
        # current_price=5.5, prior_close=4.0, prior_day_range=1.0 -> 1.5
        assert abs(cont[i_gap] - (5.5 - 4.0) / 1.0) < 1e-6
