# tests/test_dataset.py
import numpy as np
import torch

from ML_tradingAlgo.tft.dataset import compute_normalization_stats, TFTDataset


def _make_data(n=8, seed=0):
    """Build small synthetic arrays with the expected shapes."""
    rng = np.random.default_rng(seed)
    temporal = rng.normal(size=(n, 30, 47)).astype(np.float32)
    static_continuous = rng.normal(size=(n, 7)).astype(np.float32)
    static_categorical = rng.integers(0, 11, size=(n, 1)).astype(np.int64)
    y_win = rng.integers(0, 2, size=(n,)).astype(np.float32)
    y_offset = rng.normal(size=(n,)).astype(np.float32)
    sample_weights = rng.uniform(0.5, 2.0, size=(n,)).astype(np.float32)
    return temporal, static_continuous, static_categorical, y_win, y_offset, sample_weights


def test_compute_normalization_stats_keys_and_shapes():
    temporal, static_continuous, *_ = _make_data()
    stats = compute_normalization_stats(temporal, static_continuous)
    assert set(stats.keys()) == {
        "temporal_mean",
        "temporal_std",
        "static_mean",
        "static_std",
    }
    assert stats["temporal_mean"].shape == (47,)
    assert stats["temporal_std"].shape == (47,)
    assert stats["static_mean"].shape == (7,)
    assert stats["static_std"].shape == (7,)


def test_zero_variance_feature_std_clamped():
    temporal, static_continuous, *_ = _make_data()
    # Make feature column 3 constant -> zero variance.
    temporal[:, :, 3] = 5.0
    # Make static column 2 constant -> zero variance.
    static_continuous[:, 2] = -1.0
    stats = compute_normalization_stats(temporal, static_continuous)
    assert stats["temporal_std"][3] == 1e-8
    assert stats["static_std"][2] == 1e-8


def test_normalized_temporal_has_zero_mean_per_feature():
    temporal, static_continuous, static_categorical, y_win, y_offset, w = _make_data()
    stats = compute_normalization_stats(temporal, static_continuous)
    ds = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=stats,
    )
    stacked = torch.stack([ds[i]["temporal"] for i in range(len(ds))])  # (N,30,47)
    per_feature_mean = stacked.reshape(-1, 47).mean(dim=0)  # (47,)
    assert torch.all(per_feature_mean.abs() < 1e-4)


def test_no_normalization_returns_raw():
    temporal, static_continuous, static_categorical, y_win, y_offset, w = _make_data()
    ds = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=None,
    )
    out = ds[0]["temporal"].numpy()
    np.testing.assert_array_almost_equal(out, temporal[0], decimal=6)


def test_len_and_item_shapes_and_dtypes():
    temporal, static_continuous, static_categorical, y_win, y_offset, w = _make_data(n=8)
    ds = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=None,
    )
    assert len(ds) == 8
    item = ds[0]
    assert set(item.keys()) == {
        "temporal", "static_continuous", "static_categorical",
        "y_win", "y_offset", "weight",
    }
    assert item["temporal"].shape == (30, 47)
    assert item["static_continuous"].shape == (7,)
    assert item["static_categorical"].shape == (1,)
    assert item["y_win"].shape == (1,)
    assert item["y_offset"].shape == (1,)
    assert item["weight"].shape == (1,)
    assert item["temporal"].dtype == torch.float32
    assert item["static_continuous"].dtype == torch.float32
    assert item["static_categorical"].dtype == torch.int64
    assert item["y_win"].dtype == torch.float32
    assert item["y_offset"].dtype == torch.float32
    assert item["weight"].dtype == torch.float32


def test_categorical_identical_regardless_of_norm():
    temporal, static_continuous, static_categorical, y_win, y_offset, w = _make_data()
    stats = compute_normalization_stats(temporal, static_continuous)
    ds_norm = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=stats,
    )
    ds_raw = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=None,
    )
    for i in range(len(ds_norm)):
        a = ds_norm[i]["static_categorical"]
        b = ds_raw[i]["static_categorical"]
        assert torch.equal(a, b)
        assert a.item() == int(static_categorical[i, 0])


def test_augment_determinism_and_variation():
    temporal, static_continuous, static_categorical, y_win, y_offset, w = _make_data()
    stats = compute_normalization_stats(temporal, static_continuous)

    # augment=False -> deterministic across calls.
    ds_plain = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=stats, augment=False,
    )
    a1 = ds_plain[2]["temporal"]
    a2 = ds_plain[2]["temporal"]
    assert torch.equal(a1, a2)

    # augment=True with default cfg (feature_dropout=0.0) -> also deterministic
    # because the default augmentation is a no-op; noise and time-roll have been
    # removed; bar-level and mixup augmentation now happen upstream/in train loop.
    ds_aug = TFTDataset(
        temporal, static_continuous, static_categorical,
        y_win, y_offset, w, norm_stats=stats, augment=True,
    )
    torch.manual_seed(0)
    np.random.seed(0)
    b1 = ds_aug[2]["temporal"].clone()
    torch.manual_seed(1)
    np.random.seed(1)
    b2 = ds_aug[2]["temporal"].clone()
    assert torch.equal(b1, b2)


def test_dataset_no_longer_injects_noise_or_roll():
    t = np.random.RandomState(0).randn(3, 30, 69).astype("float32")
    sc = np.zeros((3, 11), "float32")
    cat = np.zeros((3, 1), "int64")
    ds = TFTDataset(t, sc, cat, y_win=np.zeros(3), y_offset=np.zeros(3),
                    sample_weights=np.ones(3), augment=True)
    item = ds[0]
    assert np.allclose(item["temporal"].numpy(), t[0])
