# tests/test_train.py
import datetime

import numpy as np
import torch

from ML_tradingAlgo.tft.dataset import TFTDataset
from ML_tradingAlgo.tft.train import (
    TRAIN_CONFIG,
    compute_class_pos_weight,
    create_walk_forward_folds,
    evaluate,
    tft_loss,
    train_all_folds,
    train_fold,
)


def _session_dates(n, start=datetime.date(2024, 1, 1)):
    """n chronologically-spaced session dates (one per day)."""
    return [start + datetime.timedelta(days=i) for i in range(n)]


def _make_assembled(n=64, seed=0, val_static_shift=0.0):
    """Build a synthetic assembled-style dict.

    Samples are chronologically ordered by session_dates. If val_static_shift
    is set, the LAST quarter of samples (which become the val split for a
    single-train-fold setup) gets a different static_continuous mean.
    """
    rng = np.random.default_rng(seed)
    temporal = rng.normal(size=(n, 30, 69)).astype(np.float32)
    static_continuous = rng.normal(size=(n, 11)).astype(np.float32)
    static_categorical = rng.integers(0, 11, size=(n, 1)).astype(np.int64)
    y_win = rng.integers(0, 2, size=(n,)).astype(np.float32)
    y_offset = rng.normal(size=(n,)).astype(np.float32)
    sample_weights = np.ones(n, dtype=np.float32)
    if val_static_shift:
        # shift the chronologically-late samples (the val tail)
        tail = n // 2
        static_continuous[tail:] += val_static_shift
    return {
        "temporal": temporal,
        "static_continuous": static_continuous,
        "static_categorical": static_categorical,
        "y_win": y_win,
        "y_offset": y_offset,
        "sample_weights": sample_weights,
        "session_dates": _session_dates(n),
    }


# --- 1. walk-forward folds ---------------------------------------------------

def test_create_walk_forward_folds_no_leakage_and_growth():
    dates = _session_dates(100)
    folds = create_walk_forward_folds(dates, num_folds=4)
    assert len(folds) == 4

    dates_arr = np.array([d.toordinal() for d in dates])
    prev_train_size = 0
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # no future leakage: max train date <= min val date
        max_train = dates_arr[train_idx].max()
        min_val = dates_arr[val_idx].min()
        assert max_train <= min_val
        # train window grows monotonically
        assert len(train_idx) > prev_train_size
        prev_train_size = len(train_idx)


def test_create_walk_forward_folds_handles_unsorted_input():
    dates = _session_dates(40)
    # shuffle the input; function must sort chronologically internally
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(dates))
    shuffled = [dates[i] for i in perm]
    folds = create_walk_forward_folds(shuffled, num_folds=4)
    dates_arr = np.array([d.toordinal() for d in shuffled])
    for train_idx, val_idx in folds:
        assert dates_arr[train_idx].max() <= dates_arr[val_idx].min()


# --- 2. pos_weight -----------------------------------------------------------

def test_compute_class_pos_weight_all_positive_guarded():
    y = np.ones(20, dtype=np.float32)
    assert compute_class_pos_weight(y) == 1.0


def test_compute_class_pos_weight_balanced():
    y = np.array([0, 1] * 10, dtype=np.float32)
    assert abs(compute_class_pos_weight(y) - 1.0) < 1e-6


def test_compute_class_pos_weight_three_to_one():
    y = np.array([0] * 30 + [1] * 10, dtype=np.float32)
    assert abs(compute_class_pos_weight(y) - 3.0) < 1e-6


# --- 3. loss -----------------------------------------------------------------

def test_tft_loss_scalar_requires_grad():
    p_win = torch.full((8, 1), 0.5, requires_grad=True)
    entry_offset = torch.zeros((8, 1), requires_grad=True)
    y_win = torch.randint(0, 2, (8, 1)).float()
    y_offset = torch.randn((8, 1))
    weight = torch.ones((8, 1))
    loss = tft_loss(p_win, entry_offset, y_win, y_offset, weight, pos_weight=1.0)
    assert loss.dim() == 0
    assert loss.requires_grad


def test_tft_loss_decreases_with_sgd():
    torch.manual_seed(0)
    # trivially separable: a single learnable logit per sample toward its label
    y_win = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    y_offset = torch.tensor([[0.5], [-0.5], [0.5], [-0.5]])
    weight = torch.ones((4, 1))

    logit = torch.zeros((4, 1), requires_grad=True)
    offset = torch.zeros((4, 1), requires_grad=True)
    opt = torch.optim.SGD([logit, offset], lr=0.5)

    def loss_fn():
        p_win = torch.sigmoid(logit)
        return tft_loss(p_win, offset, y_win, y_offset, weight, pos_weight=1.0)

    first = loss_fn().item()
    for _ in range(50):
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        opt.step()
    last = loss_fn().item()
    assert last < first


# --- 4. train_fold smoke -----------------------------------------------------

def test_train_fold_smoke(model_config, tmp_path):
    assembled = _make_assembled(n=64, seed=2)
    n = 64
    split = int(n * 0.8)
    from ML_tradingAlgo.tft.dataset import compute_normalization_stats

    def slice_ds(sl, stats, augment):
        return TFTDataset(
            assembled["temporal"][sl],
            assembled["static_continuous"][sl],
            assembled["static_categorical"][sl],
            assembled["y_win"][sl],
            assembled["y_offset"][sl],
            assembled["sample_weights"][sl],
            norm_stats=stats,
            augment=augment,
        )

    stats = compute_normalization_stats(
        assembled["temporal"][:split], assembled["static_continuous"][:split]
    )
    train_ds = slice_ds(slice(0, split), stats, True)
    val_ds = slice_ds(slice(split, n), stats, False)

    metrics = train_fold(
        train_ds, val_ds, model_config, fold_idx=0, output_dir=str(tmp_path),
        max_epochs=3,
    )
    assert set(["win_rate", "profit_factor", "loss"]).issubset(metrics.keys())

    model_path = tmp_path / "fold_0" / "model.pt"
    npz_path = tmp_path / "fold_0" / "norm_stats.npz"
    assert model_path.exists()
    assert npz_path.exists()

    # model.pt must be a state_dict (plain dict of tensors), loadable into the model
    sd = torch.load(model_path, weights_only=True)
    assert isinstance(sd, dict)
    from ML_tradingAlgo.tft.model import TemporalFusionTransformer
    m = TemporalFusionTransformer(**model_config)
    m.load_state_dict(sd)

    loaded = np.load(npz_path)
    assert set(loaded.files) == {"temporal_mean", "temporal_std", "static_mean", "static_std"}


# --- 5. leakage guard: norm stats fit on TRAIN split only --------------------

def test_train_all_folds_norm_stats_train_only(model_config, tmp_path):
    # val tail has a wildly different static_continuous mean (shift +100)
    assembled = _make_assembled(n=80, seed=3, val_static_shift=100.0)
    metrics = train_all_folds(
        assembled, model_config, output_dir=str(tmp_path), num_folds=2,
        train_config={**TRAIN_CONFIG, "max_epochs": 2},
    )
    assert len(metrics) == 2

    # Load fold_0 stats; its train split excludes the shifted tail, so the
    # static_mean must be small (near 0), NOT pulled up toward +100.
    loaded = np.load(tmp_path / "fold_0" / "norm_stats.npz")
    static_mean = loaded["static_mean"]
    assert np.all(np.abs(static_mean) < 10.0), static_mean

    # Sanity: stats over train+val would be far larger than train-only.
    full_mean = assembled["static_continuous"].mean(axis=0)
    assert np.max(np.abs(full_mean)) > np.max(np.abs(static_mean))


# --- 6. evaluate: trivially-correct predictions -> win_rate 1.0 --------------

class _PerfectModel(torch.nn.Module):
    """Returns p_win = y_win passed via static_continuous[:, 0] as a hack-free
    deterministic signal. Instead we encode the label in temporal[:,0,0]."""

    def forward(self, temporal, static_continuous, static_categorical):
        # label is stored in temporal[:, 0, 0] as 0/1
        label = temporal[:, 0, 0:1]
        p_win = label.clamp(0, 1)  # exactly 0 or 1
        entry_offset = torch.zeros_like(p_win)
        attn = None
        return p_win, entry_offset, attn


def test_evaluate_perfect_predictions():
    from torch.utils.data import DataLoader

    n = 16
    rng = np.random.default_rng(5)
    y_win = rng.integers(0, 2, size=(n,)).astype(np.float32)
    temporal = np.zeros((n, 30, 69), dtype=np.float32)
    temporal[:, 0, 0] = y_win  # encode label so _PerfectModel reproduces it
    ds = TFTDataset(
        temporal,
        np.zeros((n, 11), dtype=np.float32),
        np.zeros((n, 1), dtype=np.int64),
        y_win,
        np.zeros((n,), dtype=np.float32),
        np.ones((n,), dtype=np.float32),
        norm_stats=None,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=8)
    metrics = evaluate(_PerfectModel(), loader)
    assert metrics["win_rate"] == 1.0


def test_split_augmented_train_only():
    from ML_tradingAlgo.tft.train import split_originals_and_augmented
    is_aug = np.array([False, True, True, False, True])
    parent = np.array([0, 0, 0, 3, 3])
    train_orig = np.array([0])
    val_orig = np.array([3])
    train_idx, val_idx = split_originals_and_augmented(train_orig, val_orig, is_aug, parent)
    assert set(train_idx.tolist()) == {0, 1, 2}
    assert set(val_idx.tolist()) == {3}
    assert not is_aug[val_idx].any()
