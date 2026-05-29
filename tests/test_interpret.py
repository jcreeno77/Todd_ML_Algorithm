import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ML_tradingAlgo.tft.model import TemporalFusionTransformer
from ML_tradingAlgo.tft import interpret


def _build_model(model_config):
    return TemporalFusionTransformer(**model_config).eval()


def _build_loader(sample_batch):
    ds = TensorDataset(
        sample_batch["temporal"],
        sample_batch["static_continuous"],
        sample_batch["static_categorical"],
    )
    return DataLoader(ds, batch_size=2)


def test_feature_importance_length_and_sum(model_config, sample_batch):
    model = _build_model(model_config)
    loader = _build_loader(sample_batch)

    imp = interpret.feature_importance(model, loader)

    assert isinstance(imp, dict)
    assert len(imp) == 69
    values = list(imp.values())
    assert all(v >= 0 for v in values)
    # VSN selection weights are a softmax over features, so per-feature means
    # (averaged over batch + time) sum to ~1.
    assert abs(sum(values) - 1.0) < 1e-3


def test_feature_importance_uses_names(model_config, sample_batch):
    model = _build_model(model_config)
    loader = _build_loader(sample_batch)

    names = [f"f{i}" for i in range(69)]
    imp = interpret.feature_importance(model, loader, feature_names=names)
    assert set(imp.keys()) == set(names)


def test_static_feature_importance_length(model_config, sample_batch):
    model = _build_model(model_config)
    loader = _build_loader(sample_batch)

    imp = interpret.static_feature_importance(model, loader)
    assert isinstance(imp, dict)
    assert len(imp) == 11
    assert all(v >= 0 for v in imp.values())


def test_plot_feature_importance_writes_png(model_config, sample_batch, tmp_path):
    model = _build_model(model_config)
    loader = _build_loader(sample_batch)
    imp = interpret.feature_importance(model, loader)

    out = str(tmp_path / "fi.png")
    ret = interpret.plot_feature_importance(imp, out)
    assert ret == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_temporal_attention_writes_png(model_config, sample_batch, tmp_path):
    model = _build_model(model_config)
    with torch.no_grad():
        _, _, attn = model(
            sample_batch["temporal"],
            sample_batch["static_continuous"],
            sample_batch["static_categorical"],
        )
    # use a single sample -> (heads, 30, 30)
    attn0 = attn[0]

    out = str(tmp_path / "attn.png")
    ret = interpret.plot_temporal_attention(attn0, out)
    assert ret == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_temporal_attention_2d(model_config, sample_batch, tmp_path):
    out = str(tmp_path / "attn2d.png")
    attn = np.random.rand(30, 30)
    ret = interpret.plot_temporal_attention(attn, out)
    assert ret == out
    assert os.path.getsize(out) > 0


def test_compute_attention_drift_identical_is_zero():
    a = np.random.rand(30, 30)
    drift = interpret.compute_attention_drift(a, a)
    assert drift == 0.0 or abs(drift) < 1e-9


def test_compute_attention_drift_divergent_is_positive():
    recent = np.zeros((30,))
    recent[0] = 1.0
    historical = np.zeros((30,))
    historical[-1] = 1.0
    drift = interpret.compute_attention_drift(recent, historical)
    assert drift > 0
