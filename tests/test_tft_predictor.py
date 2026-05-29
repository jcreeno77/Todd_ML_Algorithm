import numpy as np
import pytest
import torch

from ML_tradingAlgo.tft.model import TemporalFusionTransformer
from ML_tradingAlgo.tft_predictor import TFTPredictor


@pytest.fixture
def checkpoint_dir(tmp_path, model_config):
    """Build a TFT, save its state_dict + identity norm_stats to a dir."""
    model = TemporalFusionTransformer(**model_config)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(
        tmp_path / "norm_stats.npz",
        temporal_mean=np.zeros(69, dtype=np.float32),
        temporal_std=np.ones(69, dtype=np.float32),
        static_mean=np.zeros(11, dtype=np.float32),
        static_std=np.ones(11, dtype=np.float32),
    )
    return tmp_path


@pytest.fixture
def raw_inputs():
    rng = np.random.default_rng(0)
    sequence = rng.standard_normal((30, 69)).astype(np.float32)
    static_continuous = rng.standard_normal(11).astype(np.float32)
    static_categorical = np.array([3], dtype=np.int64)
    return sequence, static_continuous, static_categorical


def test_predict_returns_tuple_of_two_floats(checkpoint_dir, model_config, raw_inputs):
    predictor = TFTPredictor(checkpoint_dir, model_config)
    result = predictor.predict(*raw_inputs)
    assert isinstance(result, tuple)
    assert len(result) == 2
    p_win, entry_offset = result
    assert isinstance(p_win, float)
    assert isinstance(entry_offset, float)


def test_p_win_in_unit_interval(checkpoint_dir, model_config, raw_inputs):
    predictor = TFTPredictor(checkpoint_dir, model_config)
    p_win, _ = predictor.predict(*raw_inputs)
    assert 0.0 <= p_win <= 1.0


def test_attention_none_before_predict_then_shape(checkpoint_dir, model_config, raw_inputs):
    predictor = TFTPredictor(checkpoint_dir, model_config)
    assert predictor.get_last_attention_weights() is None
    predictor.predict(*raw_inputs)
    attn = predictor.get_last_attention_weights()
    assert isinstance(attn, np.ndarray)
    assert attn.shape[-1] == 30


def test_normalization_is_applied(tmp_path, model_config, raw_inputs):
    """Same weights, different norm_stats -> different p_win."""
    model = TemporalFusionTransformer(**model_config)
    state_dict = model.state_dict()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    torch.save(state_dict, dir_a / "model.pt")
    torch.save(state_dict, dir_b / "model.pt")

    # Identity stats.
    np.savez(
        dir_a / "norm_stats.npz",
        temporal_mean=np.zeros(69, dtype=np.float32),
        temporal_std=np.ones(69, dtype=np.float32),
        static_mean=np.zeros(11, dtype=np.float32),
        static_std=np.ones(11, dtype=np.float32),
    )
    # Shifted mean, scaled std.
    np.savez(
        dir_b / "norm_stats.npz",
        temporal_mean=np.full(69, 5.0, dtype=np.float32),
        temporal_std=np.full(69, 3.0, dtype=np.float32),
        static_mean=np.full(11, 2.0, dtype=np.float32),
        static_std=np.full(11, 4.0, dtype=np.float32),
    )

    pred_a = TFTPredictor(dir_a, model_config)
    pred_b = TFTPredictor(dir_b, model_config)

    p_win_a, _ = pred_a.predict(*raw_inputs)
    p_win_b, _ = pred_b.predict(*raw_inputs)

    assert p_win_a != p_win_b


def test_determinism(checkpoint_dir, model_config, raw_inputs):
    predictor = TFTPredictor(checkpoint_dir, model_config)
    out1 = predictor.predict(*raw_inputs)
    out2 = predictor.predict(*raw_inputs)
    assert out1 == out2


def test_predictor_accepts_69_11_and_rejects_mismatch(tmp_path):
    import numpy as np, torch
    from ML_tradingAlgo.tft.model import TemporalFusionTransformer
    from ML_tradingAlgo.tft_predictor import TFTPredictor

    cfg = {
        "hidden_size": 32, "lstm_layers": 1, "attention_heads": 2, "dropout": 0.1,
        "num_temporal_features": 69, "num_static_continuous": 11,
        "num_static_categorical": 1, "categorical_cardinalities": [11],
        "categorical_embedding_dim": 8, "sequence_length": 30,
    }
    torch.save(TemporalFusionTransformer(**cfg).state_dict(), tmp_path / "model.pt")
    np.savez(
        tmp_path / "norm_stats.npz",
        temporal_mean=np.zeros(69, "float32"), temporal_std=np.ones(69, "float32"),
        static_mean=np.zeros(11, "float32"), static_std=np.ones(11, "float32"),
    )
    pred = TFTPredictor(tmp_path, cfg)

    p_win, offset = pred.predict(
        np.zeros((30, 69), "float32"), np.zeros(11, "float32"), np.zeros(1, "int64")
    )
    assert 0.0 <= p_win <= 1.0

    # Stale 47-wide input must fail loudly, not silently mis-broadcast.
    with pytest.raises(ValueError):
        pred.predict(np.zeros((30, 47), "float32"), np.zeros(11, "float32"),
                     np.zeros(1, "int64"))
