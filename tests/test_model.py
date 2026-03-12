import pytest
import torch
from ML_tradingAlgo.tft.model import TemporalFusionTransformer


class TestTemporalFusionTransformer:
    def test_output_shapes(self, model_config, sample_batch):
        model = TemporalFusionTransformer(**model_config)
        p_win, entry_offset, attn_weights = model(
            sample_batch["temporal"],
            sample_batch["static_continuous"],
            sample_batch["static_categorical"],
        )
        assert p_win.shape == (4, 1), f"p_win shape: {p_win.shape}"
        assert entry_offset.shape == (4, 1), f"entry_offset shape: {entry_offset.shape}"
        assert attn_weights.shape == (4, 4, 30, 30), f"attn shape: {attn_weights.shape}"

    def test_p_win_bounded_zero_one(self, model_config, sample_batch):
        model = TemporalFusionTransformer(**model_config)
        p_win, _, _ = model(
            sample_batch["temporal"],
            sample_batch["static_continuous"],
            sample_batch["static_categorical"],
        )
        assert (p_win >= 0).all() and (p_win <= 1).all()

    def test_gradient_flow(self, model_config, sample_batch):
        """All parameters should receive gradients."""
        model = TemporalFusionTransformer(**model_config)
        p_win, entry_offset, _ = model(
            sample_batch["temporal"],
            sample_batch["static_continuous"],
            sample_batch["static_categorical"],
        )
        loss = p_win.mean() + entry_offset.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self, model_config, sample_batch):
        model = TemporalFusionTransformer(**model_config)
        model.eval()
        with torch.no_grad():
            out1 = model(sample_batch["temporal"], sample_batch["static_continuous"], sample_batch["static_categorical"])
            out2 = model(sample_batch["temporal"], sample_batch["static_continuous"], sample_batch["static_categorical"])
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_variable_selection_weights_accessible(self, model_config, sample_batch):
        """We need access to VSN weights for interpretability."""
        model = TemporalFusionTransformer(**model_config)
        model.eval()
        with torch.no_grad():
            model(sample_batch["temporal"], sample_batch["static_continuous"], sample_batch["static_categorical"])
        static_weights = model.get_static_selection_weights()
        temporal_weights = model.get_temporal_selection_weights()
        assert static_weights is not None
        assert temporal_weights is not None
