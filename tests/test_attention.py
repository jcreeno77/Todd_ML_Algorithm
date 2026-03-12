import pytest
import torch
from ML_tradingAlgo.tft.attention import InterpretableMultiHeadAttention


class TestInterpretableMultiHeadAttention:
    def test_output_shape(self):
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.1
        )
        x = torch.randn(4, 30, 160)
        out, weights = attn(x)
        assert out.shape == (4, 30, 160)

    def test_attention_weights_shape(self):
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.1
        )
        x = torch.randn(4, 30, 160)
        _, weights = attn(x)
        assert weights.shape == (4, 4, 30, 30)

    def test_attention_weights_sum_to_one(self):
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.0
        )
        x = torch.randn(4, 30, 160)
        _, weights = attn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_shared_values_across_heads(self):
        """All heads should share the same value projection."""
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.1
        )
        value_params = [name for name, _ in attn.named_parameters() if 'value' in name.lower() or 'v_proj' in name.lower()]
        assert len(value_params) == 2, f"Expected 1 value proj (weight+bias), got params: {value_params}"
