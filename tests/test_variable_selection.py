import pytest
import torch
from ML_tradingAlgo.tft.variable_selection import VariableSelectionNetwork


class TestVariableSelectionNetwork:
    def test_output_shape(self):
        vsn = VariableSelectionNetwork(
            input_size=47, num_features=47, hidden_size=160, dropout=0.1
        )
        x = torch.randn(4, 30, 47)
        out, weights = vsn(x)
        assert out.shape == (4, 30, 160)
        assert weights.shape == (4, 30, 47)

    def test_selection_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(
            input_size=47, num_features=47, hidden_size=160, dropout=0.1
        )
        x = torch.randn(4, 30, 47)
        _, weights = vsn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_with_context(self):
        vsn = VariableSelectionNetwork(
            input_size=47, num_features=47, hidden_size=160, dropout=0.1, context_size=160
        )
        x = torch.randn(4, 30, 47)
        context = torch.randn(4, 160)
        out, weights = vsn(x, context)
        assert out.shape == (4, 30, 160)

    def test_static_input_2d(self):
        """VSN for static features: input is (batch, num_features)."""
        vsn = VariableSelectionNetwork(
            input_size=7, num_features=7, hidden_size=160, dropout=0.1
        )
        x = torch.randn(4, 7)
        out, weights = vsn(x)
        assert out.shape == (4, 160)
        assert weights.shape == (4, 7)
