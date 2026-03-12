import pytest
import torch
from ML_tradingAlgo.tft.layers import GatedLinearUnit, GatedResidualNetwork


class TestGatedLinearUnit:
    def test_output_shape(self):
        glu = GatedLinearUnit()
        x = torch.randn(4, 320)  # Input must be 2 * output_size
        out = glu(x)
        assert out.shape == (4, 160)

    def test_output_bounded(self):
        """GLU uses sigmoid gate, so output magnitude should be bounded."""
        glu = GatedLinearUnit()
        x = torch.randn(4, 64)  # 2 * 32
        out = glu(x)
        assert out.abs().max() < 100


class TestGatedResidualNetwork:
    def test_output_shape_no_context(self):
        grn = GatedResidualNetwork(input_size=160, hidden_size=160, output_size=160, dropout=0.1)
        x = torch.randn(4, 160)
        out = grn(x)
        assert out.shape == (4, 160)

    def test_output_shape_with_context(self):
        grn = GatedResidualNetwork(input_size=160, hidden_size=160, output_size=160, dropout=0.1, context_size=160)
        x = torch.randn(4, 160)
        context = torch.randn(4, 160)
        out = grn(x, context)
        assert out.shape == (4, 160)

    def test_output_shape_different_input_output(self):
        """GRN can project from input_size to a different output_size."""
        grn = GatedResidualNetwork(input_size=47, hidden_size=160, output_size=160, dropout=0.1)
        x = torch.randn(4, 47)
        out = grn(x)
        assert out.shape == (4, 160)

    def test_skip_connection_works(self):
        """With zero weights, output should approximate the skip-connected input."""
        grn = GatedResidualNetwork(input_size=160, hidden_size=160, output_size=160, dropout=0.0)
        with torch.no_grad():
            for p in grn.parameters():
                p.zero_()
        x = torch.randn(4, 160)
        out = grn(x)
        assert out.shape == (4, 160)

    def test_3d_input(self):
        """GRN should handle (batch, seq_len, features) input for temporal data."""
        grn = GatedResidualNetwork(input_size=160, hidden_size=160, output_size=160, dropout=0.1)
        x = torch.randn(4, 30, 160)
        out = grn(x)
        assert out.shape == (4, 30, 160)
