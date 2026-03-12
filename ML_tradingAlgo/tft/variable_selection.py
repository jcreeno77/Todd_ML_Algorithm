import torch
import torch.nn as nn
from .layers import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Selects and weights input features using learned softmax weights.

    Each feature gets its own GRN for transformation, then a shared GRN
    produces softmax selection weights across all features.
    """

    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float,
        context_size: int | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # One GRN per feature to transform each to hidden_size
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
            )
            for _ in range(num_features)
        ])

        # Shared GRN that produces selection weights
        self.weight_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
            context_size=context_size,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, [seq_len,] num_features)
        is_temporal = x.dim() == 3

        # Compute selection weights from flattened input
        weights = self.softmax(self.weight_grn(x, context))  # (batch, [seq,] num_features)

        # Transform each feature through its own GRN
        # Split along feature dimension: each is (batch, [seq,] 1)
        feature_inputs = x.unsqueeze(-1).chunk(self.num_features, dim=-2)

        transformed = []
        for i, grn in enumerate(self.feature_grns):
            fi = feature_inputs[i].squeeze(-2)  # (batch, [seq,] 1)
            transformed.append(grn(fi))  # (batch, [seq,] hidden_size)

        # Stack: (batch, [seq,] num_features, hidden_size)
        transformed = torch.stack(transformed, dim=-2)

        # Apply selection weights: (batch, [seq,] num_features, 1) * (batch, [seq,] num_features, hidden_size)
        weights_expanded = weights.unsqueeze(-1)
        selected = (weights_expanded * transformed).sum(dim=-2)  # (batch, [seq,] hidden_size)

        return selected, weights
