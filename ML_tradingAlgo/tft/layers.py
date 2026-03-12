import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU gate: splits input in half, applies sigmoid to one half, multiplies."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: Linear → ELU → Linear → GLU → skip connection + LayerNorm.

    Optionally accepts a context vector for static enrichment.
    Handles both 2D (batch, features) and 3D (batch, seq, features) inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        context_size: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Optional context projection
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

        # Skip connection projection if dimensions differ
        self.skip_proj = nn.Linear(input_size, output_size, bias=False) if input_size != output_size else None

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # Skip connection
        residual = self.skip_proj(x) if self.skip_proj else x

        # Main path
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            # Broadcast context across sequence dimension if needed
            if hidden.dim() == 3 and context.dim() == 2:
                context = context.unsqueeze(1).expand_as(hidden)
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        # Skip connection + LayerNorm
        return self.layer_norm(hidden + residual)
