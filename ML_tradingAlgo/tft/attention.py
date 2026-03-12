import torch
import torch.nn as nn
import math


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with shared value projection for interpretability.

    All heads share the same value weights (V), so each head's attention weights
    directly represent temporal importance — no mixing through different V projections.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Per-head Q and K projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)

        # SHARED value projection (single head_dim output, not full hidden_size)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(self.head_dim, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Q, K: (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # V: shared across heads — (batch, seq_len, head_dim)
        v = self.v_proj(x)

        # Attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to shared values
        v_expanded = v.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights_dropped, v_expanded)  # (batch, num_heads, seq_len, head_dim)

        # Average across heads
        attn_output = attn_output.mean(dim=1)  # (batch, seq_len, head_dim)

        # Project back to hidden_size
        output = self.out_proj(attn_output)  # (batch, seq_len, hidden_size)

        return output, attn_weights
