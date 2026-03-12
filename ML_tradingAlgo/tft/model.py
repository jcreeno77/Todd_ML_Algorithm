import torch
import torch.nn as nn
from .layers import GatedResidualNetwork, GatedLinearUnit
from .variable_selection import VariableSelectionNetwork
from .attention import InterpretableMultiHeadAttention


class TemporalFusionTransformer(nn.Module):
    """Encoder-only TFT for single-step dual-head prediction.

    Architecture:
        Static embedding → Static VSN → context vectors
        Temporal VSN (conditioned on static context) → selected features
        LSTM encoder → temporal states
        Static enrichment via GRN
        Interpretable Multi-Head Attention
        Output heads: P(win) sigmoid + entry offset linear
    """

    def __init__(
        self,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
        num_temporal_features: int,
        num_static_continuous: int,
        num_static_categorical: int,
        categorical_cardinalities: list[int],
        categorical_embedding_dim: int,
        sequence_length: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # --- Static input processing ---
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, categorical_embedding_dim)
            for card in categorical_cardinalities
        ])
        static_input_size = num_static_continuous + num_static_categorical * categorical_embedding_dim
        num_static_features = num_static_continuous + num_static_categorical

        self.static_vsn = VariableSelectionNetwork(
            input_size=static_input_size,
            num_features=num_static_features,
            hidden_size=hidden_size,
            dropout=dropout,
            feature_sizes=[1] * num_static_continuous + [categorical_embedding_dim] * num_static_categorical,
        )

        # Static context GRNs (4 context vectors)
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_state_h = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_state_c = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_vsn = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )

        # --- Temporal ---
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=num_temporal_features,
            num_features=num_temporal_features,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.post_lstm_fc = nn.Linear(hidden_size, hidden_size * 2)
        self.post_lstm_glu = GatedLinearUnit()
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        # --- Static enrichment ---
        self.enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size,
            dropout=dropout, context_size=hidden_size,
        )

        # --- Attention ---
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size, num_heads=attention_heads, dropout=dropout
        )
        self.post_attn_fc = nn.Linear(hidden_size, hidden_size * 2)
        self.post_attn_glu = GatedLinearUnit()
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # --- Output ---
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        # Output heads: GRN -> Dense 128 -> 64 -> 1 (matches spec)
        self.fc_win = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.fc_offset = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._last_static_weights = None
        self._last_temporal_weights = None

    def forward(
        self,
        temporal: torch.Tensor,
        static_continuous: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = temporal.shape[0]

        # --- Static processing ---
        cat_embeds = [emb(static_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        static_input = torch.cat([static_continuous] + cat_embeds, dim=-1)

        static_selected, static_weights = self.static_vsn(static_input)
        self._last_static_weights = static_weights.detach()

        cs_enrichment = self.static_context_enrichment(static_selected)
        cs_h = self.static_context_state_h(static_selected)
        cs_c = self.static_context_state_c(static_selected)
        cs_vsn = self.static_context_vsn(static_selected)

        # --- Temporal processing ---
        temporal_selected, temporal_weights = self.temporal_vsn(temporal, cs_vsn)
        self._last_temporal_weights = temporal_weights.detach()

        h0 = cs_h.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = cs_c.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        lstm_out, _ = self.lstm(temporal_selected, (h0, c0))

        lstm_gated = self.post_lstm_glu(self.post_lstm_fc(lstm_out))
        temporal_features = self.post_lstm_norm(lstm_gated + temporal_selected)

        # --- Static enrichment ---
        enriched = self.enrichment_grn(temporal_features, cs_enrichment)

        # --- Attention ---
        attn_out, attn_weights = self.attention(enriched)
        attn_gated = self.post_attn_glu(self.post_attn_fc(attn_out))
        temporal_output = self.post_attn_norm(attn_gated + enriched)

        # --- Output: use last timestep ---
        final = temporal_output[:, -1, :]
        output_processed = self.output_grn(final)

        p_win = self.fc_win(output_processed)
        entry_offset = self.fc_offset(output_processed)

        return p_win, entry_offset, attn_weights

    def get_static_selection_weights(self) -> torch.Tensor | None:
        return self._last_static_weights

    def get_temporal_selection_weights(self) -> torch.Tensor | None:
        return self._last_temporal_weights
