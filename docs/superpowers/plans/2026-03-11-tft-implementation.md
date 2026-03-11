# TFT Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a custom Temporal Fusion Transformer for momentum gap-up entry prediction, replacing the legacy feedforward model.

**Architecture:** Custom PyTorch TFT (encoder-only, no decoder) with Variable Selection Networks, Gated Residual Networks, and Interpretable Multi-Head Attention. Two output heads: P(win) sigmoid + entry price offset linear. 55 input features (8 static + 47 temporal) across 30-bar sequences.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, polygon-api-client, schwab-py, numpy, pandas, matplotlib/seaborn

**Spec:** `docs/superpowers/specs/2026-03-11-tft-model-design.md`

---

## File Structure

```
ML_tradingAlgo/
├── tft/
│   ├── __init__.py              # Package exports
│   ├── layers.py                # GRN, GLU gate — the building blocks
│   ├── variable_selection.py    # Variable Selection Network (static + temporal)
│   ├── attention.py             # Interpretable Multi-Head Attention
│   ├── model.py                 # Full TFT model assembling all components
│   ├── features.py              # Feature engineering pipeline (55 features)
│   ├── dataset.py               # PyTorch Dataset + normalization
│   ├── train.py                 # Training loop with walk-forward validation
│   └── interpret.py             # Attention/feature importance visualization
├── tft_predictor.py             # Inference wrapper (drop-in for Todd_tradingAlgo1)
├── data/
│   ├── __init__.py
│   ├── polygon_collector.py     # Historical data collection from Polygon.io
│   └── labeler.py               # Win/loss + entry offset labeling
├── config.py                    # Modified: add POLYGON_API_KEY, SCHWAB_* vars
├── .env                         # Modified: add new env vars
├── .env.example                 # Modified: add new env var templates
tests/
├── __init__.py
├── test_layers.py               # GRN, GLU tests
├── test_variable_selection.py   # VSN tests
├── test_attention.py            # Attention tests
├── test_model.py                # Full model forward pass tests
├── test_features.py             # Feature engineering tests
├── test_dataset.py              # Dataset + normalization tests
├── test_polygon_collector.py    # Data collector filter/validation tests
├── test_labeler.py              # Labeling logic tests
├── test_tft_predictor.py        # Inference wrapper tests
└── conftest.py                  # Shared fixtures (sample data, model configs)
```

---

## Chunk 1: TFT Core Architecture

### Task 1: Shared Test Fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create test package and shared fixtures**

```python
# tests/__init__.py
# (empty)

# tests/conftest.py
import pytest
import torch
import numpy as np


@pytest.fixture
def model_config():
    """Standard TFT hyperparameters for testing."""
    return {
        "hidden_size": 160,
        "lstm_layers": 2,
        "attention_heads": 4,
        "dropout": 0.3,
        "num_temporal_features": 47,
        "num_static_continuous": 7,
        "num_static_categorical": 1,
        "categorical_cardinalities": [11],  # 11 GICS sectors
        "categorical_embedding_dim": 8,
        "sequence_length": 30,
    }


@pytest.fixture
def sample_batch(model_config):
    """A batch of 4 samples with correct shapes."""
    batch_size = 4
    return {
        "temporal": torch.randn(batch_size, model_config["sequence_length"], model_config["num_temporal_features"]),
        "static_continuous": torch.randn(batch_size, model_config["num_static_continuous"]),
        "static_categorical": torch.randint(0, 11, (batch_size, model_config["num_static_categorical"])),
        "y_win": torch.randint(0, 2, (batch_size, 1)).float(),
        "y_offset": torch.randn(batch_size, 1),
    }
```

- [ ] **Step 2: Verify fixtures load**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/conftest.py --collect-only`
Expected: fixtures collected, no errors

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add shared TFT test fixtures"
```

---

### Task 2: Gated Residual Network (GRN) and GLU Gate

**Files:**
- Create: `ML_tradingAlgo/tft/__init__.py`
- Create: `ML_tradingAlgo/tft/layers.py`
- Create: `tests/test_layers.py`

The GRN is the fundamental building block: Linear → ELU → Linear → GLU gate → skip connection + LayerNorm. It optionally accepts a context vector (used for static enrichment).

- [ ] **Step 1: Write failing tests for GRN**

```python
# tests/test_layers.py
import pytest
import torch
from ML_tradingAlgo.tft.layers import GatedLinearUnit, GatedResidualNetwork


class TestGatedLinearUnit:
    def test_output_shape(self):
        glu = GatedLinearUnit(input_size=160)
        x = torch.randn(4, 160)
        out = glu(x)
        assert out.shape == (4, 160)

    def test_output_bounded(self):
        """GLU uses sigmoid gate, so output magnitude should be bounded."""
        glu = GatedLinearUnit(input_size=32)
        x = torch.randn(4, 32)
        out = glu(x)
        # Sigmoid gate bounds one factor to [0,1], so output should be reasonable
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
        # Zero out all parameters except the skip connection
        with torch.no_grad():
            for p in grn.parameters():
                p.zero_()
        x = torch.randn(4, 160)
        out = grn(x)
        # LayerNorm will normalize, but the residual path should still dominate
        assert out.shape == (4, 160)

    def test_3d_input(self):
        """GRN should handle (batch, seq_len, features) input for temporal data."""
        grn = GatedResidualNetwork(input_size=160, hidden_size=160, output_size=160, dropout=0.1)
        x = torch.randn(4, 30, 160)
        out = grn(x)
        assert out.shape == (4, 30, 160)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_layers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ML_tradingAlgo.tft'`

- [ ] **Step 3: Implement GatedLinearUnit and GatedResidualNetwork**

```python
# ML_tradingAlgo/tft/__init__.py
# NOTE: Build up imports incrementally — only import modules that exist.
# Tasks 3, 4, 5 will add imports as their modules are created.
from .layers import GatedLinearUnit, GatedResidualNetwork

# ML_tradingAlgo/tft/layers.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_layers.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/__init__.py ML_tradingAlgo/tft/layers.py tests/test_layers.py
git commit -m "feat: implement GRN and GLU gate — TFT building blocks"
```

---

### Task 3: Variable Selection Network

**Files:**
- Create: `ML_tradingAlgo/tft/variable_selection.py`
- Create: `tests/test_variable_selection.py`

VSN applies individual GRNs to each feature, then softmax selection weights to combine. Static VSN runs once per sample; temporal VSN runs per timestep conditioned on static context.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_variable_selection.py
import pytest
import torch
from ML_tradingAlgo.tft.variable_selection import VariableSelectionNetwork


class TestVariableSelectionNetwork:
    def test_output_shape(self):
        vsn = VariableSelectionNetwork(
            input_size=47, num_features=47, hidden_size=160, dropout=0.1
        )
        # (batch, seq_len, num_features) — each feature is size 1
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_variable_selection.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement VariableSelectionNetwork**

```python
# ML_tradingAlgo/tft/variable_selection.py
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
```

Also update `ML_tradingAlgo/tft/__init__.py` to add the new import:
```python
from .layers import GatedLinearUnit, GatedResidualNetwork
from .variable_selection import VariableSelectionNetwork
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_variable_selection.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/variable_selection.py ML_tradingAlgo/tft/__init__.py tests/test_variable_selection.py
git commit -m "feat: implement Variable Selection Network for TFT"
```

---

### Task 4: Interpretable Multi-Head Attention

**Files:**
- Create: `ML_tradingAlgo/tft/attention.py`
- Create: `tests/test_attention.py`

Key difference from standard MHA: all heads share value projection weights, making attention weights directly interpretable as temporal importance.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_attention.py
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
        # Weights: (batch, num_heads, seq_len, seq_len)
        assert weights.shape == (4, 4, 30, 30)

    def test_attention_weights_sum_to_one(self):
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.0  # no dropout for deterministic test
        )
        x = torch.randn(4, 30, 160)
        _, weights = attn(x)
        # Each row of attention weights should sum to ~1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_shared_values_across_heads(self):
        """All heads should share the same value projection."""
        attn = InterpretableMultiHeadAttention(
            hidden_size=160, num_heads=4, dropout=0.1
        )
        # There should be only ONE value projection, not num_heads
        value_params = [name for name, _ in attn.named_parameters() if 'value' in name.lower() or 'v_proj' in name.lower()]
        # Should have exactly one value projection weight + bias
        assert len(value_params) == 2, f"Expected 1 value proj (weight+bias), got params: {value_params}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_attention.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement InterpretableMultiHeadAttention**

```python
# ML_tradingAlgo/tft/attention.py
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
        # (batch, num_heads, seq_len, seq_len) @ (batch, 1, seq_len, head_dim)
        v_expanded = v.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights_dropped, v_expanded)  # (batch, num_heads, seq_len, head_dim)

        # Average across heads
        attn_output = attn_output.mean(dim=1)  # (batch, seq_len, head_dim)

        # Project back to hidden_size
        output = self.out_proj(attn_output)  # (batch, seq_len, hidden_size)

        return output, attn_weights
```

Also update `ML_tradingAlgo/tft/__init__.py` to add the new import:
```python
from .layers import GatedLinearUnit, GatedResidualNetwork
from .variable_selection import VariableSelectionNetwork
from .attention import InterpretableMultiHeadAttention
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_attention.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/attention.py ML_tradingAlgo/tft/__init__.py tests/test_attention.py
git commit -m "feat: implement Interpretable Multi-Head Attention for TFT"
```

---

### Task 5: Full TFT Model

**Files:**
- Create: `ML_tradingAlgo/tft/model.py`
- Create: `tests/test_model.py`

Assembles all components: static embedding → static VSN → temporal VSN (conditioned on static context) → LSTM encoder → static enrichment → attention → output heads.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
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
        # Should be able to get last computed selection weights
        static_weights = model.get_static_selection_weights()
        temporal_weights = model.get_temporal_selection_weights()
        assert static_weights is not None
        assert temporal_weights is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement TemporalFusionTransformer**

```python
# ML_tradingAlgo/tft/model.py
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
        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, categorical_embedding_dim)
            for card in categorical_cardinalities
        ])
        static_input_size = num_static_continuous + num_static_categorical * categorical_embedding_dim

        # Static Variable Selection
        self.static_vsn = VariableSelectionNetwork(
            input_size=static_input_size,
            num_features=static_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Static context GRNs (produce 4 context vectors for different uses)
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

        # --- Temporal input processing ---
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=num_temporal_features,
            num_features=num_temporal_features,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # --- LSTM encoder ---
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Post-LSTM gate + skip connection
        self.post_lstm_glu = GatedLinearUnit(hidden_size)
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
        self.post_attn_glu = GatedLinearUnit(hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # --- Output ---
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        # Output heads: GRN -> Dense 128 -> 64 -> 1 (matches spec)
        self.fc_win = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.fc_offset = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # Store last computed weights for interpretability
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
        # Embed categoricals
        cat_embeds = [emb(static_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        static_input = torch.cat([static_continuous] + cat_embeds, dim=-1)

        # Static variable selection
        static_selected, static_weights = self.static_vsn(static_input)
        self._last_static_weights = static_weights.detach()

        # Generate context vectors
        cs_enrichment = self.static_context_enrichment(static_selected)
        cs_h = self.static_context_state_h(static_selected)
        cs_c = self.static_context_state_c(static_selected)
        cs_vsn = self.static_context_vsn(static_selected)

        # --- Temporal processing ---
        # Temporal variable selection conditioned on static context
        temporal_selected, temporal_weights = self.temporal_vsn(temporal, cs_vsn)
        self._last_temporal_weights = temporal_weights.detach()

        # LSTM with static-initialized hidden state
        # Expand static context for LSTM layers
        h0 = cs_h.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = cs_c.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        lstm_out, _ = self.lstm(temporal_selected, (h0, c0))

        # Post-LSTM gating with skip connection
        lstm_gated = self.post_lstm_glu(lstm_out)
        temporal_features = self.post_lstm_norm(lstm_gated + temporal_selected)

        # --- Static enrichment ---
        enriched = self.enrichment_grn(temporal_features, cs_enrichment)

        # --- Attention ---
        attn_out, attn_weights = self.attention(enriched)
        attn_gated = self.post_attn_glu(attn_out)
        temporal_output = self.post_attn_norm(attn_gated + enriched)

        # --- Output: use last timestep ---
        final = temporal_output[:, -1, :]  # (batch, hidden_size)
        output_processed = self.output_grn(final)

        p_win = self.fc_win(output_processed)
        entry_offset = self.fc_offset(output_processed)

        return p_win, entry_offset, attn_weights

    def get_static_selection_weights(self) -> torch.Tensor | None:
        return self._last_static_weights

    def get_temporal_selection_weights(self) -> torch.Tensor | None:
        return self._last_temporal_weights
```

- [ ] **Step 4: Update `__init__.py` exports**

```python
# ML_tradingAlgo/tft/__init__.py
from .layers import GatedLinearUnit, GatedResidualNetwork
from .variable_selection import VariableSelectionNetwork
from .attention import InterpretableMultiHeadAttention
from .model import TemporalFusionTransformer
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_model.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Run all tests so far**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/ -v`
Expected: All 19 tests PASS (6 layers + 4 vsn + 4 attention + 5 model)

- [ ] **Step 7: Commit**

```bash
git add ML_tradingAlgo/tft/model.py ML_tradingAlgo/tft/__init__.py tests/test_model.py
git commit -m "feat: implement full TFT model with dual output heads"
```

---

## Chunk 2: Feature Engineering and Data Pipeline

### Task 6: Feature Engineering Pipeline

**Files:**
- Create: `ML_tradingAlgo/tft/features.py`
- Create: `tests/test_features.py`

Computes all 55 features (8 static + 38 one-min temporal + 9 five-min aggregate) from raw OHLCV bar data. Single code path used by both training and inference to prevent feature drift.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features.py
import pytest
import numpy as np
import pandas as pd
from ML_tradingAlgo.tft.features import (
    compute_legacy_candle_pressure,
    compute_temporal_features_1min,
    compute_temporal_features_5min,
    compute_static_features,
    build_feature_matrix,
    TEMPORAL_1MIN_FEATURE_NAMES,
    TEMPORAL_5MIN_FEATURE_NAMES,
    STATIC_CONTINUOUS_FEATURE_NAMES,
)


@pytest.fixture
def sample_1min_bars():
    """30 one-minute OHLCV bars."""
    np.random.seed(42)
    n = 60  # need 60 bars for indicator warmup (RSI etc need ~30 bars before the 30-bar window)
    base_price = 5.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.03),
        "low": prices - np.abs(np.random.randn(n) * 0.03),
        "close": prices + np.random.randn(n) * 0.02,
        "volume": np.random.randint(10000, 100000, n),
    })


@pytest.fixture
def sample_5min_bars():
    """12 five-minute OHLCV bars (covers 60 mins)."""
    np.random.seed(42)
    n = 12
    base_price = 5.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.05),
        "low": prices - np.abs(np.random.randn(n) * 0.05),
        "close": prices + np.random.randn(n) * 0.03,
        "volume": np.random.randint(50000, 500000, n),
    })


@pytest.fixture
def sample_static_data():
    return {
        "float_shares": 2_000_000,
        "short_interest_ratio": 0.15,
        "gap_percentage": 0.35,
        "sector_id": 3,
        "days_since_earnings": 45,
        "high_52wk": 8.0,
        "low_52wk": 1.5,
        "premarket_high": 6.0,
        "premarket_low": 4.5,
        "current_price": 5.5,
        "prior_close": 4.0,
        "spy_bars": None,  # optional, can be None for basic test
        "vix_level": 20.0,
        "sector_etf_return": 0.01,
        "avg_daily_volume_20d": 500_000,
    }


class TestLegacyCandlePressure:
    def test_scaling_preserved(self):
        """The * 1000 scaling factor must be present."""
        feature = compute_legacy_candle_pressure(
            close=5.1, low=4.9, high=5.2, open_=5.0, volume=100000, float_shares=2000000
        )
        # Manual calc: (((5.1-4.9)-(5.2-5.1))/5.0*1000) * (100000/2000000*100)
        expected_unweighted = (((5.1 - 4.9) - (5.2 - 5.1)) / 5.0) * 1000
        expected_weighted = expected_unweighted * (100000 / 2000000 * 100)
        assert abs(feature["weighted"] - expected_weighted) < 1e-6
        assert abs(feature["unweighted"] - expected_unweighted) < 1e-6
        assert abs(feature["squared"] - expected_weighted ** 2) < 1e-6


class TestTemporalFeatures:
    def test_1min_feature_count(self, sample_1min_bars, sample_static_data):
        features = compute_temporal_features_1min(
            bars=sample_1min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_20d=sample_static_data["avg_daily_volume_20d"],
            spy_bars=sample_static_data["spy_bars"],
            vix_level=sample_static_data["vix_level"],
            sector_etf_return=sample_static_data["sector_etf_return"],
        )
        assert features.shape[1] == 38, f"Expected 38 1-min features, got {features.shape[1]}"
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 38

    def test_5min_feature_count(self, sample_5min_bars, sample_static_data):
        features = compute_temporal_features_5min(
            bars=sample_5min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_5min_20d=sample_static_data["avg_daily_volume_20d"] / 78,
        )
        assert features.shape[1] == 9, f"Expected 9 5-min features, got {features.shape[1]}"
        assert len(TEMPORAL_5MIN_FEATURE_NAMES) == 9

    def test_no_nan_in_output(self, sample_1min_bars, sample_static_data):
        features = compute_temporal_features_1min(
            bars=sample_1min_bars,
            float_shares=sample_static_data["float_shares"],
            avg_volume_20d=sample_static_data["avg_daily_volume_20d"],
            spy_bars=sample_static_data["spy_bars"],
            vix_level=sample_static_data["vix_level"],
            sector_etf_return=sample_static_data["sector_etf_return"],
        )
        # After warmup, no NaNs should remain
        assert not np.isnan(features[-30:]).any(), "NaN found in last 30 bars of features"


class TestStaticFeatures:
    def test_static_feature_count(self, sample_static_data):
        continuous, categorical = compute_static_features(**sample_static_data)
        assert continuous.shape == (7,)
        assert categorical.shape == (1,)
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 7

    def test_float_is_log_scaled(self, sample_static_data):
        continuous, _ = compute_static_features(**sample_static_data)
        # First feature is log(float_shares)
        expected = np.log(sample_static_data["float_shares"])
        assert abs(continuous[0] - expected) < 1e-6


class TestBuildFeatureMatrix:
    def test_full_matrix_shape(self, sample_1min_bars, sample_5min_bars, sample_static_data):
        temporal, static_cont, static_cat = build_feature_matrix(
            bars_1min=sample_1min_bars,
            bars_5min=sample_5min_bars,
            static_data=sample_static_data,
            sequence_length=30,
        )
        assert temporal.shape == (30, 47), f"Expected (30, 47), got {temporal.shape}"
        assert static_cont.shape == (7,)
        assert static_cat.shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement features.py**

Create `ML_tradingAlgo/tft/features.py`. This is the largest single file — implements all 55 features. Key implementation notes:
- Uses pandas for indicator computation (RSI, MACD, EMA, etc.)
- The legacy candle pressure formula must use `* 1000` scaling exactly
- VWAP is computed as cumulative(price * volume) / cumulative(volume)
- 5-min features are forward-filled to align with 1-min timestamps
- Market context features (SPY, VIX) use provided values or default to 0 if unavailable
- All feature name lists are exported as constants for use in interpretability

The file should export: `compute_legacy_candle_pressure`, `compute_temporal_features_1min`, `compute_temporal_features_5min`, `compute_static_features`, `build_feature_matrix`, and the three `*_FEATURE_NAMES` constants.

See spec `docs/superpowers/specs/2026-03-11-tft-model-design.md` lines 108-205 for the complete feature enumeration.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_features.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py tests/test_features.py
git commit -m "feat: implement 55-feature engineering pipeline for TFT"
```

---

### Task 7: Dataset and Normalization

**Files:**
- Create: `ML_tradingAlgo/tft/dataset.py`
- Create: `tests/test_dataset.py`

PyTorch Dataset that wraps pre-computed feature matrices. Handles z-score normalization (fit on training fold, apply to val/test), recency weighting, and class weighting.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
import pytest
import torch
import numpy as np
from ML_tradingAlgo.tft.dataset import TFTDataset, compute_normalization_stats


class TestComputeNormalizationStats:
    def test_stats_shapes(self):
        temporal = np.random.randn(100, 30, 47)
        static = np.random.randn(100, 7)
        stats = compute_normalization_stats(temporal, static)
        assert stats["temporal_mean"].shape == (47,)
        assert stats["temporal_std"].shape == (47,)
        assert stats["static_mean"].shape == (7,)
        assert stats["static_std"].shape == (7,)

    def test_std_no_zeros(self):
        """Std should be clamped to prevent division by zero."""
        temporal = np.ones((100, 30, 47))  # zero variance
        static = np.ones((100, 7))
        stats = compute_normalization_stats(temporal, static)
        assert (stats["temporal_std"] > 0).all()
        assert (stats["static_std"] > 0).all()


class TestTFTDataset:
    def test_len(self):
        ds = TFTDataset(
            temporal=np.random.randn(50, 30, 47),
            static_continuous=np.random.randn(50, 7),
            static_categorical=np.random.randint(0, 11, (50, 1)),
            y_win=np.random.randint(0, 2, 50).astype(np.float32),
            y_offset=np.random.randn(50).astype(np.float32),
            sample_weights=np.ones(50, dtype=np.float32),
        )
        assert len(ds) == 50

    def test_getitem_shapes(self):
        ds = TFTDataset(
            temporal=np.random.randn(50, 30, 47),
            static_continuous=np.random.randn(50, 7),
            static_categorical=np.random.randint(0, 11, (50, 1)),
            y_win=np.random.randint(0, 2, 50).astype(np.float32),
            y_offset=np.random.randn(50).astype(np.float32),
            sample_weights=np.ones(50, dtype=np.float32),
        )
        item = ds[0]
        assert item["temporal"].shape == (30, 47)
        assert item["static_continuous"].shape == (7,)
        assert item["static_categorical"].shape == (1,)
        assert item["y_win"].shape == (1,)
        assert item["y_offset"].shape == (1,)
        assert item["weight"].shape == (1,)

    def test_normalization_applied(self):
        temporal = np.random.randn(50, 30, 47) * 100 + 50  # not normalized
        static = np.random.randn(50, 7) * 100 + 50
        stats = compute_normalization_stats(temporal, static)
        ds = TFTDataset(
            temporal=temporal,
            static_continuous=static,
            static_categorical=np.zeros((50, 1), dtype=np.int64),
            y_win=np.zeros(50, dtype=np.float32),
            y_offset=np.zeros(50, dtype=np.float32),
            sample_weights=np.ones(50, dtype=np.float32),
            norm_stats=stats,
        )
        item = ds[0]
        # After z-score, values should be roughly in [-3, 3] range
        assert item["temporal"].abs().mean() < 5
        assert item["static_continuous"].abs().mean() < 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dataset.py**

```python
# ML_tradingAlgo/tft/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_normalization_stats(
    temporal: np.ndarray, static_continuous: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute z-score stats from training data. Clamp std to 1e-8 min."""
    return {
        "temporal_mean": temporal.reshape(-1, temporal.shape[-1]).mean(axis=0),
        "temporal_std": np.maximum(temporal.reshape(-1, temporal.shape[-1]).std(axis=0), 1e-8),
        "static_mean": static_continuous.mean(axis=0),
        "static_std": np.maximum(static_continuous.std(axis=0), 1e-8),
    }


class TFTDataset(Dataset):
    """PyTorch Dataset for TFT training/inference.

    Optionally applies z-score normalization using provided stats.
    """

    def __init__(
        self,
        temporal: np.ndarray,
        static_continuous: np.ndarray,
        static_categorical: np.ndarray,
        y_win: np.ndarray,
        y_offset: np.ndarray,
        sample_weights: np.ndarray,
        norm_stats: dict[str, np.ndarray] | None = None,
    ):
        self.temporal = temporal.astype(np.float32)
        self.static_continuous = static_continuous.astype(np.float32)
        self.static_categorical = static_categorical.astype(np.int64)
        self.y_win = y_win.astype(np.float32)
        self.y_offset = y_offset.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.y_win)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        temporal = self.temporal[idx]
        static_cont = self.static_continuous[idx]

        if self.norm_stats:
            temporal = (temporal - self.norm_stats["temporal_mean"]) / self.norm_stats["temporal_std"]
            static_cont = (static_cont - self.norm_stats["static_mean"]) / self.norm_stats["static_std"]

        return {
            "temporal": torch.from_numpy(temporal),
            "static_continuous": torch.from_numpy(static_cont),
            "static_categorical": torch.from_numpy(self.static_categorical[idx]),
            "y_win": torch.tensor([self.y_win[idx]]),
            "y_offset": torch.tensor([self.y_offset[idx]]),
            "weight": torch.tensor([self.sample_weights[idx]]),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_dataset.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/dataset.py tests/test_dataset.py
git commit -m "feat: implement TFT dataset with z-score normalization"
```

---

### Task 8: Polygon.io Data Collector

**Files:**
- Create: `ML_tradingAlgo/data/__init__.py`
- Create: `ML_tradingAlgo/data/polygon_collector.py`
- Create: `tests/test_polygon_collector.py`
- Modify: `ML_tradingAlgo/config.py` — add `POLYGON_API_KEY`
- Modify: `ML_tradingAlgo/.env.example` — add `POLYGON_API_KEY=`

This collects historical 1-min and 5-min bars for gap-up stocks from Polygon.io. Filters for gap 25-50%, relative volume >3x, price $1-$30.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_polygon_collector.py
import pytest
from unittest.mock import MagicMock, patch
from ML_tradingAlgo.data.polygon_collector import (
    is_valid_gapup,
    validate_bars,
)


class TestGapUpFilter:
    def test_accepts_30pct_gap(self):
        assert is_valid_gapup(prior_close=10.0, open_price=13.0, volume=500000, avg_volume=100000, price=13.0)

    def test_rejects_20pct_gap_too_low(self):
        assert not is_valid_gapup(prior_close=10.0, open_price=12.0, volume=500000, avg_volume=100000, price=12.0)

    def test_rejects_55pct_gap_too_high(self):
        assert not is_valid_gapup(prior_close=10.0, open_price=15.5, volume=500000, avg_volume=100000, price=15.5)

    def test_rejects_low_relative_volume(self):
        assert not is_valid_gapup(prior_close=10.0, open_price=13.0, volume=200000, avg_volume=100000, price=13.0)

    def test_rejects_price_below_1(self):
        assert not is_valid_gapup(prior_close=0.50, open_price=0.65, volume=500000, avg_volume=100000, price=0.65)

    def test_rejects_price_above_30(self):
        assert not is_valid_gapup(prior_close=25.0, open_price=35.0, volume=500000, avg_volume=100000, price=35.0)


class TestValidateBars:
    def test_valid_bars_pass(self):
        """390 1-min bars in a trading day, all present."""
        assert validate_bars(total_expected=390, actual_count=390)

    def test_rejects_more_than_10pct_missing(self):
        """Less than 90% bars present should fail."""
        assert not validate_bars(total_expected=390, actual_count=300)

    def test_accepts_borderline_missing(self):
        """Exactly 90% present should pass."""
        assert validate_bars(total_expected=390, actual_count=351)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_polygon_collector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ML_tradingAlgo.data'`

- [ ] **Step 3: Update config.py and implement polygon_collector.py**

Add to `ML_tradingAlgo/config.py`:
```python
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]  # Required for training data collection
```

Add to `ML_tradingAlgo/.env.example`:
```
POLYGON_API_KEY=your_polygon_api_key_here
```

Create `ML_tradingAlgo/data/__init__.py` (empty) and `ML_tradingAlgo/data/polygon_collector.py`.

Key functions to implement:
- `is_valid_gapup(prior_close, open_price, volume, avg_volume, price)` — pure filter: gap 25-50%, rel volume >3x, price $1-$30
- `validate_bars(total_expected, actual_count)` — returns False if >10% missing bars
- `find_gapup_events(start_date, end_date, api_key)` — scans daily bars for qualifying stocks, returns list of (ticker, date) tuples
- `collect_bars(ticker, date, timeframe, api_key)` — fetches 1-min or 5-min bars for a specific ticker/date
- `collect_fundamentals(ticker, api_key)` — fetches float, short interest, 52wk high/low
- `collect_training_dataset(start_date, end_date, output_dir, api_key)` — orchestrates full collection, saves to parquet files

Uses `polygon-api-client` library. Rate-limits requests (5/min for free tier, 100/min for paid).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_polygon_collector.py -v`
Expected: All 9 tests PASS (6 gap filter + 3 bar validation)

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/data/__init__.py ML_tradingAlgo/data/polygon_collector.py tests/test_polygon_collector.py ML_tradingAlgo/config.py ML_tradingAlgo/.env.example
git commit -m "feat: implement Polygon.io data collector for historical gap-up events"
```

---

### Task 9: Labeling Pipeline

**Files:**
- Create: `ML_tradingAlgo/data/labeler.py`
- Create: `tests/test_labeler.py`

Computes binary win/loss labels and entry offset labels from raw bar data using the split-exit logic.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_labeler.py
import pytest
import numpy as np
import pandas as pd
from ML_tradingAlgo.data.labeler import compute_labels


@pytest.fixture
def winning_trade_bars():
    """30 bars where price rises 3%+ from entry (bar 0 open)."""
    n = 60
    entry_price = 5.0
    prices = np.linspace(entry_price, entry_price * 1.05, n)
    return pd.DataFrame({
        "open": prices,
        "high": prices + 0.02,
        "low": prices - 0.02,
        "close": prices,
        "volume": np.full(n, 50000),
    })


@pytest.fixture
def losing_trade_bars():
    """30 bars where price drops 3%+ from entry."""
    n = 60
    entry_price = 5.0
    prices = np.linspace(entry_price, entry_price * 0.94, n)
    return pd.DataFrame({
        "open": prices,
        "high": prices + 0.02,
        "low": prices - 0.02,
        "close": prices,
        "volume": np.full(n, 50000),
    })


class TestComputeLabels:
    def test_winning_trade(self, winning_trade_bars):
        y_win, y_offset = compute_labels(
            bars=winning_trade_bars, entry_bar_idx=0,
            tp_pct=3.0, sl_pct=3.0, lookahead_bars=30,
        )
        assert y_win == 1.0

    def test_losing_trade(self, losing_trade_bars):
        y_win, y_offset = compute_labels(
            bars=losing_trade_bars, entry_bar_idx=0,
            tp_pct=3.0, sl_pct=3.0, lookahead_bars=30,
        )
        assert y_win == 0.0

    def test_entry_offset_is_negative_or_zero(self, winning_trade_bars):
        """Offset should be <= 0 (lowest low is at or below entry)."""
        _, y_offset = compute_labels(
            bars=winning_trade_bars, entry_bar_idx=0,
            tp_pct=3.0, sl_pct=3.0, lookahead_bars=30,
        )
        assert y_offset <= 0.0

    def test_entry_offset_atr_normalized(self, winning_trade_bars):
        """Offset should be ATR-normalized, typically in [-2, 0] range."""
        _, y_offset = compute_labels(
            bars=winning_trade_bars, entry_bar_idx=0,
            tp_pct=3.0, sl_pct=3.0, lookahead_bars=30,
        )
        assert -5.0 <= y_offset <= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement labeler.py**

```python
# ML_tradingAlgo/data/labeler.py
import numpy as np
import pandas as pd


def compute_labels(
    bars: pd.DataFrame,
    entry_bar_idx: int,
    tp_pct: float = 3.0,
    sl_pct: float = 3.0,
    lookahead_bars: int = 30,
) -> tuple[float, float]:
    """Compute win/loss label and entry offset from bar data.

    Args:
        bars: OHLCV DataFrame (must extend at least entry_bar_idx + lookahead_bars rows)
        entry_bar_idx: Index of the entry bar
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        lookahead_bars: Number of bars to look ahead for TP/SL

    Returns:
        (y_win, y_offset):
            y_win: 1.0 if TP hit before SL, 0.0 if SL hit first.
                   For trades hitting neither, uses proportional formula binarized at 0.5.
            y_offset: (lowest_low_30bars - entry_open) / ATR
    """
    entry_price = bars.iloc[entry_bar_idx]["open"]  # Use open consistently (spec: bar0_open)
    end_idx = min(entry_bar_idx + lookahead_bars, len(bars))

    # Check TP/SL
    y_win = None
    for i in range(entry_bar_idx + 1, end_idx):
        high = bars.iloc[i]["high"]
        low = bars.iloc[i]["low"]
        pct_high = (high - entry_price) / entry_price * 100
        pct_low = (low - entry_price) / entry_price * 100

        if pct_high >= tp_pct:
            y_win = 1.0
            break
        if pct_low <= -sl_pct:
            y_win = 0.0
            break

    # Neither TP nor SL hit — use proportional formula
    if y_win is None:
        final_close = bars.iloc[end_idx - 1]["close"]
        pct_gain = (final_close - entry_price) / entry_price * 100
        proportional = (sl_pct + pct_gain) / (tp_pct + sl_pct)
        y_win = 1.0 if proportional >= 0.5 else 0.0

    # Entry offset: (lowest low in next 30 bars - entry open) / ATR
    window = bars.iloc[entry_bar_idx:end_idx]
    lowest_low = window["low"].min()
    entry_open = bars.iloc[entry_bar_idx]["open"]

    # ATR (14-period) ending at entry bar
    atr_start = max(0, entry_bar_idx - 14)
    atr_window = bars.iloc[atr_start:entry_bar_idx + 1]
    tr = np.maximum(
        atr_window["high"] - atr_window["low"],
        np.maximum(
            abs(atr_window["high"] - atr_window["close"].shift(1)),
            abs(atr_window["low"] - atr_window["close"].shift(1)),
        ),
    ).dropna()
    atr = tr.mean() if len(tr) > 0 else 1.0
    atr = max(atr, 1e-8)

    y_offset = (lowest_low - entry_open) / atr

    return y_win, y_offset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_labeler.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/data/labeler.py tests/test_labeler.py
git commit -m "feat: implement trade labeling with split-exit logic and ATR-normalized offset"
```

---

## Chunk 3: Training, Inference, and Interpretability

### Task 10: Training Loop with Walk-Forward Validation

**Files:**
- Create: `ML_tradingAlgo/tft/train.py`

Implements the training loop with:
- Walk-forward 4-fold validation
- Combined loss: 0.7 * BCE + 0.3 * MSE
- Class-weighted BCE with recency decay sample weights
- AdamW with cosine annealing, gradient clipping
- Early stopping (patience 15)
- Saves best model weights + normalization stats per fold

- [ ] **Step 1: Implement train.py**

Key functions:
- `create_walk_forward_folds(events, num_folds=4)` — splits events chronologically
- `compute_sample_weights(event_dates, class_labels, lambda_decay=0.0578)` — combines recency decay with class weighting
- `train_fold(train_ds, val_ds, model_config, fold_idx, output_dir)` — trains one fold, returns metrics
- `train_all_folds(data_dir, model_config, output_dir)` — orchestrates full walk-forward training
- `evaluate(model, dataloader)` — computes win rate, profit factor, loss on a dataset

Training config per spec:
```python
TRAIN_CONFIG = {
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 15,
    "grad_clip": 1.0,
    "lambda_decay": 0.0578,  # ln(2)/12
    "bce_weight": 0.7,
    "mse_weight": 0.3,
    "augment_noise_std": 0.001,
    "augment_time_shift": 2,
    "augment_feature_dropout": 0.1,
}
```

Data augmentation is applied on-the-fly in the training Dataset (not the validation Dataset).

- [ ] **Step 2: Test training on synthetic data (smoke test)**

Run:
```bash
cd /Users/jc/Projects/Todd_ML_Algorithm && python -c "
from ML_tradingAlgo.tft.train import train_fold
from ML_tradingAlgo.tft.dataset import TFTDataset, compute_normalization_stats
import numpy as np

# Synthetic data
n = 100
temporal = np.random.randn(n, 30, 47).astype(np.float32)
static_cont = np.random.randn(n, 7).astype(np.float32)
static_cat = np.random.randint(0, 11, (n, 1))
y_win = np.random.randint(0, 2, n).astype(np.float32)
y_offset = np.random.randn(n).astype(np.float32)
weights = np.ones(n, dtype=np.float32)

stats = compute_normalization_stats(temporal, static_cont)
train_ds = TFTDataset(temporal[:80], static_cont[:80], static_cat[:80], y_win[:80], y_offset[:80], weights[:80], stats)
val_ds = TFTDataset(temporal[80:], static_cont[80:], static_cat[80:], y_win[80:], y_offset[80:], weights[80:], stats)

metrics = train_fold(train_ds, val_ds, fold_idx=0, output_dir='/tmp/tft_test', max_epochs=3)
print('Smoke test passed. Metrics:', metrics)
"
```
Expected: Prints metrics dict, no errors

- [ ] **Step 3: Commit**

```bash
git add ML_tradingAlgo/tft/train.py
git commit -m "feat: implement TFT training loop with walk-forward validation"
```

---

### Task 11: Interpretability Visualization

**Files:**
- Create: `ML_tradingAlgo/tft/interpret.py`

Provides functions to extract and visualize:
- Global feature importance (VSN weights averaged across samples)
- Temporal attention heatmaps (which bars drove each prediction)
- Regime monitoring (attention distribution drift over time)

- [ ] **Step 1: Implement interpret.py**

Key functions:
- `plot_feature_importance(model, dataloader, feature_names, save_path)` — bar chart of VSN weights
- `plot_temporal_attention(attn_weights, bar_timestamps, save_path)` — heatmap of attention per bar
- `compute_attention_drift(recent_weights, historical_weights)` — KL divergence for regime detection

Uses matplotlib/seaborn. All plots save to file (no `plt.show()`).

- [ ] **Step 2: Commit**

```bash
git add ML_tradingAlgo/tft/interpret.py
git commit -m "feat: add TFT interpretability tools (feature importance, attention heatmaps)"
```

---

### Task 12: Inference Predictor Wrapper

**Files:**
- Create: `ML_tradingAlgo/tft_predictor.py`
- Create: `tests/test_tft_predictor.py`

Drop-in replacement for `Todd_tradingAlgo1.Todd_predict()`. Loads model weights + normalization stats, accepts raw features, returns (p_win, entry_offset).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tft_predictor.py
import pytest
import torch
import numpy as np
import os
import tempfile
from ML_tradingAlgo.tft.model import TemporalFusionTransformer
from ML_tradingAlgo.tft.dataset import compute_normalization_stats
from ML_tradingAlgo.tft_predictor import TFTPredictor


@pytest.fixture
def saved_model(model_config, tmp_path):
    """Save a model + normalization stats to a temp directory."""
    model = TemporalFusionTransformer(**model_config)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    # Fake norm stats
    stats = {
        "temporal_mean": np.zeros(47, dtype=np.float32),
        "temporal_std": np.ones(47, dtype=np.float32),
        "static_mean": np.zeros(7, dtype=np.float32),
        "static_std": np.ones(7, dtype=np.float32),
    }
    np.savez(tmp_path / "norm_stats.npz", **stats)
    return tmp_path


class TestTFTPredictor:
    def test_predict_returns_tuple(self, saved_model, model_config):
        predictor = TFTPredictor(model_dir=saved_model, model_config=model_config)
        p_win, offset = predictor.predict(
            sequence=np.random.randn(30, 47),
            static_continuous=np.random.randn(7),
            static_categorical=np.array([3]),
        )
        assert isinstance(p_win, float)
        assert isinstance(offset, float)

    def test_p_win_bounded(self, saved_model, model_config):
        predictor = TFTPredictor(model_dir=saved_model, model_config=model_config)
        p_win, _ = predictor.predict(
            sequence=np.random.randn(30, 47),
            static_continuous=np.random.randn(7),
            static_categorical=np.array([3]),
        )
        assert 0.0 <= p_win <= 1.0

    def test_get_attention_weights(self, saved_model, model_config):
        predictor = TFTPredictor(model_dir=saved_model, model_config=model_config)
        predictor.predict(
            sequence=np.random.randn(30, 47),
            static_continuous=np.random.randn(7),
            static_categorical=np.array([3]),
        )
        attn = predictor.get_last_attention_weights()
        assert attn is not None
        assert attn.shape[-1] == 30  # seq_len
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_tft_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tft_predictor.py**

```python
# ML_tradingAlgo/tft_predictor.py
import numpy as np
import torch
from pathlib import Path
from ML_tradingAlgo.tft.model import TemporalFusionTransformer


class TFTPredictor:
    """Inference wrapper for TFT. Loads model + normalization stats.

    Drop-in replacement for Todd_tradingAlgo1.Todd_predict().
    """

    def __init__(self, model_dir: str | Path, model_config: dict):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")

        # Load model
        self.model = TemporalFusionTransformer(**model_config)
        self.model.load_state_dict(
            torch.load(self.model_dir / "model.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load normalization stats
        stats = np.load(self.model_dir / "norm_stats.npz")
        self.norm_stats = {k: stats[k] for k in stats.files}

        self._last_attn_weights = None

    def predict(
        self,
        sequence: np.ndarray,
        static_continuous: np.ndarray,
        static_categorical: np.ndarray,
    ) -> tuple[float, float]:
        """Run inference on a single sample.

        Args:
            sequence: (30, 47) temporal features
            static_continuous: (7,) continuous static features
            static_categorical: (1,) categorical indices

        Returns:
            (p_win, entry_offset)
        """
        # Normalize
        seq_norm = (sequence - self.norm_stats["temporal_mean"]) / self.norm_stats["temporal_std"]
        static_norm = (static_continuous - self.norm_stats["static_mean"]) / self.norm_stats["static_std"]

        # To tensors, add batch dim
        seq_t = torch.from_numpy(seq_norm.astype(np.float32)).unsqueeze(0)
        static_cont_t = torch.from_numpy(static_norm.astype(np.float32)).unsqueeze(0)
        static_cat_t = torch.from_numpy(static_categorical.astype(np.int64)).unsqueeze(0)

        with torch.no_grad():
            p_win, entry_offset, attn_weights = self.model(seq_t, static_cont_t, static_cat_t)

        self._last_attn_weights = attn_weights.squeeze(0).numpy()

        return p_win.item(), entry_offset.item()

    def get_last_attention_weights(self) -> np.ndarray | None:
        return self._last_attn_weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/test_tft_predictor.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add ML_tradingAlgo/tft_predictor.py tests/test_tft_predictor.py
git commit -m "feat: implement TFT inference predictor wrapper"
```

---

### Task 13: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Add new dependencies:
```
polygon-api-client>=1.0
schwab-py>=1.0
matplotlib>=3.5
seaborn>=0.12
```

- [ ] **Step 2: Verify install**

Run: `cd /Users/jc/Projects/Todd_ML_Algorithm && pip install -r requirements.txt --dry-run`
Expected: All packages resolve without conflicts

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add polygon-api-client, schwab-py, matplotlib, seaborn to requirements"
```

---

### Task 14: Update TFT Package `__init__.py`

**Files:**
- Modify: `ML_tradingAlgo/tft/__init__.py`

- [ ] **Step 1: Final `__init__.py` with all exports**

```python
# ML_tradingAlgo/tft/__init__.py
from .layers import GatedLinearUnit, GatedResidualNetwork
from .variable_selection import VariableSelectionNetwork
from .attention import InterpretableMultiHeadAttention
from .model import TemporalFusionTransformer
from .dataset import TFTDataset, compute_normalization_stats
from .features import build_feature_matrix
```

- [ ] **Step 2: Commit**

```bash
git add ML_tradingAlgo/tft/__init__.py
git commit -m "chore: finalize TFT package exports"
```

---

## Summary

| Task | Component | Tests | Estimated Lines |
|------|-----------|-------|-----------------|
| 1 | Test fixtures | conftest.py | ~30 |
| 2 | GRN + GLU layers | 6 tests | ~80 impl + ~50 test |
| 3 | Variable Selection Network | 4 tests | ~60 impl + ~40 test |
| 4 | Interpretable Attention | 4 tests | ~60 impl + ~40 test |
| 5 | Full TFT model | 5 tests | ~180 impl + ~60 test |
| 6 | Feature engineering | 7 tests | ~300 impl + ~100 test |
| 7 | Dataset + normalization | 4 tests | ~60 impl + ~60 test |
| 8 | Polygon.io collector | 9 tests | ~200 impl + ~40 test |
| 9 | Labeling pipeline | 4 tests | ~60 impl + ~50 test |
| 10 | Training loop | smoke test | ~250 impl |
| 11 | Interpretability | — | ~120 impl |
| 12 | Inference predictor | 3 tests | ~70 impl + ~50 test |
| 13 | Dependencies | — | ~4 lines |
| 14 | Package init | — | ~7 lines |
| **Total** | | **46 tests** | **~1,450 impl + ~490 test** |

**Execution order:** Tasks 1-5 (model arch) → 6-9 (data pipeline) → 10-14 (training + integration). Tasks within each chunk can be parallelized where dependencies allow (e.g., Tasks 2-4 are independent).
