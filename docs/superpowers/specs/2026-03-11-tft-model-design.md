# TFT Model Design: Momentum Gap-Up Trading System

## Overview

Replace the legacy 3-layer feedforward PyTorch model and the planned LSTM+Attention architecture with a Temporal Fusion Transformer (TFT) for entry prediction in the momentum gap-up trading system. TFT provides built-in variable selection (automatically learns which features matter), interpretable temporal attention (shows which bars drove each prediction), and native support for mixed-frequency inputs.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature set | Merged legacy + spec features (55 total) | TFT's Variable Selection Network auto-prunes weak features |
| Sequence format | 1-min bars as temporal sequence + 5-min aggregates as aligned observed inputs | Preserves multi-timeframe signal; TFT handles mixed-frequency natively |
| Output heads | P(win) sigmoid + entry price offset linear | Probability enables threshold tuning; offset enables limit order placement |
| Historical data | Polygon.io for training, Schwab for live | Polygon has years of 1-min bars; clean separation of concerns |
| Dual-model consensus | Replaced by recency-weighted loss + attention monitoring | Single model with exponential decay weighting captures regime awareness more elegantly |
| Implementation | Custom PyTorch TFT, reference-guided | No library dependency risk; skip unused decoder/multi-step components (~40% reduction) |
| Deployment | Phased: offline eval -> paper trading -> live | Each phase has a gate before proceeding |

---

## Model Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                          │
│                                                              │
│  Static Features ──> [Variable Selection Net] ──> context    │
│  (float, sector,      learns which matter       vectors (cs) │
│   short interest,                                            │
│   gap %)                                                     │
│                                                              │
│  Temporal Features ──> [Variable Selection Net] ──> selected │
│  (1-min bars,           per-timestep selection    features    │
│   5-min aggregates,     conditioned on statics               │
│   indicators)                                                │
├─────────────────────────────────────────────────────────────┤
│                    TEMPORAL PROCESSING                        │
│                                                              │
│  selected features ──> [LSTM Encoder] ──> temporal states    │
│                         2-layer, 160 hidden                  │
│                         dropout 0.3                          │
│                         enriched with static context         │
├─────────────────────────────────────────────────────────────┤
│                    ATTENTION LAYER                            │
│                                                              │
│  temporal states ──> [Interpretable Multi-Head Attention]    │
│                       4 heads, each produces                 │
│                       per-timestep importance weights         │
│                       (regime awareness via attention drift)  │
│                    ──> [Gated skip connection]               │
├─────────────────────────────────────────────────────────────┤
│                    OUTPUT HEADS                               │
│                                                              │
│  attended context ──> [GRN -> Dense 128 -> 64]               │
│                       |──> Sigmoid: P(win)                   │
│                       └──> Linear: entry price offset        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

**Gated Residual Network (GRN):** The fundamental building block used throughout. Architecture: Linear -> ELU -> Linear -> GLU gate -> skip connection + LayerNorm. Provides non-linear processing with gated skip connections that allow the network to suppress irrelevant transformations.

**Variable Selection Network (VSN):** Applies individual GRNs to each input feature, then computes softmax selection weights across all features. Produces per-feature importance scores that sum to 1. Static VSN runs once per sample; temporal VSN runs per timestep, conditioned on static context vectors.

**Interpretable Multi-Head Attention:** Standard scaled dot-product multi-head attention with one key modification: all heads share the same value projection weights. This means attention weights from each head are directly interpretable as temporal importance — you can inspect which bars the model focused on for any given prediction.

**Static Enrichment:** Static context vectors (from the static VSN) are injected into every temporal timestep via GRN. This lets the model condition temporal processing on stock-level characteristics (e.g., "this is a 2M float stock" influences how each candle is interpreted).

### What We Skip (vs Full TFT)

The canonical TFT includes a decoder with known future inputs and multi-horizon quantile output. We skip:
- Decoder branch (no known future inputs in this use case)
- Multi-step quantile forecasting (we need single-step, two-head output)
- Known future inputs processing (no calendar/scheduled features)

This removes ~40% of standard TFT complexity.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dimension | 160 | Wider than spec's 128; TFT gating benefits from this |
| LSTM layers | 2 | |
| Attention heads | 4 | Interpretable multi-head |
| Dropout | 0.3 | Applied throughout |
| Sequence length | 30 bars (1-min) | With 5-min aggregates aligned |

---

## Feature Engineering

### Static Features (8 total, per-stock)

| # | Feature | Source | Processing |
|---|---------|--------|------------|
| 1 | float_shares | Polygon/fundamentals | Log-scaled |
| 2 | short_interest_ratio | Polygon/fundamentals | Raw |
| 3 | gap_percentage | Computed from prior close to open | Raw |
| 4 | sector_id | Mapped lookup table | Categorical, learned embedding |
| 5 | days_since_earnings | Fundamentals API | Raw |
| 6 | 52wk_high_ratio | Legacy: `1 - close/high52` | Raw |
| 7 | 52wk_low_ratio | Legacy: `1 - low52/close` | Raw |
| 8 | premarket_range | Legacy: `high_ratio + low_ratio` | Raw |

### Temporal Observed Features (38 per 1-min bar)

**Price Action (8):**

| # | Feature |
|---|---------|
| 1 | open / VWAP |
| 2 | high / VWAP |
| 3 | low / VWAP |
| 4 | close / VWAP |
| 5 | ATR-normalized range |
| 6 | candle body ratio (body / total range) |
| 7 | upper wick ratio |
| 8 | lower wick ratio |

**Volume Profile (8):**

| # | Feature |
|---|---------|
| 9 | log(volume) |
| 10 | relative volume vs 20-day avg |
| 11 | volume / 5-bar EMA |
| 12 | volume / 10-bar EMA |
| 13 | volume / 20-bar EMA |
| 14 | cumulative volume ratio (vs expected for time of day) |
| 15 | volume-price trend (VPT) |
| 16 | OBV slope |

**VWAP Dynamics (5):**

| # | Feature |
|---|---------|
| 17 | price distance from VWAP (ATR-normalized) |
| 18 | VWAP slope (5-bar) |
| 19 | VWAP touches in last 10 bars |
| 20 | bars since last VWAP cross |
| 21 | VWAP reclaim flag (boolean: price crossed above VWAP from below) |

**Momentum Indicators (8):**

| # | Feature |
|---|---------|
| 22 | RSI(14) |
| 23 | MACD line |
| 24 | MACD histogram |
| 25 | MACD histogram acceleration (1-bar delta) |
| 26 | ROC(5) |
| 27 | ROC(10) |
| 28 | stochastic %K |
| 29 | Williams %R |

**Market Context (4):**

| # | Feature |
|---|---------|
| 30 | SPY session return |
| 31 | SPY RSI(14) |
| 32 | sector ETF return (mapped via sector_id) |
| 33 | VIX level (inverted, normalized) |

**Other (2):**

| # | Feature |
|---|---------|
| 34 | money flow index |
| 35 | consecutive green/red candle count (positive = green, negative = red) |

**Legacy Core Signal (3):**

| # | Feature | Formula |
|---|---------|---------|
| 36 | candle pressure weighted | `(((close-low)-(high-close))/open*1000) * (vol/float*100)` |
| 37 | candle pressure unweighted | `(((close-low)-(high-close))/open*1000)` |
| 38 | candle pressure squared | `feature_36 ** 2` |

*(The `* 1000` scaling factor from legacy must be preserved for consistency.)*

### 5-Min Aggregate Observed Features (9 per timestamp)

5-min bars are fetched separately from the data source (not aggregated from 1-min bars), aligned to clock time starting at market open (9:30, 9:35, 9:40, ...). At each 1-min timestep, the most recent completed 5-min bar's features are used (forward-filled until the next 5-min bar completes).

| # | Feature | Notes |
|---|---------|-------|
| 1 | 5-min candle pressure weighted | Legacy formula (feature 36 applied to 5-min candle) |
| 2 | 5-min candle pressure unweighted | Legacy formula (feature 37 applied to 5-min candle) |
| 3 | 5-min candle pressure squared | Legacy formula (feature 38 applied to 5-min candle) |
| 4 | 5-min OHLC range normalized | (high - low) / ATR |
| 5 | 5-min relative volume | 5-min volume / 20-day avg 5-min volume |
| 6 | 5-min VWAP distance | ATR-normalized |
| 7 | 5-min RSI(14) | |
| 8 | 5-min MACD histogram | |
| 9 | 5-min OBV slope | |

### Total: 8 static + 47 temporal (38 one-min + 9 five-min) = 55 input features

### Input Normalization

All continuous features are z-score normalized using training set statistics (mean/std computed per walk-forward fold's training split). Normalization stats are saved alongside model weights and reused at inference. The categorical feature (`sector_id`) is excluded from normalization and handled via embedding.

---

## Training Pipeline

### Data Collection

- **Source:** Polygon.io REST API for historical 1-min bars
- **Filter criteria:** Gap 25-50% from prior close, relative volume >3x 20-day avg, price $1-$30
- **Target volume:** 12-24 months of data, 500+ gap-up events across market conditions
- **Supplementary data:** Float/short interest from Financial Modeling Prep or SEC filings

### Labeling

- **Binary label (y_win):** Did the trade hit +3% take profit before -3% stop loss using the split-exit logic? 1 = yes, 0 = no. For trades that neither hit TP nor SL within the window, use the proportional formula from legacy: `y = (|SL| + pct_gain) / (TP + |SL|)`, binarized at 0.5.
- **Entry offset label (y_offset):** The percentage distance from bar-0 open to the lowest low within the first 30 bars, normalized by ATR: `y_offset = (lowest_low_30bars - bar0_open) / ATR`. This represents the best available entry price in hindsight. At inference, the predicted offset is applied to the current price to set a limit order below market. Units are ATR-normalized percentages (typically -2.0 to 0.0, where 0.0 means enter at current price).

### Walk-Forward Validation

4 folds with expanding training window:

```
Fold 1: [==== Train (months 1-8) ====][= Val (month 9) =][= Test (month 10) =]
Fold 2: [====== Train (months 1-11) ======][= Val (month 12) =][= Test (month 13) =]
Fold 3: [======== Train (months 1-14) ========][= Val (month 15) =][= Test (month 16) =]
Fold 4: [========== Train (months 1-17) ==========][= Val (month 18) =][= Test (month 19) =]
```

### Loss Function

```
L = 0.7 * BCE(P_win, y_win) + 0.3 * MSE(entry_offset, y_offset)
```

Sample weights use exponential recency decay: `w_i = exp(-lambda * age_months_i)` where `lambda = ln(2)/12 ≈ 0.0578` (events 12 months old get 0.5x weight, 24 months get 0.25x). Lambda can be tuned via validation performance; the default is derived from the half-life formula. This replaces the dual-model consensus approach.

**Class balancing:** Apply class-weighted BCE where the positive class weight = `n_negative / n_positive`. This is combined multiplicatively with the recency weights. The legacy pipeline used oversampling (`BalanceDataSigmoid.py`); class-weighted loss achieves the same effect without duplicating samples, which is preferable for TFT since duplicate sequences would bias the attention mechanism.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 with cosine annealing (higher than project spec's 1e-4; TFT's gating and layer norm stabilize training at higher LR, and cosine annealing decays it smoothly) |
| Early stopping | Patience 15 epochs on validation loss |
| Batch size | 64 |
| Gradient clipping | Max norm 1.0 |
| Data augmentation | Gaussian noise 0.1% std on prices, random time shifts +/-2 bars, 10% feature dropout per sample |

### Interpretability Outputs

- **Feature importance:** Global VSN weights — ranked list of which of the 55 features matter most
- **Temporal attention:** Per-prediction heatmap showing which of the 30 bars drove the decision
- **Regime monitoring:** Track attention weight distribution over time; flag when recent predictions concentrate attention differently than historical baseline

---

## Inference & Integration

### Module Structure

```
ML_tradingAlgo/
├── config.py                        # Add Polygon API key, Schwab creds
├── tft/
│   ├── model.py                     # TFT architecture (GRN, VSN, attention, output heads)
│   ├── features.py                  # Feature engineering pipeline (55 features)
│   ├── dataset.py                   # PyTorch Dataset for temporal sequences
│   ├── train.py                     # Training loop with walk-forward validation
│   └── interpret.py                 # Attention/feature importance visualization
├── tft_predictor.py                 # Inference wrapper (drop-in for Todd_tradingAlgo1)
├── data/
│   ├── polygon_collector.py         # Historical data collection from Polygon.io
│   └── labeler.py                   # Win/loss labeling with split-exit logic
├── live_data_gather_unified.py      # Updated to use tft_predictor
└── ...existing files unchanged...
```

Legacy files (`Todd_tradingAlgo1.py`, `TD_Ameritrade_Data.py`, etc.) remain untouched.

### Predictor Interface

`tft_predictor.py` exposes:

```python
def predict(
    sequence: np.ndarray,
    static_continuous: np.ndarray,
    static_categorical: np.ndarray,
) -> tuple[float, float]:
    """
    Args:
        sequence: (30, 47) array of temporal features (38 one-min + 9 five-min aligned)
        static_continuous: (7,) array of continuous static features
            [float_shares, short_interest, gap_pct, days_since_earnings,
             high52_ratio, low52_ratio, premarket_range]
        static_categorical: (1,) int array of categorical feature indices
            [sector_id]  — embedded internally, embedding dim = 8

    Returns:
        (p_win, entry_offset): probability of winning trade, ATR-normalized entry offset
    """
```

This replaces the call to `Todd_tradingAlgo1.Todd_predict()` in `live_data_gather_unified.py`.

### Phased Deployment

**Phase 1 — Offline Evaluation**
- Collect historical data via Polygon.io
- Train TFT with walk-forward validation
- Gate: win rate >55%, profit factor >1.5 on held-out test folds
- Deliverables: trained weights, feature importance report, attention visualizations

**Phase 2 — Paper Trading**
- Wire `tft_predictor.py` into `live_data_gather_unified.py`
- Schwab API for live market data (replacing TD Ameritrade)
- All orders logged to file, not submitted
- Log predictions, attention weights, feature importances per trade
- Gate: 2-4 weeks of paper results consistent with backtest expectations

**Phase 3 — Live Trading**
- Enable real order submission via Schwab execution engine
- Reduced position sizes (5% equity, half of target 10%)
- Continuous attention weight monitoring for regime shift detection
- Scale up gradually as confidence builds

---

## Dependencies

New dependencies to add to `requirements.txt`:

| Package | Purpose |
|---------|---------|
| polygon-api-client | Polygon.io data collection |
| schwab-py | Schwab API for live data + execution |
| matplotlib / seaborn | Interpretability visualizations |

PyTorch is already a dependency. No new ML framework libraries needed — the TFT is implemented from scratch.

---

## Risk Considerations

- **Data quality:** Polygon.io 1-min bars may have gaps or corporate action artifacts. Need a data cleaning/validation step in `polygon_collector.py`.
- **Feature consistency:** The legacy `* 1000` scaling in the candle pressure formula must be preserved exactly. Feature engineering is shared between training (`features.py`) and inference (`tft_predictor.py`) via a single code path.
- **Overfitting:** 55 features on potentially <1000 events is a concern. TFT's variable selection and dropout mitigate this, but early stopping and walk-forward validation are critical.
- **Latency:** TFT inference is heavier than the legacy feedforward net. Should benchmark — if >100ms per prediction on CPU, consider GPU or model optimization.
- **Multi-instance support:** The legacy `live_data_gather_unified.py` supports multiple instances via `instance_id` CLI arg. The `tft_predictor.py` integration must preserve this — each instance loads the same model weights but writes to separate output CSVs (existing behavior).
- **Hardware:** Training targets a single GPU (CUDA). For the dataset size (500-1000 sequences of length 30), training should complete in <1 hour on a consumer GPU. CPU fallback is acceptable but slower. Inference runs on CPU (latency target: <50ms per prediction).
