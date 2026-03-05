# Momentum Gap-Up Trading System

## Project Overview

This system automates the identification and trading of low-float stocks gapping up 25–50% on high relative volume. It combines a PineScript backtesting strategy with a PyTorch machine learning model for entry prediction, a real-time scanner, and execution through the Charles Schwab API.

The core thesis: low-float stocks that gap up significantly on outsized volume attract momentum traders, creating predictable intraday patterns. By loading 30 minutes of premarket data, warming up during the first 30 minutes of the regular session, and trading a 2-hour window (9:30–11:30 ET), the system targets the most liquid and volatile part of the day.

---

## System Architecture

```
 PREMARKET (9:00 ET)         MARKET OPEN (9:30)           CUTOFF (11:30)
       │                           │                            │
       ▼                           ▼                            ▼
 ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
 │ SCANNER  │→ │ FEATURE ENG  │→ │  ML MODEL    │→ │   EXECUTION     │
 │          │  │              │  │              │  │                 │
 │ Gap 25-50│  │ 42 features  │  │ LSTM+Attn    │  │ Schwab API      │
 │ RelVol>3x│  │ 30-bar seq   │  │ Entry signal │  │ Split exits     │
 │ Dyn float│  │ 1-min bars   │  │ P(win)>0.65  │  │ Half@3% TP/SL   │
 │ $1-$30   │  │              │  │              │  │ Half@2.5% trail │
 └──────────┘  └──────────────┘  └──────────────┘  └─────────────────┘
       │                                                      │
       └──────────── BACKTEST ENGINE ◄────────────────────────┘
```

---

## Component 1: PineScript Strategy (Backtesting)

**File:** `momentum_gapup_strategy.pine`

The PineScript strategy runs on TradingView and serves as the primary backtesting and visualization layer. It is designed to be applied to 1-minute charts of individual tickers that meet the scanner criteria.

### What It Does

The strategy calculates a composite "ML Proxy Score" on every bar — a 0-to-1 value built from five weighted components that approximate what the full PyTorch model will learn. This score combines momentum indicators (EMA alignment, MACD acceleration, consecutive green candles), volume confirmation (relative volume, volume spikes, volume EMA crossovers), VWAP relationship (price above VWAP, VWAP reclaim events), RSI positioning within a momentum-friendly range (50–80), and price structure quality (holding above key EMAs, making new 5-bar highs).

### Entry Logic

A long entry triggers when all of the following are true simultaneously: the stock gapped up 25–50% from the prior close, relative volume exceeds 3x the 20-day average, the ML proxy score is at or above 0.65, price is above VWAP, price is above the 9 EMA, RSI is between 50 and 80, a volume spike is detected, the warmup period (30 bars after open) has elapsed, and fewer than 5 positions are open.

### Exit Logic — Split Position

The exit system implements the exact risk rules you specified. The first half of the position exits at either +3% take profit or −3% stop loss, whichever is hit first. Once the first half exits at profit, the second half switches to a 2.5% trailing stop from the high water mark. If the first half hits the stop loss, the entire position closes. Any remaining position force-closes at 11:30 ET (end of trading window).

### Visual Overlays

The strategy plots VWAP (purple), 9 EMA (blue), and 20 EMA (orange) directly on the chart. Entry signals appear as green triangles below bars. When in a position, the take profit level (green), stop loss level (red), and trailing stop level (orange) are drawn as horizontal lines. The trading window (9:30–11:30) is highlighted with a blue background, and gap days are marked in yellow. An info table in the top-right corner shows real-time values for gap %, ML score, relative volume, RSI, VWAP relationship, position status, and whether the first half has been exited.

### How to Backtest

Open TradingView and navigate to a 1-minute chart of any low-float stock that recently had a major gap-up (for example, stocks that appeared on your scanners). Paste the PineScript code into the Pine Editor and add it to the chart. The Strategy Tester tab will show trade-by-trade results, equity curve, and performance metrics. To backtest systematically, apply the strategy to a list of known gap-up tickers and record the results for each.

---

## Component 2: PyTorch ML Model (Entry Prediction)

**Purpose:** Predict the probability that a gap-up stock will continue higher after the warmup period, producing an entry signal with a confidence score and suggested entry price.

### Architecture: LSTM with Multi-Head Attention

The model uses a 2-layer LSTM (hidden size 128) followed by a 4-head multi-head attention mechanism. The LSTM captures temporal dependencies in the 30-bar price/volume sequence, and the attention layer learns which bars in the sequence matter most (e.g., the first candle after open, volume climax bars, VWAP reclaim moments).

```
Input (30 x 42 features)
        │
   ┌────▼────┐
   │  LSTM   │  2 layers, 128 hidden, dropout 0.3
   │  Encoder│
   └────┬────┘
        │
   ┌────▼────────┐
   │  Multi-Head  │  4 heads, learning temporal importance
   │  Attention   │
   └────┬────────┘
        │
   ┌────▼────┐
   │  Dense  │  128 → 64 → 2 outputs
   │  Heads  │
   └────┬────┘
        │
   ┌────▼──────────────────┐
   │ Output 1: P(continue) │  Sigmoid → 0-1 probability
   │ Output 2: Entry Offset│  Predicted optimal entry price offset
   └───────────────────────┘
```

### Input Features (42 total)

**Price Action (12 features):** open/high/low/close normalized to VWAP, gap percentage from prior close, distance from 5/9/20 EMAs, ATR-normalized range, candle body ratio (body size / total range), upper and lower wick ratios, and number of consecutive green/red candles.

**Volume Profile (10 features):** raw volume (log-scaled), relative volume vs 20-day average, volume vs 5/10/20 period EMAs, cumulative volume ratio (vs expected for time of day), bid-ask spread estimate (high-low as proxy), volume-price trend (VPT), on-balance volume (OBV) slope, and money flow index.

**VWAP Dynamics (5 features):** price distance from VWAP (normalized), VWAP slope, number of VWAP touches in last N bars, time since last VWAP cross, and whether price reclaimed VWAP from below.

**Momentum Indicators (8 features):** RSI (14), MACD line, MACD histogram, MACD histogram acceleration, rate of change (5-bar and 10-bar), stochastic %K, and Williams %R.

**Market Context (4 features):** SPY return for the current session, SPY RSI, sector ETF return (mapped to stock's sector), and VIX level (inverted, normalized).

**Microstructure (3 features):** float shares (log-scaled), short interest ratio, and days since last earnings.

### Training Pipeline

The training process uses walk-forward validation with 4 folds to prevent lookahead bias. Each fold trains on N months of data, validates on the next month, and tests on the month after that. The model optimizes a combined loss function: binary cross-entropy for the continuation probability head and mean squared error for the entry price offset head, weighted 70/30. Training uses the AdamW optimizer with a learning rate of 1e-4, cosine annealing schedule, and early stopping with patience of 15 epochs. Data augmentation includes adding Gaussian noise to prices (0.1% std), random time shifts (±2 bars), and dropout of individual features during training.

### Data Requirements

The model needs historical 1-minute bar data for stocks that gapped up 25–50% on high relative volume. Ideally this covers at least 12–24 months of data across hundreds of gap-up events. Each event is labeled with whether the stock continued higher after the warmup period (binary label) and what the optimal entry price was within the first 30 minutes of trading (regression label).

---

## Component 3: Real-Time Scanner

**Purpose:** Identify stocks meeting the gap-up criteria each morning before and during the trading session.

### Premarket Scan (Starting 9:00 ET)

The scanner queries the Schwab API market data endpoint every 60 seconds starting at 9:00 ET. It pulls all stocks with premarket activity and filters for gap percentage between 25–50% from prior close, premarket volume above 100,000 shares, price between $1 and $30, and relative volume above 3x the 20-day average. Float data is pulled from a supplementary source (financial data API) and used as a feature rather than a hard filter, since the ML model weighs float dynamically.

### Live Scan (9:30–11:30 ET)

Once the market opens, the scanner shifts to pulling 1-minute bars for all candidates identified in the premarket scan. It feeds the last 30 bars of data through the feature engineering pipeline and into the ML model for scoring. Any stock with a model score above 0.65 generates an entry alert.

### Output

The scanner produces a ranked watchlist updated in real-time, with each candidate showing the ticker, gap %, relative volume, float, ML score, and suggested entry price. The top candidates are passed to the execution engine.

---

## Component 4: Backtest Engine (Python)

**Purpose:** Validate the strategy on historical data with realistic assumptions before risking real capital.

### Design Philosophy

The backtester is event-driven (not vectorized) to accurately model the split-exit logic and trailing stops. It processes 1-minute bars sequentially, tracking position state, high water marks, and partial fills exactly as they would occur in live trading.

### Key Assumptions

Commissions are $0 (Schwab's standard equity commission). Slippage is modeled at 5 basis points per trade, which is conservative for liquid gap-up stocks but accounts for the spread on lower-float names. Fills assume execution at the next bar's open after a signal, not at the signal bar's close. Partial fills are not modeled (assumes full fills, reasonable for position sizes under $2,500 on stocks with 100K+ premarket volume).

### Walk-Forward Methodology

The backtester uses 4 walk-forward windows. For each window, the ML model is trained on the in-sample data, validated on a holdout set, and then the strategy is tested on the out-of-sample period. This prevents overfitting and gives a realistic estimate of live performance. Results are reported per-window and in aggregate.

### Metrics Reported

The backtester calculates and reports total return and annualized return, Sharpe ratio and Sortino ratio, maximum drawdown (peak-to-trough), win rate and average win/loss ratio, profit factor (gross profits / gross losses), average trade duration, number of trades per day, equity curve with drawdown overlay, trade distribution by hour and day of week, and performance by gap size bucket (25–30%, 30–40%, 40–50%).

---

## Component 5: Schwab Execution Engine

**Purpose:** Translate ML entry signals into live orders through the Charles Schwab API.

### Authentication

The Schwab API (migrated from TD Ameritrade) uses OAuth 2.0 with PKCE. The system stores refresh tokens locally and auto-refreshes access tokens before they expire. Initial setup requires a one-time browser-based authorization flow.

### Order Flow

When the ML model generates an entry signal, the execution engine performs the following sequence. First, it checks the daily circuit breaker (has the account lost more than 5% today?). Second, it verifies the position count is under the maximum. Third, it calculates position size as 10% of equity, capped at the per-trade risk limit. Fourth, it submits a limit order at the ML model's suggested entry price with a 30-second time-in-force. If filled, it immediately submits two exit orders: an OCO (one-cancels-other) bracket with the +3% limit sell and −3% stop for the first half, and a trailing stop order at 2.5% for the second half.

### Order Types Used

The system uses limit orders for entries (not market orders, to control slippage), OCO brackets for the first-half exit, trailing stop orders for the second half, and market orders only for the end-of-window forced exit at 11:30 ET.

### Safety Features

The execution engine includes several safety mechanisms: a daily loss circuit breaker that halts all trading if the account drops 5% in a day, a position count limiter (max 5 concurrent), a per-trade size cap, a kill switch that can be triggered manually to cancel all open orders and flatten all positions, and comprehensive logging of every order submission, fill, and cancellation.

### Paper Trading Mode

The engine supports a paper trading mode that runs the full pipeline (scanner → features → ML → order generation) but logs orders to a file instead of submitting them to Schwab. This allows validation of the complete system before going live.

---

## Data Pipeline

### Recommended Data Source: Schwab API + Supplementary

For live trading and recent backtesting, the Schwab API provides real-time and historical 1-minute bars, real-time quotes and Level 1 data, account positions and balances, and order submission and management. For extended historical data (training the ML model), supplement with a provider like Polygon.io (which offers 1-minute bars going back years for all US equities) or Alpha Vantage. Float and short interest data can be sourced from Financial Modeling Prep, Finviz, or SEC filings.

### Data Flow

```
Schwab API ──→ Raw 1-min bars ──→ Feature Engineering ──→ 42-feature matrix
                                          │
Polygon.io ──→ Historical bars ───────────┘
                                          │
Float Data ──→ Static features ───────────┘
                                          │
                                    ┌─────▼─────┐
                              Train │  PyTorch   │ Predict
                              Data  │   Model    │ Live
                                    └─────┬─────┘
                                          │
                                    Entry Signals
                                          │
                                    ┌─────▼─────┐
                                    │  Schwab    │
                                    │  Orders    │
                                    └───────────┘
```

---

## Getting Started — Recommended Sequence

**Phase 1 — PineScript Validation.** Load the PineScript strategy onto TradingView. Apply it to 1-minute charts of recent gap-up stocks you've traded or watched. Review the entry/exit points visually. Tune the ML proxy score threshold and other inputs to match your intuition. This phase validates the core logic before writing any Python.

**Phase 2 — Historical Data Collection.** Set up a Schwab developer account and obtain API credentials. Write the data pipeline to pull historical 1-minute bars for stocks that met the gap-up criteria over the past 12–24 months. Supplement with Polygon.io if Schwab's historical depth is insufficient. Label each gap-up event with continuation (yes/no) and optimal entry price.

**Phase 3 — ML Model Training.** Build and train the PyTorch LSTM+Attention model on the labeled dataset using walk-forward validation. Evaluate performance on held-out data. Target a win rate above 55% with a profit factor above 1.5 before proceeding.

**Phase 4 — Python Backtesting.** Run the full event-driven backtest with the trained model making entry decisions and the hard-coded exit rules managing risk. Compare results to the PineScript backtest for consistency. Analyze performance across market conditions.

**Phase 5 — Paper Trading.** Deploy the live scanner + ML model + execution engine in paper mode. Run for at least 2–4 weeks, comparing paper results to backtest expectations. Debug any discrepancies in order timing, fill assumptions, or data latency.

**Phase 6 — Live Trading.** Once paper results are satisfactory, switch to live execution with reduced position sizes (e.g., 5% instead of 10%). Scale up gradually as confidence builds.

---

## Risk Disclaimer

This system is a research and development tool. Algorithmic trading involves substantial risk of loss. Past backtest performance does not guarantee future results. Low-float momentum stocks are among the most volatile and risky securities to trade. Always paper trade extensively before risking real capital, and never trade with money you cannot afford to lose. This is not financial advice.
