# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A momentum gap-up trading system that identifies and trades low-float stocks gapping up 25–50% on high relative volume. The legacy codebase (in `ML_tradingAlgo/`) uses TD Ameritrade API, a simple feedforward PyTorch model, and Twilio for WhatsApp alerts. A new architecture spec (`momentum_trader_project_spec.md`) describes the target system: LSTM+Attention model, Schwab API, PineScript backtesting, and real-time scanner.

## Architecture

### Legacy Pipeline (`ML_tradingAlgo/`)

```
TD_Ameritrade_Data.py (training data collection)
  → convertCSVyToSigmoid.py (label binarization)
  → BalanceDataSigmoid.py (class balancing)
  → Todd_tradingAlgo1.py (PyTorch model: Linear→Tanh→Linear→LeakyReLU→Linear→Sigmoid)

live_data_gather_unified.py (live trading loop)
  → Gathers 1-min and 5-min candles via TDA API polling
  → Engineers 47 features (8×5min candles + 5×1min candles, each with weighted/unweighted/squared variants, plus fundamentals)
  → Calls Todd_tradingAlgo1.Todd_predict() for buy signals
  → Executes trades via tda-api, sends WhatsApp via Twilio
  → Split exit: half at trailing stop, half at fixed TP/SL
```

### Key Data Flow

- Features are computed as `(((close-low) - (high-close))/open * 1000) * (volume/float*100)` — the `* 1000` scaling is critical and must stay consistent between training (`TD_Ameritrade_Data.py`) and inference (`live_data_gather_unified.py`)
- Two models run in parallel: `model_tanh` (full training stats) and `model_2wk` (2-week stats) — both must predict class 1 for a buy signal
- `live_data_gather_unified.py` accepts `instance_id` CLI arg (1/2/3) to run multiple instances with separate CSV outputs

### Configuration

- All secrets in `ML_tradingAlgo/.env` (loaded via `config.py`): Twilio creds, brokerage account ID, WhatsApp numbers, TD Ameritrade client ID
- `.env.example` shows required variables

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run live trading (instance 1, 2, or 3)
cd ML_tradingAlgo && python live_data_gather_unified.py 1

# Collect training data
cd ML_tradingAlgo && python TD_Ameritrade_Data.py

# Refresh OAuth token
cd ML_tradingAlgo && python TD_Amer_token_refresh.py

# Verify all files parse
for f in ML_tradingAlgo/*.py; do python -c "import ast; ast.parse(open('$f').read())"; done
```

## Important Constraints

- The PyTorch model weights (`current_weights`, `current_weights_2wk`) need retraining — legacy TF weights are incompatible
- `live_data_gather.py` (without suffix) is a simpler data-gathering-only tool with no trading logic — keep it separate from the unified trading module
- The `tda-api` library handles OAuth 2.0 with token refresh via `token.pickle` — Schwab migration is planned per the project spec
