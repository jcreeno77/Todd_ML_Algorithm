# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A momentum gap-up trading system that identifies and trades low-float stocks gapping up 25–50% on high relative volume. The legacy codebase (in `ML_tradingAlgo/`) uses TD Ameritrade API and a simple feedforward PyTorch model. A new architecture spec (`momentum_trader_project_spec.md`) describes the target system: LSTM+Attention model, Schwab API, PineScript backtesting, and real-time scanner.

**Alerting:** Twilio has been fully removed. All alerts route through a channel-agnostic notifier (`ML_tradingAlgo/data/notify.py`), which logs by default and POSTs to a Discord webhook when `ALERT_WEBHOOK_URL` is set. See `docs/notifications.md`.

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
  → Executes trades via tda-api, sends alerts via notify() (logging / Discord webhook)
  → Split exit: half at trailing stop, half at fixed TP/SL
```

### Key Data Flow

- Features are computed as `(((close-low) - (high-close))/open * 1000) * (volume/float*100)` — the `* 1000` scaling is critical and must stay consistent between training (`TD_Ameritrade_Data.py`) and inference (`live_data_gather_unified.py`)
- Two models run in parallel: `model_tanh` (full training stats) and `model_2wk` (2-week stats) — both must predict class 1 for a buy signal
- `live_data_gather_unified.py` accepts `instance_id` CLI arg (1/2/3) to run multiple instances with separate CSV outputs

### Configuration

- All secrets in `ML_tradingAlgo/.env` (loaded via `config.py`): brokerage account ID, TD Ameritrade client ID, optional `ALERT_WEBHOOK_URL` for alerts, plus Schwab/AWS/fundamentals keys for the TFT data pipeline
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

## Cost Discipline (AWS) — IMPORTANT

This is a **personal project on a tight budget**. Keep cloud spend minimal. Treat every AWS design choice through a cost lens:

- **S3 is the only AWS service we use.** Do NOT introduce paid managed services (DynamoDB, RDS, Athena, Glue, SageMaker, MSK, etc.) without explicit sign-off — there is almost always a free local/in-process alternative (e.g. watermarks are per-symbol JSON in S3, not DynamoDB; training runs locally, not SageMaker).
- **Watch request counts, not just storage.** S3 cost here is dominated by GET/PUT request volume from many small Parquet files, not bytes stored. Prefer batched writes (the live tee already batches); the deferred weekly compaction job is the first lever if training reads get expensive.
- **Never do O(n²) S3 access patterns.** Known hotspot: per-date full-table reads (e.g. ADV recomputation in `collector`/`backfill`) scale quadratically over long windows — compute once per symbol and reuse. Be especially careful in any multi-year backfill.
- **Lifecycle + storage class.** Old/cold partitions should move to cheaper storage (S3 Infrequent Access / Glacier) or expire via a lifecycle policy. Raw bars compress well; don't store recomputable features.
- **Stay in one region, no cross-region transfer, no inter-AZ chatter.** Avoid data egress.
- **Tag/scope the IAM key to just this project's bucket** (least privilege) — limits blast radius and accidental spend.
- When proposing anything cloud-touching, **state the rough cost impact** in the plan/PR so it's a conscious decision.
