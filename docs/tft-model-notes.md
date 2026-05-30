# TFT Model — Specs, Rationale & Roadmap

Reference notes for the Temporal Fusion Transformer momentum model. Architecture in `ML_tradingAlgo/tft/`; design specs in `docs/superpowers/specs/`.

> **Status (2026-05-30):** Architecture + feature/augmentation pipeline are implemented and on `main`. **No checkpoint has been trained yet** and the live loop still calls the legacy model, not `TFTPredictor`. Every "benefit" below is a design property, not a measured result — validation waits on a trained baseline + walk-forward eval.

## Specs

Canonical config: `hidden_size=160`, `lstm_layers=2`, `attention_heads=4`, `dropout=0.3`, `sequence_length=30`, `categorical_embedding_dim=8`.

- **Parameters: ~5.44M** trainable (5,436,138; ~21.7 MB fp32). Scales with `hidden_size` — most of the model is `hidden_size`-width GRNs, so this is config-dependent.
- **Features: 80 continuous + 1 categorical per sample.**
  - Temporal: **69** per timestep (60 one-minute + 9 five-minute) over a **30-bar** window → `(30, 69)`.
  - Static: **11 continuous** (float, short interest, gap %, days-to-earnings, 52wk ratios, premarket range, day-of-week sin/cos, day-of-run, gap-vs-prior-range) **+ 1 categorical** (GICS sector, cardinality 11).
- **Tokenization: none.** Not an LLM — no tokenizer, no vocabulary. Continuous features are z-score normalized and projected through per-feature GRNs; the single categorical (sector) is the only embedded input via `nn.Embedding(11, 8)` (88 params). Closest thing to "tokenizing," but it's a standard categorical embedding.

Parameter distribution:

| Component | Params | Note |
|---|---:|---|
| Temporal VSN (per-feature GRN ×69) | 3.67M | 67% — the cost of per-feature interpretability |
| Static VSN | 637K | |
| LSTM encoder (2 layers) | 412K | |
| Static context GRNs ×4 + enrichment + output GRN | ~491K | |
| Attention + post-attn/post-lstm gating | ~168K | |
| Dual output heads (160→128→64→1, ×2) | ~58K | P(win) sigmoid + entry-offset linear |
| Sector embedding | 88 | |

## Why this architecture

1. **Variable Selection Networks → interpretability + built-in feature pruning.** The biggest win given the data regime. A per-sample softmax gate on each feature lets the model down-weight noise among the 80 inputs (many recently-added domain priors) and exposes *which* features drive predictions (`get_temporal_selection_weights`). With only ~750–2,500 events/year, automatic feature selection is the main defense against overfitting 80 inputs.
2. **Static covariates condition the temporal read.** Slow context (float, sector, gap %, day-of-run) is encoded into four context vectors that seed the LSTM state, bias the temporal VSN, and enrich every timestep — so a 2M-float 40% gap on day 3 of a run is interpreted differently from a 50M-float 25% gap, structurally rather than by hope.
3. **Local + global temporal modeling.** LSTM captures short-range bar-to-bar momentum; interpretable multi-head attention captures longer-range structure and yields per-timestep attention weights (you can see *when* it looked).
4. **Dual head matches the decision.** P(win) and entry-offset (ATR-normalized) share one representation — quality and timing learned jointly.
5. **Right-sized for small, structured time-series.** TFT targets exactly this regime (modest, feature-rich, mixed static/temporal/categorical), not the giant-corpus regime that needs tokenization.

**Tradeoff:** the interpretability machinery (a GRN per feature) makes it heavier/slower than a plain LSTM or a gradient-boosted tree. On this little data a GBM on static + aggregate temporal features is a credible baseline. The bet: the intraday *sequence* carries signal a flattened model loses, and VSN/attention interpretability is worth paying for to trust and debug what's learned. Unproven until trained + walk-forward evaluated.

## Flow matching — decision (2026-05-30)

**Verdict: not now, and not for the core predictor — but it's the right tool for the generative augmentation we already deferred, when/if we get there.**

Flow matching is generative (ODE-based transport of a base distribution to data); the TFT is discriminative. Three sub-questions:

1. **Replace the predictor head?** No. Modeling `p(win | features)` for a binary label + one scalar with a full generative apparatus is overkill; the TFT head (or a quantile / mixture-density head for uncertainty) does it directly and far cheaper.
2. **Generative data augmentation?** Yes — it's the modern, better instantiation of the TimeGAN/Diffusion-TS box the feature/aug spec punted to "phase 2." Conditional flow matching on gap-up bar sequences would synthesize events; vs alternatives it avoids GAN adversarial collapse (a real risk on tiny data) and is cheaper/more stable than diffusion (straight ODE paths, no long sampling chains). **Caveats:** a generative model needs *more* data than the discriminator it feeds → on this little data it tends to memorize/interpolate (leakage, false confidence); validating that synthetic sequences preserve the real feature→label relationship is hard; it's a whole second modeling project.
3. **Distributional outcome model?** Appealing for sizing/risk (EV, tail) but there's a cheaper 80% option — quantile-regression or mixture-density heads on the TFT. Reach for a full flow-matching outcome model only if those prove insufficient.

**The decisive factor is sequencing, not the technique** — there's no trained baseline yet, so flow matching now is premature optimization with nothing to measure against. Recommended order:

1. Train the TFT baseline + walk-forward eval (the actual open gap).
2. If data scarcity looks like the bottleneck (high variance / underfitting more data would fix), first test the cheap bar-level jitter/warp + mixup already built; measure the lift.
3. Only if that's not enough and synthetic data is clearly the constraint → generative augmentation, and there pick **flow matching** over GAN/diffusion.
4. If distributional outputs are needed for sizing/risk → quantile/mixture-density heads before a flow-matching outcome model.
