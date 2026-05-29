# Feature Expansion & Augmentation Redesign for the TFT Momentum Model

## Overview

The TFT momentum model has a complete architecture and data pipeline but **no
trained baseline yet**, and the qualifying-event dataset is thin (~3–10 names
per day → roughly 750–2,500 events/year). A 55-feature dual-head model on a
dataset that small sits in a high-variance regime where the dominant risk is
**overfitting**, not underfitting.

This spec covers two levers to improve prediction quality, both chosen because
they are *free* (no new paid data, no new AWS services) and grounded in domain
priors specific to low-float gap-up runners:

1. **Feature expansion** — add high-prior features derivable from data already
   in the warehouse (1-min bars, timestamps, Schwab daily bars, fundamentals
   already fetched, and the warehouse's own history). Keep all 55 existing
   features.
2. **Augmentation redesign (phase 1)** — replace the current per-feature noise
   with **bar-level augmentation that recomputes features and labels**, and add
   **mixup**. This makes augmented samples physically consistent and keeps the
   triple-barrier labels valid.

Generative augmentation (TimeGAN / Diffusion-TS) is explicitly **deferred to a
future phase 2** and is out of scope here.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature strategy | Add Tier A + Batch 2; keep all 55 existing | User directive: add-only. TFT's VSN can down-weight weak features; pruning is a separate follow-up. |
| Feature code path | Single `features.py` for train + live | Prevents train/serve skew; new features must be computed identically in both paths. |
| Augmentation locus | Raw OHLCV bars, upstream of feature engineering | Per-feature noise breaks the deterministic relationships among the 47 derived features; bar-level perturbation keeps them consistent. |
| Label handling under augmentation | Re-run `labeler.compute_labels` on perturbed bars | Window-slice / time-warp change the price path; recomputing keeps labels valid. |
| Mixup | Interpolate temporal + static-continuous + both labels | Labels are already continuous (`y_win`∈[0,1], `y_offset`∈ℝ), so mixup has no label-validity problem. |
| Augmented-sample usage | **Training folds only**, never validation | Prevents leakage of near-duplicates into the val set and keeps walk-forward eval honest. |
| Generative augmentation | Deferred (phase 2) | Chicken-and-egg (needs data to train the generator) and heavy; revisit after baseline. |

---

## Part 1 — Feature Expansion

### Principle

All new features flow through `tft/features.py` so the live inference path
(`build_feature_matrix`) produces them identically. New features that require
inputs not currently passed (prior-day bars, intraday volume profile, day-of-run)
must be added to the `static_data` contract and supplied by **both**
`data/assemble.py` (training) and the live loop (inference).

### Tier A features

| # | Feature(s) | Group | Definition | New input required |
|---|-----------|-------|------------|--------------------|
| A1 | `tod_sin`, `tod_cos` | 1-min temporal | Cyclical encoding of minutes-since-09:30 over the 390-min session. | bar timestamps (already indexed) |
| A2 | `float_rotation` | 1-min temporal | Cumulative session volume ÷ float_shares. | float_shares (have) |
| A3 | `pm_high_dist_atr`, `pm_low_dist_atr`, `broke_pm_high` | 1-min temporal | (close − pm_high)/ATR, (close − pm_low)/ATR, and a 0/1 flag for close > pm_high. | premarket_high/low (have, currently static) |
| A4 | `day_of_run`, `gap_vs_prior_range` | static continuous | Integer session number of the current run (1,2,3…); gap size ÷ prior-day true range. | prior-day daily bars; multi-day run detection |
| A5 | `round_number_dist_atr` | 1-min temporal | Distance from close to nearest whole/half dollar, in ATR units. | none |
| A6 | `log_dollar_volume` | 1-min temporal | log(1 + close × volume) per bar. | none |
| A7 | `pullback_depth_atr`, `higher_low_count` | 1-min temporal | (session_high − close)/ATR; count of higher-lows over the window. | none |

### Batch 2 features

| # | Feature(s) | Group | Definition | New input required |
|---|-----------|-------|------------|--------------------|
| B1 | `or_high_dist_atr`, `or_break_flag` | 1-min temporal | Distance to the first-15-min opening-range high in ATR; 0/1 break flag. | none (derived intra-window) |
| B2 | `intraday_rvol` | 1-min temporal | Bar volume ÷ the *typical* volume for that minute-of-day (warehouse profile). | intraday volume-profile artifact |
| B3 | `anchored_vwap_dist_atr` | 1-min temporal | (close − VWAP anchored at session open)/ATR. | none |
| B4 | `new_hod_flag`, `bars_since_hod` | 1-min temporal | 1 if bar set a new high-of-day; bars since the last high-of-day. | none |
| B5 | `prior_close_dist_atr`, `prior_high_dist_atr` | 1-min temporal | (close − prior_day_close)/ATR; (close − prior_day_high)/ATR. | prior-day daily bars |
| B6 | `gap_fill_progress` | 1-min temporal | Fraction of the open gap (open − prior_close) retraced by current close. | prior_day_close |
| B7 | `price_accel`, `volume_accel` | 1-min temporal | 2nd difference of close; 2nd difference of volume. | none |
| B8 | `ema_overextension_atr` | 1-min temporal | (close − EMA20)/ATR. | none |
| B9 | `dow_sin`, `dow_cos` | static continuous | Cyclical encoding of day-of-week. | session date (have) |

### Resulting feature counts

| Group | Before | New | After |
|-------|--------|-----|-------|
| 1-min temporal | 38 | +22 | 60 |
| 5-min temporal | 9 | 0 | 9 |
| Temporal total | 47 | +22 | **69** |
| Static continuous | 7 | +4 | **11** |
| Static categorical | 1 | 0 | 1 |

> **Top risk — feature-count growth.** Going from 47→69 temporal features on a
> thin dataset increases overfitting pressure. This is accepted per the
> add-only directive. **Immediate recommended follow-up (separate effort):**
> after a baseline exists, use `tft/interpret.py` variable-selection weights to
> prune dead-weight features back to a disciplined core.

### New data-contract inputs

These must be supplied by `assemble.py` (training) **and** the live loop
(inference), via `static_data`:

- **Prior-day daily bar** (`prior_day_close`, `prior_day_high`,
  `prior_day_range`) — from Schwab daily bars (free).
- **Day-of-run** — detect whether the prior N sessions were also qualifying
  gap/run days for the symbol.
- **Intraday volume profile** — a small precomputed artifact mapping
  minute-of-day → typical volume, computed **once** from warehouse history and
  cached (see Cost notes).

---

## Part 2 — Augmentation Redesign (Phase 1)

### Current state (to be replaced)

`TFTDataset.__getitem__` perturbs the final `(30, 47)` tensor with independent
Gaussian noise, a time-roll, and feature dropout. Independent per-feature noise
breaks the deterministic relationships among derived features (e.g. RSI vs the
price it came from), producing samples that cannot occur in reality.

### New design

Add a pure module `tft/augment.py` with two layers of augmentation:

**1. Bar-level transforms (operate on an OHLCV DataFrame):**

- `jitter_bars(bars, sigma=0.01)` — multiplicative Gaussian noise on OHLC
  (volume scaled proportionally), magnitude per the financial-augmentation
  literature.
- `window_slice_bars(bars, frac∈[0.4,0.8])` — take a contiguous sub-window and
  interpolate back to full length ("magnify" — a top performer for financial
  series).
- `time_warp_bars(bars, n_knots, sigma)` — cubic-spline time distortion (a top
  performer for financial series).
- **Never** time-reverse (empirically harmful; destroys causal momentum
  direction).

**Pipeline per augmented sample:** perturb bars → `build_feature_matrix` →
`labeler.compute_labels` on the perturbed bars → a fully consistent
(features, labels) pair.

**2. Mixup (operates on batched tensors, in the training loop):**

- `mixup_batch(batch, alpha≈0.3)` — sample λ~Beta(α,α); interpolate `temporal`,
  `static_continuous`, `y_win`, `y_offset`, and `weight` between shuffled pairs.
- `static_categorical` (sector) is **not** interpolated — take the category of
  the dominant (λ>0.5) sample.

### Where augmentation runs (leakage-safe)

Augmentation must never let a near-duplicate of a training sample land in
validation. Rule: **augmented samples are training-only.**

- Walk-forward folds are computed on **original samples only**
  (`create_walk_forward_folds` unchanged).
- For each fold, the **train** set = original train samples + bar-level
  augmented variants generated from *those* train samples' raw bars
  (K variants each, K≈3–5). The **val** set = original val samples only.
- **Mixup** is applied per-batch inside the training loop only.
- **Normalization stats** are fit on the **original** train samples only (not
  augmented copies), then applied to augmented + val — preserving the existing
  no-lookahead guarantee.

This requires raw bars (or a re-generatable handle) to be available at
train-assembly time for the augmentation step. Design choice: generate
augmented variants during fold setup in `train.py` (or a helper in
`assemble.py`) keyed by parent sample, tagged so they are excluded from val.

### Changes to existing modules

- `tft/dataset.py` — remove the independent-noise and time-roll augmentation.
  Feature-dropout becomes optional and **defaults off** (redundant with model
  dropout). `TFTDataset` becomes a plain consumer of tensors.
- `tft/train.py` — generate train-only augmented variants per fold; apply
  `mixup_batch` per training batch; fit normalization on originals only.
- `tft/augment.py` — **new** module (bar transforms + mixup).
- `data/assemble.py` — supply new `static_data` inputs; optionally expose a
  per-event raw-bars handle for augmentation.
- `tft/features.py` — implement all new features + name constants; update
  temporal/static dimensions.
- Model instantiation — update temporal input dim (47→69) and static-continuous
  dim (7→11) wherever `TemporalFusionTransformer` is constructed.
- Live inference path — supply the new `static_data` inputs so live features
  match training exactly.

---

## Testing (TDD)

Per the project's established TDD workflow, each unit gets tests before
implementation:

- **Features:** per new feature — output shape, a known-value case, and NaN/zero
  guards (warmup periods, zero ATR, zero float).
- **Bar transforms:** shape preservation; `window_slice`/`time_warp` produce a
  *different but plausible* path; labels recomputed on perturbed bars differ
  appropriately and stay in valid ranges (`y_win`∈[0,1]).
- **Mixup:** interpolation math (λ correctness); categorical taken from dominant
  sample; output shapes unchanged.
- **Leakage guard:** assert no augmented sample appears in any validation fold;
  assert normalization stats are computed without augmented copies.
- **Shape propagation:** the model forward pass accepts the new 69/11 dims.
- **Train/serve parity:** the live feature vector for a fixed input equals the
  training feature vector (single-code-path guard).

## Cost notes (AWS — budget discipline)

- **Intraday volume profile** is computed **once** from warehouse history and
  cached (small JSON/Parquet artifact). Do **not** recompute per-event or
  per-date (that would be the known O(n²) full-table-read hotspot). Compute per
  minute-of-day across symbols once, reuse.
- Bar-level augmentation runs **locally at train time** — no S3 cost; it reads
  the same bars already pulled for assembly.
- No new AWS services. No SageMaker. Generative augmentation (which would add
  real compute/complexity) stays deferred.

## Out of scope

- Generative augmentation (TimeGAN, Diffusion-TS) — future phase 2.
- Recency / time-decay weighting and peer/sympathy features — dropped.
- Tier B/C features (LULD halts, EDGAR dilution flags, Level 2, cost-to-borrow).
- Feature pruning via `interpret.py` — recommended immediate follow-up, but a
  separate effort gated on having a baseline.

## Open dependency

None of these changes are *verifiable* until a baseline is trained and run
through walk-forward evaluation. Augmentation's payoff in particular is
robustness, observable only by measuring. Establishing that first baseline is
the natural precursor to merging this work.
