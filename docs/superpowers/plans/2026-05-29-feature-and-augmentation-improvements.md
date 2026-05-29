# Feature Expansion & Augmentation Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~22 domain-prior temporal features and 4 static features to the TFT pipeline (keeping all 55 existing), and replace per-feature augmentation with leakage-safe bar-level augmentation (recomputing features AND labels) plus mixup.

**Architecture:** All features flow through the single `tft/features.py` code path (train + live parity). New feature inputs are **optional with neutral fallbacks**, mirroring the existing `spy_bars`/`vix_level` pattern, so callers that don't supply them (e.g. the live path, test fixtures) still work. Augmentation moves upstream: bar-level transforms perturb raw OHLCV → `build_feature_matrix` + `labeler.compute_labels` recompute consistent (features, labels); augmented samples are **training-only** and excluded from validation folds; normalization is fit on originals only; mixup is applied per training batch.

**Tech Stack:** Python, NumPy, pandas, PyTorch, pytest. Worktree: `.worktrees/tft-implementation`, branch `feature/tft-implementation`. All commands run from the worktree root.

**Conventions:**
- New temporal feature names are **appended to the END** of `TEMPORAL_1MIN_FEATURE_NAMES` so existing feature indices (0–37) never shift.
- New static names appended to the END of `STATIC_CONTINUOUS_FEATURE_NAMES`.
- Run tests with `python -m pytest` from the worktree root.
- Commit after each task.

**Final dimensions:** temporal 47 → **69** (1-min 38 → 60, 5-min unchanged at 9); static continuous 7 → **11**; static categorical unchanged at 1.

---

## Part 1 — Feature Expansion

### Task 1: Timestamp threading + time/calendar features

Adds `tod_sin`, `tod_cos` (1-min temporal) and `dow_sin`, `dow_cos` (static). Establishes the timestamp-passing pattern the later tasks reuse.

**Files:**
- Modify: `ML_tradingAlgo/tft/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features.py`:

```python
class TestTimeFeatures:
    def test_tod_features_present_and_bounded(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=2_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        assert feats.shape[1] == len(TEMPORAL_1MIN_FEATURE_NAMES)
        i_sin = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_sin")
        i_cos = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_cos")
        assert np.all(np.abs(feats[:, i_sin]) <= 1.0 + 1e-6)
        assert np.all(np.abs(feats[:, i_cos]) <= 1.0 + 1e-6)

    def test_tod_uses_datetime_index_when_present(self):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        idx = pd.date_range("2026-05-29 09:30", periods=20, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({
            "open": np.linspace(5, 5.2, 20), "high": np.linspace(5.05, 5.25, 20),
            "low": np.linspace(4.95, 5.15, 20), "close": np.linspace(5, 5.2, 20),
            "volume": np.full(20, 10000),
        }, index=idx)
        feats = compute_temporal_features_1min(
            bars, float_shares=2_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i_sin = TEMPORAL_1MIN_FEATURE_NAMES.index("tod_sin")
        # First bar at 09:30 -> minute 0 -> sin(0) == 0.
        assert abs(feats[0, i_sin]) < 1e-6

    def test_dow_features_present(self, sample_static_data):
        from ML_tradingAlgo.tft.features import (
            compute_static_features, STATIC_CONTINUOUS_FEATURE_NAMES,
        )
        data = dict(sample_static_data)
        data["session_date"] = "2026-05-29"  # a Friday (weekday 4)
        cont, cat = compute_static_features(**data)
        assert cont.shape[0] == len(STATIC_CONTINUOUS_FEATURE_NAMES)
        i_sin = STATIC_CONTINUOUS_FEATURE_NAMES.index("dow_sin")
        assert -1.0 - 1e-6 <= cont[i_sin] <= 1.0 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py::TestTimeFeatures -v`
Expected: FAIL — `"tod_sin" not in list` / shape mismatch.

- [ ] **Step 3: Implement**

In `ML_tradingAlgo/tft/features.py`, append to `TEMPORAL_1MIN_FEATURE_NAMES` (after `"candle_pressure_squared"`):

```python
    # Time encoding (2)
    "tod_sin",
    "tod_cos",
```

Append to `STATIC_CONTINUOUS_FEATURE_NAMES` (after `"premarket_range"`):

```python
    "dow_sin",
    "dow_cos",
```

In `compute_temporal_features_1min`, immediately after `df = bars.copy().reset_index(drop=True)` capture the timeline (before the reset loses it). Replace:

```python
    df = bars.copy().reset_index(drop=True)
    n = len(df)
```

with:

```python
    # Capture a minute-of-day timeline from the DatetimeIndex if present;
    # otherwise fall back to bar position (assume 1-min spacing from the open).
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
        minute_of_day = (idx.hour * 60 + idx.minute - (9 * 60 + 30)).to_numpy(dtype=float)
        minute_of_day = np.clip(minute_of_day, 0, 389)
    else:
        minute_of_day = np.arange(len(bars), dtype=float)

    df = bars.copy().reset_index(drop=True)
    n = len(df)
    minute_of_day = minute_of_day[:n]
```

Just before the `features = pd.DataFrame({...})` assembly block, add:

```python
    # --- Time encoding (2) ---
    tod_angle = 2.0 * np.pi * (minute_of_day / 390.0)
    tod_sin = pd.Series(np.sin(tod_angle))
    tod_cos = pd.Series(np.cos(tod_angle))
```

Add these two entries to the `features = pd.DataFrame({...})` dict (use the names so order follows the list):

```python
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
```

In `compute_static_features`, add a `session_date=None` parameter (place it among the keyword params, before `**kwargs`). Before building the `continuous` array, add:

```python
    # day-of-week cyclical encoding (neutral 0,0 when unavailable)
    dow_sin = 0.0
    dow_cos = 0.0
    if session_date is not None:
        d = pd.Timestamp(session_date)
        if not pd.isna(d):
            angle = 2.0 * np.pi * (d.weekday() / 7.0)
            dow_sin = float(np.sin(angle))
            dow_cos = float(np.cos(angle))
```

Append `dow_sin, dow_cos` to the `continuous = np.array([...])` list (after `premarket_range`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py::TestTimeFeatures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py tests/test_features.py
git commit -m "feat(features): add time-of-day and day-of-week encodings"
```

---

### Task 2: Volume & liquidity temporal features + intraday volume profile

Adds `float_rotation`, `log_dollar_volume`, `intraday_rvol`, plus a pure `build_intraday_volume_profile` helper.

**Files:**
- Modify: `ML_tradingAlgo/tft/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestVolumeFeatures:
    def test_float_rotation_monotonic(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("float_rotation")
        col = feats[:, i]
        assert np.all(np.diff(col) >= -1e-6)  # cumulative -> non-decreasing
        assert col[-1] > 0

    def test_log_dollar_volume_present(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("log_dollar_volume")
        assert np.all(feats[:, i] > 0)

    def test_intraday_rvol_profile_and_fallback(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("intraday_rvol")
        # fallback path (no profile) must still produce finite values
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        assert np.all(np.isfinite(feats[:, i]))
        # profile path: a profile of all-1.0 expected volume -> rvol == volume
        prof = np.ones(390)
        feats2 = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            intraday_volume_profile=prof,
        )
        assert np.all(np.isfinite(feats2[:, i]))

    def test_build_intraday_volume_profile(self):
        from ML_tradingAlgo.tft.features import build_intraday_volume_profile
        idx = pd.date_range("2026-05-29 09:30", periods=10, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({"volume": np.arange(10) + 1.0}, index=idx)
        prof = build_intraday_volume_profile([bars])
        assert len(prof) == 390
        assert prof[0] == 1.0  # minute 0 -> first bar volume
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py::TestVolumeFeatures -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `TEMPORAL_1MIN_FEATURE_NAMES`:

```python
    # Volume / liquidity (3)
    "float_rotation",
    "log_dollar_volume",
    "intraday_rvol",
```

Add a `intraday_volume_profile=None` keyword parameter to `compute_temporal_features_1min` (after `sector_etf_return`). In the body, after the existing volume block, add:

```python
    # --- Volume / liquidity (3) ---
    float_safe_v = float_shares if float_shares and float_shares != 0 else 1e-8
    float_rotation = v.cumsum().astype(float) / float_safe_v
    log_dollar_volume = np.log1p((c * v).clip(lower=0).astype(float))

    if intraday_volume_profile is not None:
        prof = np.asarray(intraday_volume_profile, dtype=float)
        exp_vol = np.array([
            prof[int(m)] if 0 <= int(m) < len(prof) and prof[int(m)] > 0 else avg_vol_per_min
            for m in minute_of_day
        ], dtype=float)
        intraday_rvol = pd.Series(v.to_numpy(dtype=float) / np.where(exp_vol > 0, exp_vol, 1.0))
    else:
        intraday_rvol = rel_vol.copy()
```

Add to the assembly dict:

```python
        "float_rotation": pd.Series(float_rotation),
        "log_dollar_volume": pd.Series(log_dollar_volume),
        "intraday_rvol": intraday_rvol.reset_index(drop=True),
```

Add the pure builder at module level (near the other helpers):

```python
def build_intraday_volume_profile(bars_list, session_minutes: int = 390) -> np.ndarray:
    """Typical (mean) volume per minute-of-day across many sessions.

    Each element of ``bars_list`` is a 1-min OHLCV frame with a DatetimeIndex.
    Returns a length-``session_minutes`` array; minutes with no observations
    fall back to the global mean (or 1.0 if empty).
    """
    sums = np.zeros(session_minutes, dtype=float)
    counts = np.zeros(session_minutes, dtype=float)
    for bars in bars_list:
        if not isinstance(bars.index, pd.DatetimeIndex):
            continue
        mins = (bars.index.hour * 60 + bars.index.minute - (9 * 60 + 30)).to_numpy()
        vols = bars["volume"].to_numpy(dtype=float)
        for m, vol in zip(mins, vols):
            if 0 <= m < session_minutes:
                sums[m] += vol
                counts[m] += 1
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    global_mean = np.nanmean(profile) if np.any(counts > 0) else 1.0
    return np.where(np.isnan(profile), global_mean, profile)
```

Thread `intraday_volume_profile` through `build_feature_matrix`: in the `compute_temporal_features_1min(...)` call inside `build_feature_matrix`, add the argument:

```python
        intraday_volume_profile=static_data.get("intraday_volume_profile"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py::TestVolumeFeatures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py tests/test_features.py
git commit -m "feat(features): add float rotation, dollar volume, intraday RVOL profile"
```

---

### Task 3: Price-level temporal features

Adds `pm_high_dist_atr`, `pm_low_dist_atr`, `broke_pm_high`, `round_number_dist_atr`, `or_high_dist_atr`, `or_break_flag`, `anchored_vwap_dist_atr`, `prior_close_dist_atr`, `prior_high_dist_atr`, `gap_fill_progress`, `ema_overextension_atr` (11 features).

**Files:**
- Modify: `ML_tradingAlgo/tft/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
class TestPriceLevelFeatures:
    NEW = [
        "pm_high_dist_atr", "pm_low_dist_atr", "broke_pm_high",
        "round_number_dist_atr", "or_high_dist_atr", "or_break_flag",
        "anchored_vwap_dist_atr", "prior_close_dist_atr",
        "prior_high_dist_atr", "gap_fill_progress", "ema_overextension_atr",
    ]

    def test_all_present_and_finite(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            premarket_high=5.3, premarket_low=4.7, prior_close=4.5,
            prior_day_high=5.0,
        )
        for name in self.NEW:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name

    def test_broke_pm_high_is_binary(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
            premarket_high=5.0,
        )
        i = TEMPORAL_1MIN_FEATURE_NAMES.index("broke_pm_high")
        assert set(np.unique(feats[:, i])).issubset({0.0, 1.0})

    def test_neutral_when_inputs_missing(self, sample_1min_bars):
        # No premarket/prior inputs -> features must still be finite (neutral).
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        for name in ["pm_high_dist_atr", "prior_close_dist_atr", "gap_fill_progress"]:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::TestPriceLevelFeatures -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `TEMPORAL_1MIN_FEATURE_NAMES`:

```python
    # Price levels (11)
    "pm_high_dist_atr",
    "pm_low_dist_atr",
    "broke_pm_high",
    "round_number_dist_atr",
    "or_high_dist_atr",
    "or_break_flag",
    "anchored_vwap_dist_atr",
    "prior_close_dist_atr",
    "prior_high_dist_atr",
    "gap_fill_progress",
    "ema_overextension_atr",
```

Add keyword params to `compute_temporal_features_1min`: `premarket_high=None, premarket_low=None, prior_close=None, prior_day_high=None` (after `intraday_volume_profile`). In the body, after the VWAP-dynamics block, add:

```python
    # --- Price levels (11) ---
    zeros = pd.Series(np.zeros(n))

    pm_high_dist = (c - premarket_high) / atr_safe if premarket_high is not None else zeros.copy()
    pm_low_dist = (c - premarket_low) / atr_safe if premarket_low is not None else zeros.copy()
    broke_pm_high = (c > premarket_high).astype(float) if premarket_high is not None else zeros.copy()

    nearest_half = (c / 0.5).round() * 0.5
    round_number_dist = (c - nearest_half) / atr_safe

    or_window = min(15, n)
    or_high_val = float(h.iloc[:or_window].max())
    or_high_dist = (c - or_high_val) / atr_safe
    or_break_flag = (c > or_high_val).astype(float)

    # Anchored VWAP from the first regular-hours bar (minute_of_day == 0),
    # else from data start. Falls back to the cumulative session VWAP.
    anchor_pos = int(np.argmax(minute_of_day >= 0)) if n else 0
    tp = (h + l + c) / 3
    cum_tp = (tp * v).cumsum()
    cum_v = v.cumsum().replace(0, np.nan)
    anchored_vwap = (cum_tp / cum_v).fillna(method="bfill").fillna(c)
    anchored_vwap_dist = (c - anchored_vwap) / atr_safe

    prior_close_dist = (c - prior_close) / atr_safe if prior_close is not None else zeros.copy()
    prior_high_dist = (c - prior_day_high) / atr_safe if prior_day_high is not None else zeros.copy()

    if prior_close is not None and n > 0:
        open0 = float(o.iloc[0])
        gap = open0 - prior_close
        gap_safe = gap if abs(gap) > 1e-8 else 1e-8
        gap_fill_progress = ((open0 - c) / gap_safe).clip(-1.0, 2.0)
    else:
        gap_fill_progress = zeros.copy()

    ema20 = _ema(c, 20)
    ema_overextension = (c - ema20) / atr_safe
```

Add to the assembly dict (order matches the names list):

```python
        "pm_high_dist_atr": pm_high_dist.reset_index(drop=True),
        "pm_low_dist_atr": pm_low_dist.reset_index(drop=True),
        "broke_pm_high": broke_pm_high.reset_index(drop=True),
        "round_number_dist_atr": round_number_dist.reset_index(drop=True),
        "or_high_dist_atr": or_high_dist.reset_index(drop=True),
        "or_break_flag": or_break_flag.reset_index(drop=True),
        "anchored_vwap_dist_atr": anchored_vwap_dist.reset_index(drop=True),
        "prior_close_dist_atr": prior_close_dist.reset_index(drop=True),
        "prior_high_dist_atr": prior_high_dist.reset_index(drop=True),
        "gap_fill_progress": gap_fill_progress.reset_index(drop=True),
        "ema_overextension_atr": ema_overextension.reset_index(drop=True),
```

Thread the four new inputs through `build_feature_matrix`'s `compute_temporal_features_1min(...)` call:

```python
        premarket_high=static_data.get("premarket_high"),
        premarket_low=static_data.get("premarket_low"),
        prior_close=static_data.get("prior_close"),
        prior_day_high=static_data.get("prior_day_high"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py::TestPriceLevelFeatures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py tests/test_features.py
git commit -m "feat(features): add price-level distance features (PM/OR/prior-day/gap-fill)"
```

---

### Task 4: Trend & structure temporal features

Adds `pullback_depth_atr`, `higher_low_count`, `new_hod_flag`, `bars_since_hod`, `price_accel`, `volume_accel` (6 features). This completes the 1-min temporal block at 60.

**Files:**
- Modify: `ML_tradingAlgo/tft/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
class TestStructureFeatures:
    def test_structure_features_present(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        for name in ["pullback_depth_atr", "higher_low_count", "new_hod_flag",
                     "bars_since_hod", "price_accel", "volume_accel"]:
            i = TEMPORAL_1MIN_FEATURE_NAMES.index(name)
            assert np.all(np.isfinite(feats[:, i])), name

    def test_pullback_nonnegative_and_hod_binary(self, sample_1min_bars):
        from ML_tradingAlgo.tft.features import (
            compute_temporal_features_1min, TEMPORAL_1MIN_FEATURE_NAMES,
        )
        feats = compute_temporal_features_1min(
            sample_1min_bars, float_shares=1_000_000, avg_volume_20d=500_000,
            spy_bars=None, vix_level=20.0, sector_etf_return=0.01,
        )
        i_pb = TEMPORAL_1MIN_FEATURE_NAMES.index("pullback_depth_atr")
        i_hod = TEMPORAL_1MIN_FEATURE_NAMES.index("new_hod_flag")
        assert np.all(feats[:, i_pb] >= -1e-6)
        assert set(np.unique(feats[:, i_hod])).issubset({0.0, 1.0})

    def test_total_1min_feature_count_is_60(self):
        from ML_tradingAlgo.tft.features import TEMPORAL_1MIN_FEATURE_NAMES
        assert len(TEMPORAL_1MIN_FEATURE_NAMES) == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::TestStructureFeatures -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `TEMPORAL_1MIN_FEATURE_NAMES`:

```python
    # Trend / structure (6)
    "pullback_depth_atr",
    "higher_low_count",
    "new_hod_flag",
    "bars_since_hod",
    "price_accel",
    "volume_accel",
```

In `compute_temporal_features_1min`, after the price-levels block, add:

```python
    # --- Trend / structure (6) ---
    hod = h.cummax()
    pullback_depth = (hod - c) / atr_safe

    higher_low = (l > l.shift(1)).astype(float).fillna(0)
    higher_low_count = higher_low.rolling(10, min_periods=1).sum()

    new_hod_flag = (h >= hod).astype(float)

    bars_since_hod = pd.Series(np.zeros(n), dtype=float)
    last_hod = 0
    new_hod_vals = new_hod_flag.to_numpy()
    for i in range(n):
        if new_hod_vals[i] > 0:
            last_hod = i
        bars_since_hod.iloc[i] = float(i - last_hod)

    price_accel = c.diff().diff().fillna(0)
    volume_accel = v.astype(float).diff().diff().fillna(0)
```

Add to the assembly dict:

```python
        "pullback_depth_atr": pullback_depth.reset_index(drop=True),
        "higher_low_count": higher_low_count.reset_index(drop=True),
        "new_hod_flag": new_hod_flag.reset_index(drop=True),
        "bars_since_hod": bars_since_hod,
        "price_accel": price_accel.reset_index(drop=True),
        "volume_accel": volume_accel.reset_index(drop=True),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py::TestStructureFeatures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py tests/test_features.py
git commit -m "feat(features): add trend/structure features (pullback, HOD, acceleration)"
```

---

### Task 5: Static features + assemble wiring

Adds `day_of_run`, `gap_vs_prior_range` (static); wires the new static inputs through `build_static_data`, and computes `day_of_run` in `assemble_dataset`.

**Files:**
- Modify: `ML_tradingAlgo/tft/features.py`, `ML_tradingAlgo/data/assemble.py`
- Test: `tests/test_features.py`, `tests/test_assemble.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_features.py`:

```python
class TestNewStaticFeatures:
    def test_static_count_is_11(self):
        from ML_tradingAlgo.tft.features import STATIC_CONTINUOUS_FEATURE_NAMES
        assert len(STATIC_CONTINUOUS_FEATURE_NAMES) == 11

    def test_day_of_run_and_gap_vs_range(self, sample_static_data):
        from ML_tradingAlgo.tft.features import (
            compute_static_features, STATIC_CONTINUOUS_FEATURE_NAMES,
        )
        data = dict(sample_static_data)
        data.update({"day_of_run": 2, "prior_day_range": 1.0,
                     "prior_day_high": 5.0, "session_date": "2026-05-29"})
        cont, _ = compute_static_features(**data)
        i_run = STATIC_CONTINUOUS_FEATURE_NAMES.index("day_of_run")
        i_gap = STATIC_CONTINUOUS_FEATURE_NAMES.index("gap_vs_prior_range")
        assert cont[i_run] == 2.0
        # gap_vs_prior_range = (current_price - prior_close)/prior_day_range
        assert abs(cont[i_gap] - (5.5 - 4.0) / 1.0) < 1e-6
```

In `tests/test_assemble.py` (append a focused test; reuse the file's existing synthetic-bars/event fixtures style):

```python
def test_build_static_data_includes_new_keys():
    from ML_tradingAlgo.data.assemble import build_static_data
    import pandas as pd
    bars = pd.DataFrame({
        "open": [5.0, 5.1], "high": [5.2, 5.2], "low": [4.9, 5.0],
        "close": [5.1, 5.15], "volume": [10000, 12000],
    })
    sd = build_static_data(
        event_row={"prior_close": 4.0, "float_shares": 1_000_000,
                   "gap_pct": 0.3, "session_date": "2026-05-29",
                   "premarket_high": 5.3, "premarket_low": 4.7,
                   "prior_day_high": 5.0, "prior_day_range": 1.2, "day_of_run": 1},
        bars_1min=bars, fundamentals_row=None, current_price=5.15,
    )
    for key in ["prior_day_high", "prior_day_range", "day_of_run",
                "session_date", "intraday_volume_profile"]:
        assert key in sd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py::TestNewStaticFeatures tests/test_assemble.py::test_build_static_data_includes_new_keys -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `ML_tradingAlgo/tft/features.py`, append to `STATIC_CONTINUOUS_FEATURE_NAMES` (after `dow_cos` from Task 1):

```python
    "day_of_run",
    "gap_vs_prior_range",
```

> Note: `STATIC_CONTINUOUS_FEATURE_NAMES` final order is: the original 7, then `dow_sin`, `dow_cos`, `day_of_run`, `gap_vs_prior_range` = 11.

Add params to `compute_static_features`: `day_of_run=1, prior_day_range=None, prior_day_high=None` (alongside the others; `**kwargs` already swallows anything else). Before assembling `continuous`, add:

```python
    day_of_run_val = float(day_of_run) if day_of_run is not None else 1.0
    if prior_day_range and prior_day_range > 0:
        gap_vs_prior_range = (current_price - prior_close) / prior_day_range
    else:
        gap_vs_prior_range = 0.0
```

Append to the `continuous = np.array([...])` list, AFTER `dow_sin, dow_cos`:

```python
        day_of_run_val,
        gap_vs_prior_range,
```

In `ML_tradingAlgo/data/assemble.py`, in `build_static_data`, add to the returned dict (compute `prior_day_high`/`prior_day_range` from `event_row` with neutral fallbacks):

```python
        "session_date": event_row.get("session_date"),
        "prior_day_high": _finite(event_row.get("prior_day_high"), default=prior_close),
        "prior_day_range": _finite(event_row.get("prior_day_range"), default=0.0),
        "day_of_run": int(_finite(event_row.get("day_of_run"), default=1)),
        "intraday_volume_profile": event_row.get("intraday_volume_profile"),
```

In `assemble_dataset`, compute `day_of_run` per symbol before the event loop (consecutive prior trading days). Right after `events` is obtained and confirmed non-empty, insert:

```python
    # day_of_run: count consecutive prior session_dates for the same symbol.
    _run_map = {}
    if events is not None and len(events):
        ev = events.copy()
        ev["_d"] = [_to_date(x) for x in ev["session_date"]]
        for sym, grp in ev.groupby("symbol"):
            dates = sorted(d for d in grp["_d"] if d is not None)
            run = 0
            prev = None
            for d in dates:
                if prev is not None and 0 < (d - prev).days <= 4:
                    run += 1
                else:
                    run = 1
                _run_map[(sym, d)] = run
                prev = d
```

Then inside the event loop, after `event_row = event.to_dict()`, inject the run count:

```python
        event_row["day_of_run"] = _run_map.get(
            (event_row.get("symbol"), _to_date(event_row.get("session_date"))), 1
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py::TestNewStaticFeatures tests/test_assemble.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/features.py ML_tradingAlgo/data/assemble.py tests/test_features.py tests/test_assemble.py
git commit -m "feat(features): add day-of-run and gap-vs-prior-range static features"
```

---

### Task 6: Dimension propagation (47→69, 7→11)

Update every place that hardcodes the old dims, and add an end-to-end shape test through the model.

**Files:**
- Modify: `ML_tradingAlgo/data/assemble.py` (`_empty_dataset`, docstrings), `tests/conftest.py`, `ML_tradingAlgo/tft/dataset.py` (docstrings), `ML_tradingAlgo/tft_predictor.py` (docstrings **and** a functional dimension guard), `ML_tradingAlgo/tft/features.py` (`build_feature_matrix` docstring)
- Test: `tests/test_features.py`, `tests/test_tft_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
class TestFeatureMatrixDims:
    def test_build_feature_matrix_shapes(self, sample_1min_bars, sample_5min_bars, sample_static_data):
        from ML_tradingAlgo.tft.features import build_feature_matrix
        data = dict(sample_static_data)
        data.update({"session_date": "2026-05-29", "prior_day_high": 6.0,
                     "prior_day_range": 1.5, "day_of_run": 1,
                     "intraday_volume_profile": None})
        temporal, cont, cat = build_feature_matrix(
            sample_1min_bars, sample_5min_bars, data, sequence_length=30,
        )
        assert temporal.shape == (30, 69)
        assert cont.shape == (11,)
        assert cat.shape == (1,)

    def test_model_forward_with_new_dims(self):
        import torch
        from ML_tradingAlgo.tft.model import TemporalFusionTransformer
        cfg = {
            "hidden_size": 32, "lstm_layers": 1, "attention_heads": 2,
            "dropout": 0.1, "num_temporal_features": 69,
            "num_static_continuous": 11, "num_static_categorical": 1,
            "categorical_cardinalities": [11], "categorical_embedding_dim": 8,
            "sequence_length": 30,
        }
        m = TemporalFusionTransformer(**cfg)
        p, off, attn = m(torch.randn(2, 30, 69), torch.randn(2, 11),
                         torch.randint(0, 11, (2, 1)))
        assert p.shape[0] == 2 and off.shape[0] == 2
```

Add to `tests/test_tft_predictor.py` (builds a tiny on-disk checkpoint at the new dims and exercises the guard):

```python
def test_predictor_accepts_69_11_and_rejects_mismatch(tmp_path):
    import numpy as np, torch
    from ML_tradingAlgo.tft.model import TemporalFusionTransformer
    from ML_tradingAlgo.tft_predictor import TFTPredictor

    cfg = {
        "hidden_size": 32, "lstm_layers": 1, "attention_heads": 2, "dropout": 0.1,
        "num_temporal_features": 69, "num_static_continuous": 11,
        "num_static_categorical": 1, "categorical_cardinalities": [11],
        "categorical_embedding_dim": 8, "sequence_length": 30,
    }
    torch.save(TemporalFusionTransformer(**cfg).state_dict(), tmp_path / "model.pt")
    np.savez(
        tmp_path / "norm_stats.npz",
        temporal_mean=np.zeros(69, "float32"), temporal_std=np.ones(69, "float32"),
        static_mean=np.zeros(11, "float32"), static_std=np.ones(11, "float32"),
    )
    pred = TFTPredictor(tmp_path, cfg)

    p_win, offset = pred.predict(
        np.zeros((30, 69), "float32"), np.zeros(11, "float32"), np.zeros(1, "int64")
    )
    assert 0.0 <= p_win <= 1.0

    # Stale 47-wide input must fail loudly, not silently mis-broadcast.
    with pytest.raises(ValueError):
        pred.predict(np.zeros((30, 47), "float32"), np.zeros(11, "float32"),
                     np.zeros(1, "int64"))
```

> Ensure `tests/test_tft_predictor.py` imports `pytest` and `numpy` at the top (add if missing).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::TestFeatureMatrixDims -v`
Expected: FAIL on the `(30, 69)` / `(11,)` assertions (until prior tasks are merged) — if prior tasks are done, this should already pass for `build_feature_matrix`; the failing piece is the hardcoded dims below.

- [ ] **Step 3: Implement**

In `ML_tradingAlgo/data/assemble.py`, `_empty_dataset` — change the hardcoded shapes:

```python
        "temporal": np.empty((0, sequence_length, 69), dtype=np.float32),
        "static_continuous": np.empty((0, 11), dtype=np.float32),
```

Update the `assemble_event` / `assemble_dataset` docstrings that say `(sequence_length, 47)`, `(7,)`, `(N, 30, 47)`, `(N, 7)` to `69` / `11` accordingly.

In `tests/conftest.py`, update the `model_config` fixture:

```python
        "num_temporal_features": 69,
        "num_static_continuous": 11,
```

In `ML_tradingAlgo/tft/dataset.py`, update the module/`compute_normalization_stats`/class docstrings that reference `(30, 47)`, `(7,)`, `(N, 30, 47)`, `(N, 7)` → `(30, 69)`, `(11,)`, `(N, 30, 69)`, `(N, 11)`.

In `ML_tradingAlgo/tft_predictor.py`:

1. Update the `predict` docstring shapes `(30, 47)` → `(30, 69)` and `(7,)` → `(11,)`.
2. Add a functional dimension guard at the top of `predict`, immediately after the three `np.asarray(...)` coercions and **before** the z-score normalization, so a stale-width input fails loudly instead of NumPy broadcasting against the wrong-length `norm_stats`:

```python
        exp_temporal = self.norm_stats["temporal_mean"].shape[0]
        exp_static = self.norm_stats["static_mean"].shape[0]
        if sequence.shape[-1] != exp_temporal:
            raise ValueError(
                f"temporal feature width {sequence.shape[-1]} != expected {exp_temporal} "
                f"(model/norm_stats). Live feature builder is out of sync with the model."
            )
        if static_continuous.shape[-1] != exp_static:
            raise ValueError(
                f"static_continuous width {static_continuous.shape[-1]} != expected {exp_static}."
            )
```

> This makes `tft_predictor.py` self-defending: the predictor stays parameterized by `model_config` + `norm_stats` (which auto-size to 69/11 after retraining), and now any caller still emitting the old 47-wide vector gets a clear error rather than silent garbage. The live feature builder must call `build_feature_matrix` with the new `static_data` inputs to produce a 69-wide sequence.

In `ML_tradingAlgo/tft/features.py`, update the `build_feature_matrix` docstring (`(sequence_length, 47)`, `static_continuous: (7,)`, and the module header "all 55 features (8 static + 38 one-min + 9 five-min)") to reflect 69 temporal / 11 static-continuous.

- [ ] **Step 4: Run the full feature/model/assemble suites**

Run: `python -m pytest tests/test_features.py tests/test_assemble.py tests/test_model.py tests/test_train.py tests/test_tft_predictor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo tests/conftest.py tests/test_features.py tests/test_tft_predictor.py
git commit -m "chore(tft): propagate feature dims 47->69, 7->11; add predictor dim guard"
```

---

## Part 2 — Augmentation Redesign

### Task 7: Bar-level transforms (`tft/augment.py`)

Length-preserving, index-preserving OHLCV transforms: `jitter_bars`, `window_slice_bars`, `time_warp_bars`.

**Files:**
- Create: `ML_tradingAlgo/tft/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_augment.py`:

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bars():
    idx = pd.date_range("2026-05-29 09:30", periods=40, freq="1min", tz="US/Eastern")
    base = 5.0 + np.cumsum(np.random.RandomState(0).randn(40) * 0.02)
    return pd.DataFrame({
        "open": base, "high": base + 0.03, "low": base - 0.03,
        "close": base + 0.01, "volume": np.full(40, 10000.0),
    }, index=idx)


def _ohlc_valid(df):
    return (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all() and \
           (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()


class TestBarTransforms:
    def test_jitter_preserves_shape_index_and_ohlc(self, bars):
        from ML_tradingAlgo.tft.augment import jitter_bars
        out = jitter_bars(bars, sigma=0.01, rng=np.random.RandomState(1))
        assert list(out.index) == list(bars.index)
        assert out.shape == bars.shape
        assert _ohlc_valid(out)
        assert not np.allclose(out["close"].to_numpy(), bars["close"].to_numpy())

    def test_window_slice_preserves_length(self, bars):
        from ML_tradingAlgo.tft.augment import window_slice_bars
        out = window_slice_bars(bars, frac=0.6, rng=np.random.RandomState(2))
        assert len(out) == len(bars)
        assert list(out.index) == list(bars.index)
        assert _ohlc_valid(out)

    def test_time_warp_preserves_length(self, bars):
        from ML_tradingAlgo.tft.augment import time_warp_bars
        out = time_warp_bars(bars, n_knots=4, sigma=0.2, rng=np.random.RandomState(3))
        assert len(out) == len(bars)
        assert list(out.index) == list(bars.index)
        assert _ohlc_valid(out)

    def test_no_time_reverse_helper_exists(self):
        # Guard: reversal is empirically harmful; it must NOT be offered.
        import ML_tradingAlgo.tft.augment as aug
        assert not hasattr(aug, "reverse_bars")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_augment.py::TestBarTransforms -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement**

Create `ML_tradingAlgo/tft/augment.py`:

```python
"""Bar-level data augmentation for the TFT momentum pipeline.

Transforms operate on a 1-min OHLCV DataFrame and are **length- and
index-preserving** so that downstream feature engineering and triple-barrier
labeling can be re-run on the perturbed bars without changing the entry-bar
index. Reversal is deliberately NOT provided (empirically harmful for causal
momentum series). ``mixup_batch`` operates on already-batched tensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = ["open", "high", "low", "close"]


def _rng(rng):
    return rng if rng is not None else np.random.RandomState()


def _repair_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Re-impose high >= max(o,c) and low <= min(o,c)."""
    hi = df[["open", "close", "high"]].max(axis=1)
    lo = df[["open", "close", "low"]].min(axis=1)
    df["high"] = hi
    df["low"] = lo
    df["volume"] = df["volume"].clip(lower=0)
    return df


def jitter_bars(bars: pd.DataFrame, sigma: float = 0.01, rng=None) -> pd.DataFrame:
    """Multiplicative Gaussian noise on OHLC (and volume), per the financial
    augmentation literature (sigma ~ 0.01)."""
    rng = _rng(rng)
    out = bars.copy()
    noise = 1.0 + rng.randn(len(bars), len(_OHLC)) * sigma
    out[_OHLC] = bars[_OHLC].to_numpy() * noise
    vol_noise = 1.0 + rng.randn(len(bars)) * sigma
    out["volume"] = (bars["volume"].to_numpy() * np.clip(vol_noise, 0.0, None))
    return _repair_ohlc(out)


def window_slice_bars(bars: pd.DataFrame, frac: float = 0.6, rng=None) -> pd.DataFrame:
    """"Magnify": take a contiguous sub-window of length int(n*frac) and
    interpolate each column back to the original length."""
    rng = _rng(rng)
    n = len(bars)
    w = max(2, int(round(n * frac)))
    start = int(rng.randint(0, max(1, n - w + 1)))
    sub = bars.iloc[start:start + w]
    src = np.linspace(0.0, 1.0, len(sub))
    dst = np.linspace(0.0, 1.0, n)
    out = bars.copy()
    for col in _OHLC + ["volume"]:
        out[col] = np.interp(dst, src, sub[col].to_numpy(dtype=float))
    return _repair_ohlc(out)


def time_warp_bars(bars: pd.DataFrame, n_knots: int = 4, sigma: float = 0.2, rng=None) -> pd.DataFrame:
    """Monotonic time-axis warp via random knot offsets + linear resampling."""
    rng = _rng(rng)
    n = len(bars)
    knots = np.linspace(0.0, 1.0, n_knots + 2)
    offsets = np.concatenate([[0.0], rng.randn(n_knots) * sigma / n_knots, [0.0]])
    warped_knots = np.clip(knots + offsets, 0.0, 1.0)
    warped_knots = np.maximum.accumulate(warped_knots)  # enforce monotonicity
    base = np.linspace(0.0, 1.0, n)
    warped_time = np.interp(base, knots, warped_knots)
    out = bars.copy()
    for col in _OHLC + ["volume"]:
        out[col] = np.interp(warped_time, base, bars[col].to_numpy(dtype=float))
    return _repair_ohlc(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_augment.py::TestBarTransforms -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/augment.py tests/test_augment.py
git commit -m "feat(augment): add length-preserving bar-level transforms"
```

---

### Task 8: Mixup (`tft/augment.py`)

**Files:**
- Modify: `ML_tradingAlgo/tft/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestMixup:
    def _batch(self):
        import torch
        return {
            "temporal": torch.randn(8, 30, 69),
            "static_continuous": torch.randn(8, 11),
            "static_categorical": torch.randint(0, 11, (8, 1)),
            "y_win": torch.randint(0, 2, (8, 1)).float(),
            "y_offset": torch.randn(8, 1),
            "weight": torch.ones(8, 1),
        }

    def test_mixup_preserves_shapes_and_categorical_dtype(self):
        from ML_tradingAlgo.tft.augment import mixup_batch
        import torch
        out = mixup_batch(self._batch(), alpha=0.3, rng=np.random.RandomState(0))
        assert out["temporal"].shape == (8, 30, 69)
        assert out["static_continuous"].shape == (8, 11)
        assert out["static_categorical"].dtype == torch.int64
        assert out["y_win"].shape == (8, 1)

    def test_mixup_alpha_zero_is_identity(self):
        from ML_tradingAlgo.tft.augment import mixup_batch
        import torch
        b = self._batch()
        out = mixup_batch(b, alpha=0.0, rng=np.random.RandomState(0))
        assert torch.allclose(out["temporal"], b["temporal"])
        assert torch.allclose(out["y_win"], b["y_win"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_augment.py::TestMixup -v`
Expected: FAIL (`mixup_batch` undefined).

- [ ] **Step 3: Implement**

Append to `ML_tradingAlgo/tft/augment.py`:

```python
import torch  # noqa: E402  (kept local-friendly; torch only needed for mixup)


def mixup_batch(batch: dict, alpha: float = 0.3, rng=None) -> dict:
    """Mixup for the dual-head TFT.

    Interpolates ``temporal``, ``static_continuous``, ``y_win``, ``y_offset``
    and ``weight`` between the batch and a shuffled copy of itself using a
    single lambda ~ Beta(alpha, alpha). ``static_categorical`` is taken from the
    dominant sample (lambda >= 0.5 -> original order, else the shuffled order).
    ``alpha <= 0`` returns the batch unchanged.
    """
    if alpha is None or alpha <= 0:
        return batch
    rng = _rng(rng)
    lam = float(rng.beta(alpha, alpha))
    bsz = batch["temporal"].shape[0]
    perm = torch.as_tensor(rng.permutation(bsz), dtype=torch.long)

    out = dict(batch)
    for key in ("temporal", "static_continuous", "y_win", "y_offset", "weight"):
        if key in batch:
            out[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]
    if "static_categorical" in batch:
        out["static_categorical"] = (
            batch["static_categorical"] if lam >= 0.5 else batch["static_categorical"][perm]
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_augment.py::TestMixup -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/augment.py tests/test_augment.py
git commit -m "feat(augment): add mixup_batch for dual continuous heads"
```

---

### Task 9: Assemble refactor + bar-level augmented sample generation

Split bar-loading from sample-building (so augmentation reuses one S3 read — cost discipline), add a `bar_transform` hook, and extend `assemble_dataset` to emit train-only augmented variants tagged with `is_augmented` + `parent_index`.

**Files:**
- Modify: `ML_tradingAlgo/data/assemble.py`
- Test: `tests/test_assemble.py`

- [ ] **Step 1: Write the failing tests**

```python
def _synthetic_event_and_bars():
    import pandas as pd, numpy as np
    idx = pd.date_range("2026-05-29 09:30", periods=45, freq="1min", tz="US/Eastern")
    base = 5.0 + np.cumsum(np.random.RandomState(0).randn(45) * 0.02)
    bars = pd.DataFrame({
        "ts": idx, "open": base, "high": base + 0.05, "low": base - 0.05,
        "close": base + 0.01, "volume": np.full(45, 20000.0), "symbol": "TST",
    })
    event = {"symbol": "TST", "session_date": "2026-05-29", "float_shares": 1_000_000,
             "prior_close": 4.0, "gap_pct": 0.25, "premarket_high": 5.2,
             "premarket_low": 4.6}
    return event, bars


def test_bar_transform_changes_features_keeps_shape():
    from ML_tradingAlgo.data import assemble
    from ML_tradingAlgo.tft.augment import jitter_bars
    import numpy as np
    event, bars = _synthetic_event_and_bars()
    read = lambda *a, **k: bars.copy()
    base = assemble.assemble_event(event, read_bars=read)
    aug = assemble.assemble_event(
        event, read_bars=read,
        bar_transform=lambda df: jitter_bars(df, sigma=0.02, rng=np.random.RandomState(7)),
    )
    assert base["temporal"].shape == aug["temporal"].shape == (30, 69)
    assert not np.allclose(base["temporal"], aug["temporal"])


def test_assemble_dataset_augmented_are_tagged_train_only():
    from ML_tradingAlgo.data import assemble
    from ML_tradingAlgo.tft.augment import jitter_bars
    import numpy as np
    event, bars = _synthetic_event_and_bars()

    def fake_build_labeled_dataset(*a, **k):
        import pandas as pd
        return pd.DataFrame([event])

    # patch labeler.build_labeled_dataset via the read_bars indirection used in assemble
    read = lambda *a, **k: bars.copy()
    import ML_tradingAlgo.data.labeler as labeler
    orig = labeler.build_labeled_dataset
    labeler.build_labeled_dataset = fake_build_labeled_dataset
    try:
        out = assemble.assemble_dataset(
            ("2026-05-29", "2026-05-29"), read_bars=read,
            augment=True, n_augment=3,
            bar_transforms=[lambda df: jitter_bars(df, sigma=0.02, rng=np.random.RandomState(1))],
        )
    finally:
        labeler.build_labeled_dataset = orig

    assert "is_augmented" in out and "parent_index" in out
    n = len(out["is_augmented"])
    assert n == 4  # 1 original + 3 augmented
    assert out["is_augmented"].sum() == 3
    # every augmented row points to a valid original parent
    for j in np.where(out["is_augmented"])[0]:
        p = out["parent_index"][j]
        assert not out["is_augmented"][p]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_assemble.py -k "bar_transform or augmented" -v`
Expected: FAIL (`bar_transform` / `augment` params unknown).

- [ ] **Step 3: Implement**

In `ML_tradingAlgo/data/assemble.py`, refactor `_assemble_event_with_reason` to accept `bar_transform=None`. Apply it right after the DatetimeIndex is installed and before entry detection:

```python
    bars = bars.sort_values("ts")
    bars = bars.set_index(pd.DatetimeIndex(pd.to_datetime(bars["ts"])))

    if bar_transform is not None:
        transformed = bar_transform(bars[["open", "high", "low", "close", "volume"]])
        transformed.index = bars.index  # length-preserving contract
        for col in ["open", "high", "low", "close", "volume"]:
            bars[col] = transformed[col].to_numpy()
```

Add `bar_transform=None` to the signatures of both `_assemble_event_with_reason` and `assemble_event`, and forward it from `assemble_event`:

```python
def assemble_event(event_row, ..., spy_bars=None, bar_transform=None) -> dict | None:
    result = _assemble_event_with_reason(..., spy_bars=spy_bars, bar_transform=bar_transform)
    return result[0]
```

Extend `assemble_dataset` with `augment=False, n_augment=0, bar_transforms=None, augment_rng=None`. After an original sample is appended (inside the event loop, in the `if result is None: ... continue` / success branch), record its index and, when augmenting, generate variants:

```python
        parent_pos = len(temporal_list)  # index this original will occupy
        temporal_list.append(result["temporal"])
        static_cont_list.append(result["static_continuous"])
        static_cat_list.append(result["static_categorical"])
        y_win_list.append(result["y_win"])
        y_offset_list.append(result["y_offset"])
        session_dates.append(result["session_date"])
        symbols.append(result["symbol"])
        is_aug_list.append(False)
        parent_list.append(parent_pos)

        if augment and n_augment > 0:
            transforms = bar_transforms or [lambda df: df]
            rng = augment_rng if augment_rng is not None else np.random.RandomState()
            for _k in range(n_augment):
                tf = transforms[int(rng.randint(0, len(transforms)))]
                aug_res, _r = _assemble_event_with_reason(
                    event_row, tp_pct=tp_pct, sl_pct=sl_pct,
                    lookahead_bars=lookahead_bars, sequence_length=sequence_length,
                    min_bars=min_bars, read_bars=rb,
                    fundamentals_lookup=fundamentals_lookup, spy_bars=spy_bars,
                    bar_transform=tf,
                )
                if aug_res is None:
                    continue
                temporal_list.append(aug_res["temporal"])
                static_cont_list.append(aug_res["static_continuous"])
                static_cat_list.append(aug_res["static_categorical"])
                y_win_list.append(aug_res["y_win"])
                y_offset_list.append(aug_res["y_offset"])
                session_dates.append(aug_res["session_date"])
                symbols.append(aug_res["symbol"])
                is_aug_list.append(True)
                parent_list.append(parent_pos)
```

Initialize `is_aug_list: list[bool] = []` and `parent_list: list[int] = []` alongside the other lists, and add to the returned dict:

```python
        "is_augmented": np.asarray(is_aug_list, dtype=bool),
        "parent_index": np.asarray(parent_list, dtype=np.int64),
```

Add the same two keys (empty arrays) to `_empty_dataset`:

```python
        "is_augmented": np.empty((0,), dtype=bool),
        "parent_index": np.empty((0,), dtype=np.int64),
```

> Cost note: each augmented variant re-runs feature/label computation **in memory** on the already-loaded bars (`read_bars` returns the same frame); no extra S3 GETs beyond the single original read per event.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assemble.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/data/assemble.py tests/test_assemble.py
git commit -m "feat(assemble): bar-transform hook + train-only augmented sample generation"
```

---

### Task 10: Train-loop wiring + dataset cleanup

Route augmented samples to training only, fit normalization on originals only, apply mixup per batch; strip the old per-feature noise/time-roll from `TFTDataset`.

**Files:**
- Modify: `ML_tradingAlgo/tft/train.py`, `ML_tradingAlgo/tft/dataset.py`
- Test: `tests/test_train.py`, `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_train.py`:

```python
def test_split_augmented_train_only():
    from ML_tradingAlgo.tft.train import split_originals_and_augmented
    import numpy as np
    is_aug = np.array([False, True, True, False, True])
    parent = np.array([0, 0, 0, 3, 3])
    # originals at positions 0 and 3
    train_orig = np.array([0])
    val_orig = np.array([3])
    train_idx, val_idx = split_originals_and_augmented(train_orig, val_orig, is_aug, parent)
    # train gets original 0 + its augmented children (1,2); val is original 3 ONLY
    assert set(train_idx.tolist()) == {0, 1, 2}
    assert set(val_idx.tolist()) == {3}
    # no augmented row ever appears in val
    assert not is_aug[val_idx].any()
```

In `tests/test_dataset.py`, replace any test asserting noise/time-roll behavior with:

```python
def test_dataset_no_longer_injects_noise_or_roll():
    import numpy as np
    from ML_tradingAlgo.tft.dataset import TFTDataset
    t = np.random.RandomState(0).randn(3, 30, 69).astype("float32")
    sc = np.zeros((3, 11), "float32"); cat = np.zeros((3, 1), "int64")
    ds = TFTDataset(t, sc, cat, y_win=np.zeros(3), y_offset=np.zeros(3),
                    sample_weights=np.ones(3), augment=True)  # augment is now a no-op default
    item = ds[0]
    assert np.allclose(item["temporal"].numpy(), t[0])  # unchanged (no noise/roll)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train.py::test_split_augmented_train_only tests/test_dataset.py::test_dataset_no_longer_injects_noise_or_roll -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `ML_tradingAlgo/tft/dataset.py`: remove `_augment_temporal`'s noise and time-roll branches; keep only optional `feature_dropout` and default it OFF. Update `DEFAULT_AUGMENT_CFG`:

```python
DEFAULT_AUGMENT_CFG = {
    "feature_dropout": 0.0,  # bar-level augmentation + mixup now live upstream
}
```

Simplify `_augment_temporal` to apply only feature dropout (drop the `noise_std`/`time_shift` code). The `augment` flag still gates it, but with the new default it is a no-op unless `feature_dropout` > 0 is explicitly passed.

In `ML_tradingAlgo/tft/train.py`:

Add `"mixup_alpha": 0.3` to `TRAIN_CONFIG`. Remove the `augment_noise_std`/`augment_time_shift`/`augment_feature_dropout` keys (and the `augment_cfg` block in `train_all_folds`).

Add the split helper:

```python
def split_originals_and_augmented(train_orig, val_orig, is_augmented, parent_index):
    """Expand original train indices with their augmented children; keep val pure.

    ``train_orig`` / ``val_orig`` index into the ORIGINAL samples only. Returns
    ``(train_idx, val_idx)`` into the full (original+augmented) arrays, where
    train includes every augmented row whose ``parent_index`` is in
    ``train_orig`` and val contains only the original validation rows.
    """
    import numpy as np
    train_orig = set(int(i) for i in train_orig)
    val_idx = np.asarray(sorted(int(i) for i in val_orig), dtype=np.int64)
    train_list = list(train_orig)
    for j in np.where(np.asarray(is_augmented, dtype=bool))[0]:
        if int(parent_index[j]) in train_orig:
            train_list.append(int(j))
    train_idx = np.asarray(sorted(set(train_list)), dtype=np.int64)
    return train_idx, val_idx
```

Rework `train_all_folds` to: (a) compute folds on ORIGINALS only, (b) expand via the helper, (c) fit norm stats on original-train rows only, (d) build `train_ds` with `augment=False`. Replace the fold setup:

```python
    is_augmented = np.asarray(assembled.get("is_augmented",
                              np.zeros(len(y_win), dtype=bool)), dtype=bool)
    parent_index = np.asarray(assembled.get("parent_index",
                              np.arange(len(y_win))), dtype=np.int64)

    orig_pos = np.where(~is_augmented)[0]
    orig_dates = [session_dates[i] for i in orig_pos]
    orig_folds = create_walk_forward_folds(orig_dates, num_folds=num_folds)

    all_metrics = []
    for fold_idx, (tr_rel, va_rel) in enumerate(orig_folds):
        train_orig = orig_pos[tr_rel]
        val_orig = orig_pos[va_rel]
        train_idx, val_idx = split_originals_and_augmented(
            train_orig, val_orig, is_augmented, parent_index
        )

        # Normalization fit on ORIGINAL train rows only (no augmented, no leakage).
        train_stats = compute_normalization_stats(
            temporal[train_orig], static_continuous[train_orig]
        )

        train_ds = TFTDataset(
            temporal[train_idx], static_continuous[train_idx],
            static_categorical[train_idx], y_win[train_idx], y_offset[train_idx],
            sample_weights[train_idx], norm_stats=train_stats, augment=False,
        )
        val_ds = TFTDataset(
            temporal[val_idx], static_continuous[val_idx],
            static_categorical[val_idx], y_win[val_idx], y_offset[val_idx],
            sample_weights[val_idx], norm_stats=train_stats, augment=False,
        )
        metrics = train_fold(train_ds, val_ds, model_config, fold_idx, output_dir,
                             train_config=cfg)
        all_metrics.append(metrics)
    return all_metrics
```

In `train_fold`, apply mixup inside the training batch loop, just before the forward pass:

```python
            optimizer.zero_grad()
            if cfg.get("mixup_alpha", 0.0) and cfg["mixup_alpha"] > 0:
                from .augment import mixup_batch
                batch = mixup_batch(batch, alpha=cfg["mixup_alpha"])
            p_win, entry_offset, _ = model(
                batch["temporal"], batch["static_continuous"], batch["static_categorical"],
            )
```

> Note: `mixup_batch` needs a `weight` key in the batch. The `TFTDataset.__getitem__` already returns `weight`, so batches include it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train.py tests/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ML_tradingAlgo/tft/train.py ML_tradingAlgo/tft/dataset.py tests/test_train.py tests/test_dataset.py
git commit -m "feat(train): train-only augmentation routing, norm on originals, batch mixup"
```

---

### Task 11: Full-suite verification

**Files:** none (verification + final commit only).

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: PASS (no failures). If any pre-existing test referenced 47/7 dims or old augmentation behavior, fix it to the new dims/behavior in the same task and note it in the commit.

- [ ] **Step 2: Syntax sanity check**

Run: `for f in ML_tradingAlgo/tft/*.py ML_tradingAlgo/data/*.py; do python -c "import ast; ast.parse(open('$f').read())"; done; echo OK`
Expected: `OK`.

- [ ] **Step 3: Smoke-test the augmented training path end-to-end**

Run:
```bash
python -m pytest tests/test_train.py -k "fold or augment" -v
```
Expected: PASS — confirms folds build, augmented rows route to train only, and a fold trains for a few epochs.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: green full suite after feature + augmentation changes"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Tier A features → Tasks 1 (tod), 2 (float_rotation, dollar-vol), 3 (pm levels, round number), 4 (pullback), 5 (day-of-run, gap-vs-range). ✅
- Batch 2 features → Tasks 1 (dow), 2 (intraday RVOL), 3 (OR break, anchored VWAP, prior-day, gap-fill, EMA overextension), 4 (new-HOD, accel). ✅
- Keep all 55 existing (append-only) → enforced by "append to END" convention. ✅
- New static_data contract inputs (prior-day bars, day-of-run, intraday volume profile) supplied by assemble + optional for live → Tasks 2, 3, 5; optional-with-fallback throughout. ✅
- Dimension propagation 47→69 / 7→11 (incl. `tft_predictor.py` docstrings + a functional dimension guard with a test) → Task 6. ✅
- `tft/augment.py` bar-level jitter/window-slice/time-warp → Task 7. ✅
- Recompute features AND labels on perturbed bars → Task 9 (`bar_transform` re-runs `build_feature_matrix` + `compute_labels` via `_assemble_event_with_reason`). ✅
- Mixup → Tasks 8 (impl) + 10 (wired into train loop). ✅
- Augmented = train-only, normalization on originals only → Task 10. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✅

**Type/name consistency:** `bar_transform` (singular, per-event) vs `bar_transforms` (list, dataset-level) used consistently; `is_augmented`/`parent_index` keys consistent across assemble (Task 9) and train (Task 10); `mixup_batch`, `split_originals_and_augmented`, `build_intraday_volume_profile` names match between definition and call sites. Final counts (60 one-min, 9 five-min = 69 temporal; 11 static continuous) reconcile across Tasks 1–6. ✅

**Deviations from spec:** none of substance. `anchored_vwap_dist_atr` is anchored at the first regular-hours bar (via timestamps) to avoid exact duplication with the existing premarket-inclusive cumulative `vwap_distance_atr`; falls back to cumulative VWAP when timestamps are absent.

**Out of scope (unchanged):** generative augmentation, recency weighting, peer features, Tier B/C features, `interpret.py`-based pruning.
