"""Baseline TFT run: chronological-holdout training + held-out test evaluation.

This is a thin driver that wires together existing pieces (no new modeling logic):

    events table  ->  chronological cutoff (most recent ~test_frac reserved)
                  ->  assemble TRAINVAL (augmented) + TEST (pure) over DISJOINT date ranges
                  ->  train_all_folds  (expanding-window walk-forward CV)
                  ->  load the FINAL fold's checkpoint + its train-fit norm stats
                  ->  evaluate that model on the untouched TEST set
                  ->  baseline_report.json + a gating verdict.

Why a strict chronological holdout: every TEST ``session_date`` is strictly greater
than every TRAINVAL ``session_date`` (the two assemble ranges are disjoint and ordered),
and the TEST set is assembled with ``augment=False`` so no synthetic row ever enters
evaluation. TEST features are normalized with the FINAL fold's TRAIN-fit statistics
(loaded from its ``norm_stats.npz``), never with stats fit on test data -> no leakage.

Cost: TRAINVAL and TEST cover disjoint dates, so total S3 minute reads ~= one full pass.
Augmentation reuses each event's already-read bars, so ``--n-augment`` does not multiply
S3 reads.

Run (after backfill has populated S3):
    python -m ML_tradingAlgo.tft.run_baseline --start 2025-12-15 --end 2026-05-30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ML_tradingAlgo.data.assemble import assemble_dataset
from ML_tradingAlgo.data import store
from ML_tradingAlgo.tft.augment import jitter_bars, time_warp_bars, window_slice_bars
from ML_tradingAlgo.tft.dataset import TFTDataset
from ML_tradingAlgo.tft.model import TemporalFusionTransformer
from ML_tradingAlgo.tft.train import TRAIN_CONFIG, evaluate, train_all_folds

# Canonical architecture config (mirrors tests/conftest.py model_config fixture).
MODEL_CONFIG = {
    "hidden_size": 160,
    "lstm_layers": 2,
    "attention_heads": 4,
    "dropout": 0.3,
    "num_temporal_features": 69,
    "num_static_continuous": 11,
    "num_static_categorical": 1,
    "categorical_cardinalities": [11],  # 11 GICS sectors
    "categorical_embedding_dim": 8,
    "sequence_length": 30,
}

# Deployment gating criteria (from momentum_trader_project_spec.md).
WIN_RATE_GATE = 0.55
PROFIT_FACTOR_GATE = 1.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_date(x):
    """Coerce date / datetime / Timestamp / 'YYYY-MM-DD' string to a date."""
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    try:
        import pandas as pd

        ts = pd.Timestamp(x)
        return ts.date()
    except Exception:
        return None


def _load_env():
    """Load ML_tradingAlgo/.env so AWS/S3/Schwab creds are available."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)


def _build_bar_transforms(seed):
    """Return (bar_transforms, selection_rng) for train-only augmentation.

    A single seeded perturbation RNG is shared across the three length-preserving
    transforms; a separate RNG governs which transform is chosen per augmentation.
    Both seeded so an augmented run is reproducible.
    """
    pert = np.random.RandomState(seed)
    transforms = [
        lambda df: jitter_bars(df, sigma=0.01, rng=pert),
        lambda df: window_slice_bars(df, frac=0.6, rng=pert),
        lambda df: time_warp_bars(df, n_knots=4, sigma=0.2, rng=pert),
    ]
    sel_rng = np.random.RandomState(seed + 1)
    return transforms, sel_rng


def _chronological_cutoff(start, end, test_frac):
    """Split the window's distinct event dates into trainval | test by recency.

    Reads ONLY the ``events`` table (no minute pull). Reserves the most recent
    ``test_frac`` of distinct event dates for test. Returns
    ``(trainval_end, test_start, distinct_dates)`` where ``trainval_end <
    test_start`` so the two assemble ranges are disjoint, or raises SystemExit
    when there are too few distinct dates to hold out a test set.

    Note: uses ALL events in the table (not just ``passed_filters``), matching
    what ``assemble_dataset`` actually consumes.
    """
    events = store.read_bars("events", date_range=(start, end))
    if events is None or len(events) == 0:
        raise SystemExit(
            f"No events in S3 for {start}..{end}. Run the backfill first "
            f"(python -m ML_tradingAlgo.data.backfill ...)."
        )
    dates = sorted({d for d in (_to_date(x) for x in events["session_date"]) if d})
    n = len(dates)
    if n < 2:
        raise SystemExit(
            f"Only {n} distinct event date(s) in {start}..{end}; need >=2 to hold "
            f"out a chronological test set. Backfill more symbols/days."
        )
    # number of trainval date-buckets; clamp so >=1 date lands in each side
    k = int(round(n * (1.0 - test_frac)))
    k = max(1, min(k, n - 1))
    return dates[k - 1], dates[k], dates


def _eval_on_test(test, fold_dir, batch_size):
    """Load fold checkpoint + train-fit stats, evaluate on the pure test set."""
    fold_dir = Path(fold_dir)
    stats = np.load(fold_dir / "norm_stats.npz")
    norm_stats = {
        "temporal_mean": np.asarray(stats["temporal_mean"], dtype=np.float32),
        "temporal_std": np.asarray(stats["temporal_std"], dtype=np.float32),
        "static_mean": np.asarray(stats["static_mean"], dtype=np.float32),
        "static_std": np.asarray(stats["static_std"], dtype=np.float32),
    }
    model = TemporalFusionTransformer(**MODEL_CONFIG)
    model.load_state_dict(
        torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    ds = TFTDataset(
        test["temporal"],
        test["static_continuous"],
        test["static_categorical"],
        test["y_win"],
        test["y_offset"],
        test["sample_weights"],
        norm_stats=norm_stats,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate(model, loader)


def _n_originals(assembled):
    return int((~np.asarray(assembled.get("is_augmented", []), dtype=bool)).sum())


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    _load_env()
    start = _to_date(args.start)
    end = _to_date(args.end)

    # 1. chronological cutoff from the events table (events-only read).
    trainval_end, test_start, distinct_dates = _chronological_cutoff(
        start, end, args.test_frac
    )
    print(
        f"[cutoff] {len(distinct_dates)} distinct event dates | "
        f"trainval {start}..{trainval_end} | test {test_start}..{end}"
    )

    # 2. assemble disjoint ranges. TRAINVAL augmented; TEST pure (augment=False).
    bar_transforms, sel_rng = _build_bar_transforms(args.seed)
    trainval = assemble_dataset(
        (start, trainval_end),
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        lookahead_bars=args.lookahead_bars,
        sequence_length=MODEL_CONFIG["sequence_length"],
        augment=True,
        n_augment=args.n_augment,
        bar_transforms=bar_transforms,
        augment_rng=sel_rng,
    )
    test = assemble_dataset(
        (test_start, end),
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        lookahead_bars=args.lookahead_bars,
        sequence_length=MODEL_CONFIG["sequence_length"],
        augment=False,
        n_augment=0,
    )

    n_tv_orig = _n_originals(trainval)
    n_tv_total = len(trainval["y_win"])
    n_test = len(test["y_win"])
    print(
        f"[assemble] trainval: {n_tv_orig} originals (+{n_tv_total - n_tv_orig} "
        f"augmented) | test: {n_test} originals"
    )

    if n_tv_orig < 2:
        raise SystemExit(
            f"Only {n_tv_orig} trainval original sample(s) survived assembly; need "
            f">=2 to train at least one walk-forward fold. Backfill more data. "
            f"(skip reasons: {dict(Counter(r for *_, r in trainval['skipped']))})"
        )

    # 3. walk-forward training (auto-reduce folds to fit sample count).
    eff_folds = min(args.num_folds, max(1, n_tv_orig - 1))
    if eff_folds != args.num_folds:
        print(f"[folds] reduced num_folds {args.num_folds} -> {eff_folds} "
              f"(only {n_tv_orig} trainval originals)")

    train_config = dict(TRAIN_CONFIG)
    if args.max_epochs is not None:
        train_config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    fold_metrics = train_all_folds(
        trainval, MODEL_CONFIG, args.output_dir,
        num_folds=eff_folds, train_config=train_config,
    )

    # 4 + 5. select FINAL fold, evaluate on the untouched test set.
    selected = eff_folds - 1
    test_metrics = None
    if n_test > 0:
        # leakage assertion: every test date strictly after every trainval date.
        tv_orig_dates = [
            d for d, a in zip(trainval["session_dates"], trainval["is_augmented"]) if not a
        ]
        assert max(_to_date(d) for d in tv_orig_dates) < min(
            _to_date(d) for d in test["session_dates"]
        ), "chronological leakage: a test date is <= a trainval date"
        test_metrics = _eval_on_test(
            test, Path(args.output_dir) / f"fold_{selected}", train_config["batch_size"]
        )
        print(f"[test] {test_metrics}")
    else:
        print("[test] no test samples assembled -> skipping held-out eval")

    # 6. report + gating verdict.
    gate_pass = bool(
        test_metrics
        and test_metrics["win_rate"] > WIN_RATE_GATE
        and test_metrics["profit_factor"] > PROFIT_FACTOR_GATE
    )
    report = {
        "window": {"start": str(start), "end": str(end)},
        "split": {
            "trainval_range": [str(start), str(trainval_end)],
            "test_range": [str(test_start), str(end)],
            "distinct_event_dates": len(distinct_dates),
            "test_frac": args.test_frac,
        },
        "counts": {
            "trainval_originals": n_tv_orig,
            "trainval_augmented": n_tv_total - n_tv_orig,
            "trainval_total": n_tv_total,
            "test_originals": n_test,
            "skipped_trainval": dict(Counter(r for *_, r in trainval["skipped"])),
            "skipped_test": dict(Counter(r for *_, r in test["skipped"])),
        },
        "training": {
            "num_folds": eff_folds,
            "selected_fold": selected,
            "fold_val_metrics": fold_metrics,
            "train_config": {
                k: train_config[k] for k in ("lr", "batch_size", "max_epochs", "patience")
            },
            "n_augment": args.n_augment,
        },
        "test_metrics": test_metrics,
        "gating": {
            "win_rate_gate": WIN_RATE_GATE,
            "profit_factor_gate": PROFIT_FACTOR_GATE,
            "pass": gate_pass,
            "note": (
                "win_rate = precision on predicted-positives; profit_factor is a "
                "proxy (sum p_win on true wins / on false wins), NOT a backtested P&L."
            ),
        },
        "model_config": MODEL_CONFIG,
    }

    report_path = Path(args.output_dir) / "baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[report] wrote {report_path}")
    verdict = "PASS" if gate_pass else "FAIL/insufficient"
    print(f"[gating] {verdict} (win_rate>{WIN_RATE_GATE}, profit_factor>{PROFIT_FACTOR_GATE})")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="window start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="window end YYYY-MM-DD (inclusive)")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="fraction of most-recent distinct event dates reserved for test")
    p.add_argument("--n-augment", type=int, default=3,
                   help="augmented variants per trainval original (0 disables)")
    p.add_argument("--num-folds", type=int, default=4,
                   help="walk-forward folds (auto-reduced if too few samples)")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--tp-pct", type=float, default=3.0)
    p.add_argument("--sl-pct", type=float, default=3.0)
    p.add_argument("--lookahead-bars", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override TRAIN_CONFIG max_epochs (e.g. small for smoke runs)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override TRAIN_CONFIG batch_size")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
