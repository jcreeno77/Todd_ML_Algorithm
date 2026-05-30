# tests/test_run_baseline.py
"""Dry tests for the baseline driver: chronological split + leakage guards.

No S3 / network: ``store.read_bars`` and ``assemble_dataset`` are monkeypatched
to return synthetic frames/arrays. Training runs for 1 epoch on a handful of
synthetic samples purely to exercise the wiring end to end.
"""
import argparse
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from ML_tradingAlgo.tft import run_baseline


def _dates(n, start=datetime.date(2024, 1, 1)):
    return [start + datetime.timedelta(days=i) for i in range(n)]


def _assembled(dates, n_aug_per=0, seed=0):
    """Synthetic assembled dict; originals first, then augmented children."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    temporal, statc, statk, yw, yo, sw, sess, is_aug, parent = (
        [], [], [], [], [], [], [], [], []
    )
    for i, d in enumerate(dates):
        temporal.append(rng.normal(size=(30, 69)).astype(np.float32))
        statc.append(rng.normal(size=(11,)).astype(np.float32))
        statk.append(rng.integers(0, 11, size=(1,)).astype(np.int64))
        yw.append(float(rng.integers(0, 2)))
        yo.append(float(rng.normal()))
        sw.append(1.0)
        sess.append(d)
        is_aug.append(False)
        parent.append(i)
    # augmented children appended after all originals (matches assemble ordering
    # closely enough for the split logic, which keys off is_augmented/parent).
    for i, d in enumerate(dates):
        for _ in range(n_aug_per):
            temporal.append(rng.normal(size=(30, 69)).astype(np.float32))
            statc.append(rng.normal(size=(11,)).astype(np.float32))
            statk.append(rng.integers(0, 11, size=(1,)).astype(np.int64))
            yw.append(float(rng.integers(0, 2)))
            yo.append(float(rng.normal()))
            sw.append(1.0)
            sess.append(d)
            is_aug.append(True)
            parent.append(i)
    return {
        "temporal": np.stack(temporal).astype(np.float32),
        "static_continuous": np.stack(statc).astype(np.float32),
        "static_categorical": np.stack(statk).astype(np.int64),
        "y_win": np.asarray(yw, dtype=np.float32),
        "y_offset": np.asarray(yo, dtype=np.float32),
        "sample_weights": np.asarray(sw, dtype=np.float32),
        "session_dates": sess,
        "symbols": ["SYN"] * len(sess),
        "skipped": [("SYN", datetime.date(2024, 1, 1), "short_session")],
        "is_augmented": np.asarray(is_aug, dtype=bool),
        "parent_index": np.asarray(parent, dtype=np.int64),
    }


# --- 1. chronological cutoff ------------------------------------------------ #
def test_chronological_cutoff_disjoint_and_recent_test(monkeypatch):
    ev = pd.DataFrame({"symbol": ["X"] * 10, "session_date": _dates(10)})
    monkeypatch.setattr(run_baseline.store, "read_bars", lambda *a, **k: ev)

    trainval_end, test_start, distinct = run_baseline._chronological_cutoff(
        datetime.date(2024, 1, 1), datetime.date(2024, 1, 10), test_frac=0.2
    )
    assert len(distinct) == 10
    assert trainval_end < test_start  # disjoint, ordered
    # test_frac 0.2 of 10 dates -> 8 trainval buckets, 2 test
    assert trainval_end == datetime.date(2024, 1, 8)
    assert test_start == datetime.date(2024, 1, 9)


def test_chronological_cutoff_raises_when_too_few_dates(monkeypatch):
    ev = pd.DataFrame({"symbol": ["X"], "session_date": _dates(1)})
    monkeypatch.setattr(run_baseline.store, "read_bars", lambda *a, **k: ev)
    with pytest.raises(SystemExit):
        run_baseline._chronological_cutoff(
            datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), test_frac=0.2
        )


def test_chronological_cutoff_raises_when_no_events(monkeypatch):
    monkeypatch.setattr(run_baseline.store, "read_bars", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit):
        run_baseline._chronological_cutoff(
            datetime.date(2024, 1, 1), datetime.date(2024, 6, 1), test_frac=0.2
        )


# --- 2. end-to-end wiring (synthetic, 1 epoch) ------------------------------ #
def test_run_pure_test_no_leakage_and_artifacts(monkeypatch, tmp_path):
    train_dates = _dates(8, start=datetime.date(2024, 1, 1))   # 01-01 .. 01-08
    test_dates = _dates(4, start=datetime.date(2024, 1, 9))    # 01-09 .. 01-12
    all_dates = train_dates + test_dates

    ev = pd.DataFrame({"symbol": ["X"] * len(all_dates), "session_date": all_dates})
    monkeypatch.setattr(run_baseline.store, "read_bars", lambda *a, **k: ev)

    trainval = _assembled(train_dates, n_aug_per=1, seed=1)   # 8 orig + 8 aug
    test = _assembled(test_dates, n_aug_per=0, seed=2)         # 4 pure originals

    calls = []

    def fake_assemble(session_date_range, **kwargs):
        calls.append((session_date_range, kwargs))
        return trainval if len(calls) == 1 else test

    monkeypatch.setattr(run_baseline, "assemble_dataset", fake_assemble)

    args = argparse.Namespace(
        start="2024-01-01", end="2024-01-12", test_frac=0.2, n_augment=1,
        num_folds=2, output_dir=str(tmp_path), tp_pct=3.0, sl_pct=3.0,
        lookahead_bars=30, seed=42, max_epochs=1, batch_size=8,
    )
    report = run_baseline.run(args)

    # driver requested an AUGMENTED trainval and a PURE test set
    assert calls[0][1]["augment"] is True and calls[0][1]["n_augment"] == 1
    assert calls[0][1]["bar_transforms"]  # non-empty
    assert calls[1][1]["augment"] is False and calls[1][1]["n_augment"] == 0

    # counts reflect originals-only test and augmented trainval
    assert report["counts"]["test_originals"] == 4
    assert report["counts"]["trainval_originals"] == 8
    assert report["counts"]["trainval_augmented"] == 8

    # held-out eval ran and produced the three metric keys
    assert report["test_metrics"] is not None
    assert set(report["test_metrics"]) == {"win_rate", "profit_factor", "loss"}

    # artifacts written for the selected (final) fold + report
    selected = report["training"]["selected_fold"]
    assert (tmp_path / f"fold_{selected}" / "model.pt").exists()
    assert (tmp_path / f"fold_{selected}" / "norm_stats.npz").exists()
    saved = json.loads((tmp_path / "baseline_report.json").read_text())
    assert saved["gating"]["pass"] in (True, False)
