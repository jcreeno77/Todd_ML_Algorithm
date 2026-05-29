"""Walk-forward training loop for the Temporal Fusion Transformer.

This module trains the dual-head TFT (P(win) + entry offset) using an
expanding-window walk-forward cross-validation scheme. For each fold:

  1. Sort samples chronologically by session date.
  2. Compute normalization statistics on the TRAIN split ONLY (never on
     val/all data) to avoid lookahead leakage.
  3. Inject those train-fit stats into both the train and val datasets.
  4. Train with early stopping on validation loss, gradient clipping, and the
     class-weighted + per-sample-weighted dual loss.
  5. Save the BEST checkpoint as a bare state_dict plus the train-fit
     normalization stats, so a downstream predictor can reload them.

Checkpoint contract (read by a separate predictor module):
  output_dir/fold_{i}/model.pt        = torch.save(model.state_dict(), path)
  output_dir/fold_{i}/norm_stats.npz  = np.savez(path, temporal_mean=...,
                                          temporal_std=..., static_mean=...,
                                          static_std=...)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import TFTDataset, compute_normalization_stats
from .model import TemporalFusionTransformer

TRAIN_CONFIG = {
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 15,
    "grad_clip": 1.0,
    "bce_weight": 0.7,
    "mse_weight": 0.3,
    "mixup_alpha": 0.3,
}


def split_originals_and_augmented(train_orig, val_orig, is_augmented, parent_index):
    """Expand original train indices with their augmented children; keep val pure.

    train_orig / val_orig index into ORIGINAL samples only. Returns (train_idx,
    val_idx) into the full (original+augmented) arrays: train includes every
    augmented row whose parent_index is in train_orig; val contains only the
    original validation rows.
    """
    train_orig_set = set(int(i) for i in train_orig)
    val_idx = np.asarray(sorted(int(i) for i in val_orig), dtype=np.int64)
    train_list = list(train_orig_set)
    for j in np.where(np.asarray(is_augmented, dtype=bool))[0]:
        if int(parent_index[j]) in train_orig_set:
            train_list.append(int(j))
    train_idx = np.asarray(sorted(set(train_list)), dtype=np.int64)
    return train_idx, val_idx


def create_walk_forward_folds(session_dates, num_folds=4):
    """Build expanding-window (train_idx, val_idx) walk-forward splits.

    Sample indices are sorted chronologically by ``session_dates``. The sorted
    timeline is divided into ``num_folds + 1`` contiguous blocks. Fold ``k``
    (0-indexed) trains on blocks ``[0 .. k]`` and validates on block ``k + 1``.
    This guarantees:
      - every training index is chronologically <= its validation indices
        (no future leakage), and
      - the training window grows monotonically across folds.

    Args:
        session_dates: sequence of orderable (e.g. ``datetime.date``) values,
            one per sample, in sample order (need NOT be pre-sorted).
        num_folds: number of (train, val) splits to produce.

    Returns:
        list of (train_idx, val_idx) tuples of ``np.ndarray`` of int indices
        into the ORIGINAL (unsorted) sample order.
    """
    session_dates = list(session_dates)
    n = len(session_dates)
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")
    if n < num_folds + 1:
        raise ValueError(
            f"need at least num_folds+1={num_folds + 1} samples, got {n}"
        )

    # Indices into the original array, ordered chronologically.
    order = np.argsort(np.asarray(session_dates), kind="stable")

    # Split the chronological timeline into num_folds + 1 contiguous blocks.
    blocks = np.array_split(order, num_folds + 1)

    folds = []
    for k in range(num_folds):
        train_idx = np.concatenate(blocks[: k + 1])
        val_idx = blocks[k + 1]
        folds.append((np.asarray(train_idx), np.asarray(val_idx)))
    return folds


def compute_class_pos_weight(y_win) -> float:
    """Return ``n_negative / n_positive`` for use as BCE ``pos_weight``.

    Guards the degenerate single-class cases: returns ``1.0`` when there are no
    positive samples (avoids divide-by-zero) or no negative samples (a
    pos_weight of 0 would silently zero out the positive class).
    """
    y = np.asarray(y_win, dtype=np.float32).reshape(-1)
    n_pos = float((y > 0.5).sum())
    n_neg = float((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos


def tft_loss(p_win, entry_offset, y_win, y_offset, weight, pos_weight,
             bce_weight=0.7, mse_weight=0.3) -> torch.Tensor:
    """Combined dual-head loss.

    Returns ``bce_weight * weighted_BCE + mse_weight * weighted_MSE`` as a
    scalar tensor.

    The BCE term is class-weighted (positive samples scaled by ``pos_weight``)
    and per-sample-weighted (by ``weight``); ``p_win`` is already in [0, 1]
    (the model applies a sigmoid), so plain ``binary_cross_entropy`` is used
    (NOT the with-logits variant). The MSE term is per-sample-weighted.
    """
    # Flatten everything to (B, 1) for safe broadcasting.
    p_win = p_win.reshape(-1, 1)
    entry_offset = entry_offset.reshape(-1, 1)
    y_win = y_win.reshape(-1, 1)
    y_offset = y_offset.reshape(-1, 1)
    weight = weight.reshape(-1, 1)

    # Clamp to avoid log(0) blowing up the BCE.
    p_win = p_win.clamp(1e-7, 1.0 - 1e-7)

    # Per-element BCE, then apply class weighting (pos_weight on positives)
    # and the per-sample weight, then mean.
    bce = F.binary_cross_entropy(p_win, y_win, reduction="none")
    class_weight = torch.where(
        y_win > 0.5,
        torch.full_like(y_win, float(pos_weight)),
        torch.ones_like(y_win),
    )
    bce = (bce * class_weight * weight).mean()

    mse = F.mse_loss(entry_offset, y_offset, reduction="none")
    mse = (mse * weight).mean()

    return bce_weight * bce + mse_weight * mse


def evaluate(model, dataloader) -> dict:
    """Evaluate a model over a dataloader.

    Returns a dict with:
      - ``win_rate``: fraction of predicted-positive (``p_win > 0.5``)
        predictions whose label also satisfies ``y_win > 0.5``. (When there are
        no predicted positives, this is 0.0.)
      - ``profit_factor``: simple proxy = sum of "gains" / sum of "losses" over
        predicted-positive trades, where each predicted-positive trade
        contributes ``p_win`` to gains if it was a true win (``y_win > 0.5``)
        and ``p_win`` to losses otherwise. (No losses -> inf when gains > 0,
        else 0.0.)
      - ``loss``: mean ``tft_loss`` over the loader (pos_weight = 1.0).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    correct_pos = 0
    pred_pos = 0
    gains = 0.0
    losses = 0.0

    with torch.no_grad():
        for batch in dataloader:
            p_win, entry_offset, _ = model(
                batch["temporal"],
                batch["static_continuous"],
                batch["static_categorical"],
            )
            y_win = batch["y_win"]
            y_offset = batch["y_offset"]
            weight = batch["weight"]

            loss = tft_loss(
                p_win, entry_offset, y_win, y_offset, weight, pos_weight=1.0
            )
            total_loss += loss.item()
            n_batches += 1

            p = p_win.reshape(-1)
            yw = y_win.reshape(-1)
            is_pred_pos = p > 0.5
            is_true = yw > 0.5

            pred_pos += int(is_pred_pos.sum().item())
            correct_pos += int((is_pred_pos & is_true).sum().item())

            for pi, ti in zip(p[is_pred_pos].tolist(), is_true[is_pred_pos].tolist()):
                if ti:
                    gains += pi
                else:
                    losses += pi

    win_rate = (correct_pos / pred_pos) if pred_pos > 0 else 0.0
    if losses > 0:
        profit_factor = gains / losses
    else:
        profit_factor = float("inf") if gains > 0 else 0.0
    mean_loss = (total_loss / n_batches) if n_batches > 0 else 0.0

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "loss": float(mean_loss),
    }


def _save_checkpoint(model, norm_stats, fold_dir):
    """Persist the bare state_dict and the train-fit normalization stats."""
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
    np.savez(
        os.path.join(fold_dir, "norm_stats.npz"),
        temporal_mean=np.asarray(norm_stats["temporal_mean"]),
        temporal_std=np.asarray(norm_stats["temporal_std"]),
        static_mean=np.asarray(norm_stats["static_mean"]),
        static_std=np.asarray(norm_stats["static_std"]),
    )


def train_fold(train_ds, val_ds, model_config, fold_idx, output_dir,
               train_config=TRAIN_CONFIG, max_epochs=None) -> dict:
    """Train one walk-forward fold with early stopping; save best checkpoint.

    Builds DataLoaders over ``train_ds``/``val_ds``, trains a fresh
    ``TemporalFusionTransformer`` with Adam, gradient clipping, and early
    stopping on validation loss (``patience`` epochs). The best (lowest val
    loss) model's state_dict is saved to ``output_dir/fold_{fold_idx}/model.pt``
    and the train-fit normalization stats (read from ``train_ds.norm_stats``)
    to ``norm_stats.npz`` alongside it.

    Args:
        max_epochs: optional override of ``train_config["max_epochs"]`` (used
            for fast smoke tests).

    Returns:
        the ``evaluate()`` metrics dict computed on ``val_ds`` using the best
        restored weights.
    """
    cfg = dict(TRAIN_CONFIG)
    cfg.update(train_config or {})
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs

    batch_size = cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class imbalance weight from the training labels.
    pos_weight = compute_class_pos_weight(train_ds.y_win)

    model = TemporalFusionTransformer(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    norm_stats = train_ds.norm_stats

    for _epoch in range(cfg["max_epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if cfg.get("mixup_alpha", 0.0) and cfg["mixup_alpha"] > 0:
                from .augment import mixup_batch
                batch = mixup_batch(batch, alpha=cfg["mixup_alpha"])
            p_win, entry_offset, _ = model(
                batch["temporal"],
                batch["static_continuous"],
                batch["static_categorical"],
            )
            loss = tft_loss(
                p_win, entry_offset,
                batch["y_win"], batch["y_offset"], batch["weight"],
                pos_weight=pos_weight,
                bce_weight=cfg["bce_weight"], mse_weight=cfg["mse_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        val_metrics = evaluate(model, val_loader)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                break

    # Restore best weights (fallback to final if none recorded).
    if best_state is not None:
        model.load_state_dict(best_state)

    _save_checkpoint(model, norm_stats, fold_dir)

    return evaluate(model, val_loader)


def train_all_folds(assembled, model_config, output_dir, num_folds=4,
                    train_config=TRAIN_CONFIG) -> list:
    """Run walk-forward training across all folds.

    For each fold, normalization stats are fit on the TRAIN split ONLY and
    injected into both the (augmented) train dataset and the (non-augmented)
    val dataset, and saved alongside the per-fold weights.

    Args:
        assembled: dict from ``data.assemble.assemble_dataset`` with keys
            ``temporal`` (N,30,69), ``static_continuous`` (N,11),
            ``static_categorical`` (N,1), ``y_win`` (N,), ``y_offset`` (N,),
            ``sample_weights`` (N,), and ``session_dates`` (list of N).

    Returns:
        list of per-fold metric dicts (one per fold, in fold order).
    """
    cfg = dict(TRAIN_CONFIG)
    cfg.update(train_config or {})

    temporal = np.asarray(assembled["temporal"], dtype=np.float32)
    static_continuous = np.asarray(assembled["static_continuous"], dtype=np.float32)
    static_categorical = np.asarray(assembled["static_categorical"])
    y_win = np.asarray(assembled["y_win"], dtype=np.float32).reshape(-1)
    y_offset = np.asarray(assembled["y_offset"], dtype=np.float32).reshape(-1)
    sample_weights = np.asarray(assembled["sample_weights"], dtype=np.float32).reshape(-1)
    session_dates = assembled["session_dates"]

    # Support datasets that include augmented children alongside originals.
    # Safe fallback: treat all rows as originals with identity parent mapping.
    is_augmented = np.asarray(
        assembled.get("is_augmented", np.zeros(len(y_win), dtype=bool)),
        dtype=bool,
    )
    parent_index = np.asarray(
        assembled.get("parent_index", np.arange(len(y_win))),
        dtype=np.int64,
    )

    # Build walk-forward folds on ORIGINALS ONLY to avoid leaking augmented
    # children into validation and to keep fold boundaries clean.
    orig_pos = np.where(~is_augmented)[0]
    orig_dates = [session_dates[i] for i in orig_pos]
    orig_folds = create_walk_forward_folds(orig_dates, num_folds=num_folds)

    all_metrics = []
    for fold_idx, (tr_rel, va_rel) in enumerate(orig_folds):
        # Map relative-to-orig_pos indices back to full-array positions.
        train_orig = orig_pos[tr_rel]
        val_orig = orig_pos[va_rel]

        # Expand training set with augmented children; val stays pure originals.
        train_idx, val_idx = split_originals_and_augmented(
            train_orig, val_orig, is_augmented, parent_index
        )

        # Fit normalization stats on ORIGINAL train samples only (no leakage).
        train_stats = compute_normalization_stats(
            temporal[train_orig], static_continuous[train_orig]
        )

        train_ds = TFTDataset(
            temporal[train_idx],
            static_continuous[train_idx],
            static_categorical[train_idx],
            y_win[train_idx],
            y_offset[train_idx],
            sample_weights[train_idx],
            norm_stats=train_stats,
            augment=False,
        )
        val_ds = TFTDataset(
            temporal[val_idx],
            static_continuous[val_idx],
            static_categorical[val_idx],
            y_win[val_idx],
            y_offset[val_idx],
            sample_weights[val_idx],
            norm_stats=train_stats,
            augment=False,
        )

        metrics = train_fold(
            train_ds, val_ds, model_config, fold_idx, output_dir,
            train_config=cfg,
        )
        all_metrics.append(metrics)

    return all_metrics
