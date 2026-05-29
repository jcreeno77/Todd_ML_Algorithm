"""Interpretability utilities for the Temporal Fusion Transformer.

Provides feature-importance extraction from the Variable Selection Networks,
attention-heatmap plotting, and an attention-drift (KL divergence) metric.

All plotting uses the headless Agg backend and savefig only (no plt.show()).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .features import (
    STATIC_CONTINUOUS_FEATURE_NAMES,
    TEMPORAL_1MIN_FEATURE_NAMES,
    TEMPORAL_5MIN_FEATURE_NAMES,
)

# Canonical temporal feature ordering (69 = 60 one-minute + 9 five-minute).
TEMPORAL_FEATURE_NAMES = TEMPORAL_1MIN_FEATURE_NAMES + TEMPORAL_5MIN_FEATURE_NAMES


def _iter_batches(dataloader):
    """Yield (temporal, static_continuous, static_categorical) tuples.

    Supports DataLoaders that yield either a tuple/list of tensors (in the
    model's forward-arg order) or a dict with the standard batch keys.
    """
    for batch in dataloader:
        if isinstance(batch, dict):
            yield (
                batch["temporal"],
                batch["static_continuous"],
                batch["static_categorical"],
            )
        else:
            yield batch[0], batch[1], batch[2]


def feature_importance(model, dataloader, feature_names=None) -> dict[str, float]:
    """Per-temporal-feature importance from the temporal VSN selection weights.

    Runs forward passes over ``dataloader``, pulls
    ``model.get_temporal_selection_weights()`` (shape (B, seq, num_features))
    after each, and averages across batch and time to get one weight per
    feature. Because VSN weights are a softmax over features, the returned
    values are non-negative and sum to ~1.

    Returns a dict keyed by ``feature_names`` if provided, else the canonical
    temporal names, else ``feat_{i}``.
    """
    model.eval()
    summed = None
    count = 0

    with torch.no_grad():
        for temporal, static_continuous, static_categorical in _iter_batches(dataloader):
            model(temporal, static_continuous, static_categorical)
            weights = model.get_temporal_selection_weights()  # (B, seq, num_features)
            w = weights.detach().float()
            # Average over time, sum over batch (accumulate batch counts for a
            # correct grand mean across uneven batch sizes).
            per_sample = w.mean(dim=1)  # (B, num_features)
            batch_sum = per_sample.sum(dim=0)  # (num_features,)
            summed = batch_sum if summed is None else summed + batch_sum
            count += per_sample.shape[0]

    if summed is None:
        raise ValueError("dataloader yielded no batches")

    importances = (summed / count).cpu().numpy()
    num_features = importances.shape[0]

    if feature_names is None:
        if num_features == len(TEMPORAL_FEATURE_NAMES):
            feature_names = TEMPORAL_FEATURE_NAMES
        else:
            feature_names = [f"feat_{i}" for i in range(num_features)]

    return {name: float(importances[i]) for i, name in enumerate(feature_names)}


def static_feature_importance(model, dataloader, names=None) -> dict[str, float]:
    """Per-static-continuous-feature importance from the static VSN weights.

    The static VSN scores all static inputs (continuous + categorical); the
    continuous features come first in the concat order, so we report the first
    ``len(STATIC_CONTINUOUS_FEATURE_NAMES)`` (7) weights.
    """
    model.eval()
    summed = None
    count = 0

    with torch.no_grad():
        for temporal, static_continuous, static_categorical in _iter_batches(dataloader):
            model(temporal, static_continuous, static_categorical)
            weights = model.get_static_selection_weights()  # (B, num_static_features)
            w = weights.detach().float()
            batch_sum = w.sum(dim=0)  # (num_static_features,)
            summed = batch_sum if summed is None else summed + batch_sum
            count += w.shape[0]

    if summed is None:
        raise ValueError("dataloader yielded no batches")

    importances = (summed / count).cpu().numpy()

    if names is None:
        names = STATIC_CONTINUOUS_FEATURE_NAMES
    n = len(names)
    importances = importances[:n]

    return {name: float(importances[i]) for i, name in enumerate(names)}


def plot_feature_importance(importance: dict, save_path: str) -> str:
    """Horizontal bar chart of feature importances, sorted ascending.

    Saves to ``save_path`` and returns it.
    """
    items = sorted(importance.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    height = max(3.0, 0.3 * len(labels))
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(range(len(labels)), values, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def plot_temporal_attention(attn_weights, save_path: str, bar_labels=None) -> str:
    """Plot attention heatmap(s).

    Accepts attention of shape (heads, seq, seq) -> one subplot per head, or
    (seq, seq) -> single heatmap. Saves to ``save_path`` and returns it.
    """
    attn = _to_numpy(attn_weights)

    if attn.ndim == 2:
        attn = attn[None, ...]  # treat as single head

    num_heads = attn.shape[0]
    ncols = min(num_heads, 4)
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    last_im = None
    for h in range(num_heads):
        ax = axes[h // ncols][h % ncols]
        last_im = ax.imshow(attn[h], aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"Head {h}" if num_heads > 1 else "Attention")
        ax.set_xlabel("Key bar")
        ax.set_ylabel("Query bar")
        if bar_labels is not None:
            ax.set_xticks(range(len(bar_labels)))
            ax.set_xticklabels(bar_labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(bar_labels)))
            ax.set_yticklabels(bar_labels, fontsize=6)

    # Hide any unused axes.
    for idx in range(num_heads, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def compute_attention_drift(recent, historical) -> float:
    """KL divergence between two attention distributions over the bar axis.

    Both inputs are flattened, clipped to non-negative, epsilon-smoothed, and
    normalized into probability distributions; returns KL(recent || historical).
    Returns 0.0 for identical inputs and a positive value for divergent ones.
    """
    eps = 1e-10
    p = _to_numpy(recent).astype(np.float64).ravel()
    q = _to_numpy(historical).astype(np.float64).ravel()

    if p.shape != q.shape:
        raise ValueError(
            f"attention arrays must have the same size, got {p.shape} vs {q.shape}"
        )

    p = np.clip(p, 0, None) + eps
    q = np.clip(q, 0, None) + eps
    p = p / p.sum()
    q = q / q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    # Guard against tiny negative values from floating-point rounding.
    return max(kl, 0.0)
