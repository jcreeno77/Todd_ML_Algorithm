"""Dataset and normalization utilities for the Temporal Fusion Transformer.

The TFT consumes three input groups per sample:
  - temporal: (30, 69) sliding-window candle/feature matrix
  - static_continuous: (11,) per-symbol continuous features (e.g. float, ADV)
  - static_categorical: (1,) categorical code (e.g. GICS sector) -- never normalized

Normalization statistics are computed across both the sample and timestep axes
(per feature) so that every feature column has zero mean / unit variance. The
categorical feature is explicitly excluded from normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

_STD_FLOOR = 1e-8

DEFAULT_AUGMENT_CFG = {
    "feature_dropout": 0.0,  # bar-level augmentation + mixup now live upstream
}


def compute_normalization_stats(temporal: np.ndarray, static_continuous: np.ndarray) -> dict:
    """Compute per-feature normalization statistics.

    Args:
        temporal: (N, 30, 69) float array.
        static_continuous: (N, 11) float array.

    Returns:
        dict with keys "temporal_mean" (69,), "temporal_std" (69,),
        "static_mean" (11,), "static_std" (11,). Standard deviations are clamped
        to a floor of 1e-8 to avoid division by zero on constant features.
        The categorical feature is not handled here.
    """
    temporal = np.asarray(temporal, dtype=np.float64)
    static_continuous = np.asarray(static_continuous, dtype=np.float64)

    temporal_mean = temporal.mean(axis=(0, 1))
    temporal_std = temporal.std(axis=(0, 1))
    static_mean = static_continuous.mean(axis=0)
    static_std = static_continuous.std(axis=0)

    temporal_std = np.maximum(temporal_std, _STD_FLOOR)
    static_std = np.maximum(static_std, _STD_FLOOR)

    return {
        "temporal_mean": temporal_mean.astype(np.float32),
        "temporal_std": temporal_std.astype(np.float32),
        "static_mean": static_mean.astype(np.float32),
        "static_std": static_std.astype(np.float32),
    }


class TFTDataset(Dataset):
    """Map-style dataset yielding TFT-ready tensor dicts.

    When ``norm_stats`` is provided, temporal and static-continuous features are
    normalized per feature. The categorical feature is always returned as-is.
    When ``augment`` is True, temporal features are perturbed on-the-fly in
    ``__getitem__`` (training only): Gaussian noise, random time roll, and
    random feature-column dropout.
    """

    def __init__(self, temporal, static_continuous, static_categorical,
                 y_win, y_offset, sample_weights, norm_stats=None,
                 augment=False, augment_cfg=None):
        self.temporal = np.asarray(temporal, dtype=np.float32)
        self.static_continuous = np.asarray(static_continuous, dtype=np.float32)
        self.static_categorical = np.asarray(static_categorical, dtype=np.int64)

        # y_win may arrive as (N,) or (N,1); flatten to (N,).
        self.y_win = np.asarray(y_win, dtype=np.float32).reshape(-1)
        self.y_offset = np.asarray(y_offset, dtype=np.float32).reshape(-1)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        self.norm_stats = norm_stats
        self.augment = augment

        cfg = dict(DEFAULT_AUGMENT_CFG)
        if augment_cfg:
            cfg.update(augment_cfg)
        self.augment_cfg = cfg

        if norm_stats is not None:
            self._temporal_mean = np.asarray(norm_stats["temporal_mean"], dtype=np.float32)
            self._temporal_std = np.asarray(norm_stats["temporal_std"], dtype=np.float32)
            self._static_mean = np.asarray(norm_stats["static_mean"], dtype=np.float32)
            self._static_std = np.asarray(norm_stats["static_std"], dtype=np.float32)

    def __len__(self) -> int:
        return self.temporal.shape[0]

    def _augment_temporal(self, temporal: np.ndarray) -> np.ndarray:
        """Apply optional feature dropout to a (T, F) array.

        Noise and time-roll augmentation have been removed; those transforms now
        live upstream (bar-level jitter/warp in ``augment.py``) or at the batch
        level (``mixup_batch`` in the training loop). Only feature-column dropout
        is retained here, and it defaults to 0.0 (a no-op) so that the overall
        in-dataset augmentation is off by default.
        """
        cfg = self.augment_cfg

        feature_dropout = cfg.get("feature_dropout", 0.0)
        if feature_dropout and feature_dropout > 0:
            num_features = temporal.shape[1]
            keep_mask = (torch.rand(num_features).numpy() >= feature_dropout)
            temporal = temporal * keep_mask.astype(np.float32)[None, :]

        return temporal.astype(np.float32)

    def __getitem__(self, idx) -> dict:
        temporal = self.temporal[idx].copy()
        static_continuous = self.static_continuous[idx].copy()
        static_categorical = self.static_categorical[idx]

        if self.norm_stats is not None:
            temporal = (temporal - self._temporal_mean) / self._temporal_std
            static_continuous = (static_continuous - self._static_mean) / self._static_std

        if self.augment:
            temporal = self._augment_temporal(temporal)

        return {
            "temporal": torch.as_tensor(temporal, dtype=torch.float32),
            "static_continuous": torch.as_tensor(static_continuous, dtype=torch.float32),
            "static_categorical": torch.as_tensor(static_categorical, dtype=torch.int64).reshape(1),
            "y_win": torch.tensor([self.y_win[idx]], dtype=torch.float32),
            "y_offset": torch.tensor([self.y_offset[idx]], dtype=torch.float32),
            "weight": torch.tensor([self.sample_weights[idx]], dtype=torch.float32),
        }
