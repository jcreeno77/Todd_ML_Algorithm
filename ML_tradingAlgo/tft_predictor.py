"""Inference wrapper for the Temporal Fusion Transformer.

Drop-in replacement for the legacy ``Todd_predict()``: given a single raw
input sample, applies the stored z-score normalization, runs a deterministic
forward pass, and returns ``(p_win, entry_offset)``.

Checkpoint contract (written by ``train.py``):
  - ``model.pt``      -- a plain ``state_dict`` (``torch.save(model.state_dict(), path)``).
  - ``norm_stats.npz`` -- ``np.savez`` with keys ``temporal_mean``,
    ``temporal_std``, ``static_mean``, ``static_std``.
The model architecture is reconstructed from ``model_config`` (not stored in
the checkpoint), then the state_dict is loaded.
"""

from pathlib import Path

import numpy as np
import torch

from .tft.model import TemporalFusionTransformer


class TFTPredictor:
    def __init__(self, model_dir, model_config):
        model_dir = Path(model_dir)

        model = TemporalFusionTransformer(**model_config)
        state_dict = torch.load(
            model_dir / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

        stats = np.load(model_dir / "norm_stats.npz")
        self.norm_stats = {
            "temporal_mean": np.asarray(stats["temporal_mean"], dtype=np.float32),
            "temporal_std": np.asarray(stats["temporal_std"], dtype=np.float32),
            "static_mean": np.asarray(stats["static_mean"], dtype=np.float32),
            "static_std": np.asarray(stats["static_std"], dtype=np.float32),
        }

        self._last_attn = None

    def predict(self, sequence, static_continuous, static_categorical):
        """Run inference on a single sample.

        Args:
            sequence: (30, 69) float array.
            static_continuous: (11,) float array.
            static_categorical: (1,) int array.

        Returns:
            (p_win, entry_offset) as python floats; p_win is in [0, 1].
        """
        sequence = np.asarray(sequence, dtype=np.float32)
        static_continuous = np.asarray(static_continuous, dtype=np.float32)
        static_categorical = np.asarray(static_categorical, dtype=np.int64)

        # Explicit dimension guard: fail loudly if feature widths don't match norm_stats.
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

        # z-score normalize (categorical untouched).
        sequence = (sequence - self.norm_stats["temporal_mean"]) / self.norm_stats["temporal_std"]
        static_continuous = (
            static_continuous - self.norm_stats["static_mean"]
        ) / self.norm_stats["static_std"]

        temporal_t = torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0)
        static_cont_t = torch.as_tensor(static_continuous, dtype=torch.float32).unsqueeze(0)
        static_cat_t = torch.as_tensor(static_categorical, dtype=torch.int64).unsqueeze(0)

        with torch.no_grad():
            p_win, entry_offset, attn_weights = self.model(
                temporal_t, static_cont_t, static_cat_t
            )

        self._last_attn = attn_weights[0].cpu().numpy()

        return float(p_win.item()), float(entry_offset.item())

    def get_last_attention_weights(self):
        """Return last forward's attention as (heads, 30, 30), or None."""
        return self._last_attn
