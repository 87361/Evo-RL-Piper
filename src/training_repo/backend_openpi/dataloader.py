"""Minimal array preparation helpers for OpenPI backend training."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_training_arrays(
    train_samples: list[dict[str, Any]],
    stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert adapter samples into normalized arrays for training loop."""
    if not train_samples:
        raise ValueError("No train samples found. Build dataset first.")

    obs_mean = np.asarray(stats["obs_state"]["mean"], dtype=np.float64)
    obs_std = np.asarray(stats["obs_state"]["std"], dtype=np.float64)
    action_mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
    action_std = np.asarray(stats["action"]["std"], dtype=np.float64)

    x_all = np.asarray([sample["obs_state"] for sample in train_samples], dtype=np.float64)
    y_all = np.asarray([sample["action"] for sample in train_samples], dtype=np.float64)
    x_all = (x_all - obs_mean) / obs_std
    y_all = (y_all - action_mean) / action_std

    sample_types = [sample["sample_type"] for sample in train_samples]
    return x_all, y_all, sample_types
