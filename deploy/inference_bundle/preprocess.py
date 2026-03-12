from __future__ import annotations

import numpy as np


def normalize_obs(obs_state: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    return (obs_state - obs_mean) / obs_std
