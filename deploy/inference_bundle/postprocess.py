from __future__ import annotations

import numpy as np


def denormalize_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return action_norm * action_std + action_mean
