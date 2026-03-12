from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

_BUNDLE_DIR = Path(__file__).resolve().parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

from preprocess import normalize_obs
from postprocess import denormalize_action

_REQUIRED_KEYS = ("weight", "bias", "obs_mean", "obs_std", "action_mean", "action_std")


def _validate_policy(policy: dict[str, np.ndarray]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in policy]
    if missing:
        raise ValueError(
            "Policy missing required keys: "
            f"{missing}. Expected keys: {list(_REQUIRED_KEYS)}."
        )


def _to_1d_obs(obs_state: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_state)
    if obs.ndim != 1:
        raise ValueError(
            "obs_state must be a 1D array with shape [obs_state_dim], "
            f"but got shape {tuple(obs.shape)}."
        )
    return obs


def _load_weights_locator(bundle_dir: Path) -> dict[str, Any]:
    locator_path = bundle_dir / "weights_locator.yaml"
    with locator_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_weight_path(bundle_dir: Path, override_path: str | None = None) -> Path:
    if override_path:
        return Path(override_path)
    locator = _load_weights_locator(bundle_dir)
    policy_path = Path(locator["policy_weights_path"])
    if policy_path.exists():
        return policy_path
    fallback = Path(locator["local_debug_fallback"])
    return fallback


def load_policy(bundle_dir: str, weight_path: str | None = None) -> dict[str, np.ndarray]:
    bundle_path = Path(bundle_dir)
    resolved = _resolve_weight_path(bundle_path, weight_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Cannot find weights file: {resolved}. "
            "Check weights_locator.yaml or pass weight_path explicitly."
        )
    ckpt = np.load(resolved)
    ckpt_keys = set(ckpt.files)
    missing = [key for key in _REQUIRED_KEYS if key not in ckpt_keys]
    if missing:
        raise ValueError(
            f"Invalid weights file: {resolved}. Missing arrays {missing}. "
            f"Found keys: {sorted(ckpt_keys)}."
        )
    policy = {
        "weight": ckpt["weight"],
        "bias": ckpt["bias"],
        "obs_mean": ckpt["obs_mean"],
        "obs_std": ckpt["obs_std"],
        "action_mean": ckpt["action_mean"],
        "action_std": ckpt["action_std"],
    }
    _validate_policy(policy)
    return policy


def predict_action(policy: dict[str, np.ndarray], obs_state: np.ndarray) -> np.ndarray:
    if isinstance(obs_state, dict):
        raise ValueError(
            "obs_state must be a 1D numeric array, not a dict. "
            "If your observation is a dict, extract the state vector first."
        )
    _validate_policy(policy)
    x = _to_1d_obs(obs_state)
    expected_obs_dim = int(policy["weight"].shape[0])
    if x.shape[0] != expected_obs_dim:
        raise ValueError(
            "obs_state dimension mismatch: "
            f"expected {expected_obs_dim}, got {x.shape[0]}."
        )
    x = normalize_obs(x, policy["obs_mean"], policy["obs_std"])
    y_norm = x @ policy["weight"] + policy["bias"]
    expected_action_dim = int(policy["weight"].shape[1])
    if y_norm.shape[0] != expected_action_dim:
        raise ValueError(
            "Predicted action dimension mismatch: "
            f"expected {expected_action_dim}, got {y_norm.shape[0]}."
        )
    y = denormalize_action(y_norm, policy["action_mean"], policy["action_std"])
    return y
