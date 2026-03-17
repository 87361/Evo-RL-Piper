"""Export a self-contained inference bundle for TeleManipulation delivery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training_repo.common.io import read_yaml, write_yaml


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")


def export_inference_bundle(config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    artifact_dir = Path(cfg["artifact_dir"])
    output_dir = Path(cfg.get("output_dir", "deploy/inference_bundle"))
    tos_weights_path = str(cfg["tos_weights_path"])
    artifact_weights_file = str(cfg.get("artifact_weights_file", "model.npz"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / artifact_weights_file
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing trained weights at "
            f"{model_path}. Check `artifact_weights_file` in export config "
            "or point to an existing legacy linear-policy artifact."
        )
    model = np.load(model_path)
    weight = model["weight"]
    bias = model["bias"]
    obs_dim = int(weight.shape[0])
    action_dim = int(weight.shape[1])

    model_spec = {
        "format_version": "v0.1.0",
        "backend": "openpi",
        "weights_format": "npz",
        "weights_file_hint": artifact_weights_file,
        "input": {"obs_state_dim": obs_dim},
        "output": {"action_dim": action_dim},
        "preprocess": {
            "obs_mean_key": "obs_mean",
            "obs_std_key": "obs_std",
        },
        "postprocess": {
            "action_mean_key": "action_mean",
            "action_std_key": "action_std",
        },
    }
    write_yaml(output_dir / "model_spec.yaml", model_spec)

    weights_locator = {
        "source": "tos_mount",
        "policy_weights_path": tos_weights_path,
        "local_debug_fallback": str(model_path),
    }
    write_yaml(output_dir / "weights_locator.yaml", weights_locator)

    _write_text(output_dir / "__init__.py", '"""Inference bundle package."""')

    _write_text(
        output_dir / "preprocess.py",
        """from __future__ import annotations

import numpy as np


def normalize_obs(obs_state: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    return (obs_state - obs_mean) / obs_std
""",
    )

    _write_text(
        output_dir / "postprocess.py",
        """from __future__ import annotations

import numpy as np


def denormalize_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return action_norm * action_std + action_mean
""",
    )

    _write_text(
        output_dir / "inference_runner.py",
        """from __future__ import annotations

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
""",
    )

    _write_text(
        output_dir / "README_deploy.md",
        """# Inference Bundle Deploy Guide

## Purpose
Copy this folder into TeleManipulation and call `inference_runner.load_policy()` and
`inference_runner.predict_action()`.

## Runtime dependency boundary
- This inference bundle is self-contained for runtime.
- OpenPI / APO / RLinf / TeleManipulation repository paths are development references only.
- They are NOT runtime prerequisites for this copied bundle.

## Environment split (important)
- Training repository environment uses `pyproject.toml` + `uv.lock`.
- Copied inference bundle runtime uses only local `requirements.txt`.
- Do not assume training dependencies are available inside TeleManipulation runtime.

## Quick start
1. Copy this whole directory into TeleManipulation repository.
2. Confirm mounted TOS path in `weights_locator.yaml`.
3. Install minimal dependencies: `pip install -r requirements.txt`
4. In TeleManipulation runtime:
   - `from inference_runner import load_policy, predict_action`
   - `policy = load_policy(bundle_dir="deploy/inference_bundle")`
   - `action = predict_action(policy, obs_state)`
5. If default TOS mount path is unavailable, pass an explicit path:
   - `policy = load_policy(bundle_dir="deploy/inference_bundle", weight_path="/path/to/model.npz")`

## Weight policy
- Weights are NOT shipped in code repository.
- Runtime should read from mounted TOS path by default.
""",
    )

    _write_text(output_dir / "requirements.txt", "numpy>=1.24.0\nPyYAML>=6.0\n")

    files = [
        "__init__.py",
        "inference_runner.py",
        "model_spec.yaml",
        "postprocess.py",
        "preprocess.py",
        "README_deploy.md",
        "requirements.txt",
        "weights_locator.yaml",
    ]
    return {
        "output_dir": str(output_dir),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "artifact_weights_file": artifact_weights_file,
        "files": files,
    }

