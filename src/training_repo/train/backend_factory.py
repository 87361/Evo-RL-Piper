"""Factory for selecting training backend from unified config."""

from __future__ import annotations

from typing import Any

from training_repo.backends.lerobot_backend import LerobotBackend
from training_repo.backends.openpi_backend import OpenPIBackend
from training_repo.train.interfaces import TrainingBackend

_LEROBOT_POLICIES = {"act", "diffusion", "pi0", "pi05"}
_OPENPI_BACKEND_NAMES = {"openpi", "openpi_jax", "openpi_torch"}


def _infer_backend_name(config: dict[str, Any]) -> str:
    raw_backend = str(config.get("backend", "")).strip().lower()
    if raw_backend in _OPENPI_BACKEND_NAMES:
        return "openpi"
    if raw_backend == "lerobot":
        return "lerobot"

    policy_cfg = config.get("policy", {})
    if isinstance(policy_cfg, dict):
        policy_type = str(policy_cfg.get("type", "")).strip().lower()
        if policy_type in _LEROBOT_POLICIES:
            return "lerobot"
        if policy_type.startswith("openpi"):
            return "openpi"

    if "openpi" in config:
        return "openpi"
    if "lerobot" in config:
        return "lerobot"
    raise ValueError("Cannot infer backend. Set `backend: openpi|lerobot` or provide policy.type.")


def create_backend(config: dict[str, Any]) -> TrainingBackend:
    backend_name = _infer_backend_name(config)
    if backend_name == "openpi":
        return OpenPIBackend()
    if backend_name == "lerobot":
        return LerobotBackend()
    raise ValueError(f"Unsupported backend: {backend_name}")

