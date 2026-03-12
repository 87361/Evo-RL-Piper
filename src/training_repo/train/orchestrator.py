"""Train orchestration entrypoint for training_repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training_repo.backend_openpi.backend import OpenPIBackend
from training_repo.common.io import read_yaml
from training_repo.train.interfaces import TrainResult


def run_training(config_path: Path) -> TrainResult:
    config: dict[str, Any] = read_yaml(config_path)
    backend_name = str(config.get("backend", "openpi")).lower()
    if backend_name != "openpi":
        raise ValueError(f"Unsupported backend: {backend_name}. Only 'openpi' is available in v0.")

    backend = OpenPIBackend()
    return backend.train(config)

