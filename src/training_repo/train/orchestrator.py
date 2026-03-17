"""Config-driven train orchestration entrypoint for training_repo."""

from __future__ import annotations

from pathlib import Path

from training_repo.common.io import read_yaml
from training_repo.train.backend_factory import create_backend
from training_repo.train.interfaces import TrainResult


def run_training(config_path: Path) -> TrainResult:
    cfg = read_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    cfg["__config_dir__"] = str(config_path.parent.resolve())
    backend = create_backend(cfg)
    return backend.train(cfg)

