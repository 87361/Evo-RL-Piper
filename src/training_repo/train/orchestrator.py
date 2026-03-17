"""Legacy train orchestration entrypoint for training_repo."""

from __future__ import annotations

from pathlib import Path
from training_repo.train.interfaces import TrainResult


def run_training(config_path: Path) -> TrainResult:
    raise RuntimeError(
        "training_repo legacy linear OpenPI backend has been removed. "
        "Use `python scripts/train_pi.py <config_name> [--backend openpi_jax|openpi_torch]` "
        "or pass `--config configs/train_pi0_openpi.yaml`."
    )

