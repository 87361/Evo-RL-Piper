"""OpenPI backend that forwards to third_party/openpi scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from training_repo.train.interfaces import TrainResult, TrainingBackend


class OpenPIBackend(TrainingBackend):
    def train(self, config: dict[str, Any]) -> TrainResult:
        cfg_dir = Path(config.get("__config_dir__", ".")).resolve()

        legacy_backend = str(config.get("backend", ""))
        openpi_cfg = config.get("openpi", {})
        if not isinstance(openpi_cfg, dict):
            raise ValueError("`openpi` section must be a mapping when provided.")

        selected_backend = str(openpi_cfg.get("backend", legacy_backend or "openpi_torch"))
        if selected_backend not in {"openpi_jax", "openpi_torch"}:
            raise ValueError(f"Unsupported OpenPI backend: {selected_backend}")

        config_name = openpi_cfg.get("config_name", config.get("config_name"))
        if not config_name:
            raise ValueError("Missing OpenPI config_name.")

        openpi_root_raw = openpi_cfg.get("openpi_root", config.get("openpi_root", "third_party/openpi"))
        openpi_root = Path(str(openpi_root_raw))
        if not openpi_root.is_absolute():
            openpi_root = (cfg_dir / openpi_root).resolve()

        script_rel = "scripts/train.py" if selected_backend == "openpi_jax" else "scripts/train_pytorch.py"
        script_path = openpi_root / script_rel
        if not script_path.exists():
            raise FileNotFoundError(
                f"OpenPI training entry not found: {script_path}. "
                "Please ensure third_party/openpi is initialized."
            )

        extra_args_raw = openpi_cfg.get("extra_args", config.get("extra_args", []))
        if not isinstance(extra_args_raw, list):
            raise ValueError("OpenPI extra_args must be a list.")
        extra_args = [str(x) for x in extra_args_raw]

        command = [sys.executable, str(script_path), str(config_name), *extra_args]
        subprocess.run(command, check=True, cwd=str(openpi_root))

        artifact_dir = Path(str(config.get("artifact_dir", "artifacts/openpi"))).resolve()
        return TrainResult(
            artifact_dir=artifact_dir,
            final_loss=float("nan"),
            epochs=0,
            num_samples=0,
        )

