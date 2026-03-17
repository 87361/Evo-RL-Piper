"""LeRobot backend that forwards to lerobot_train CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from training_repo.train.interfaces import TrainResult, TrainingBackend


def _to_cli_flag(name: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, list):
        text = "[" + ",".join(str(v) for v in value) + "]"
    else:
        text = str(value)
    return f"--{name}={text}"


class LerobotBackend(TrainingBackend):
    def train(self, config: dict[str, Any]) -> TrainResult:
        cfg_dir = Path(config.get("__config_dir__", ".")).resolve()
        lerobot_cfg = config.get("lerobot", {})
        if not isinstance(lerobot_cfg, dict):
            raise ValueError("`lerobot` section must be a mapping when provided.")

        script_path_raw = lerobot_cfg.get("script_path", "src/lerobot/scripts/lerobot_train.py")
        script_path = Path(str(script_path_raw))
        if not script_path.is_absolute():
            script_path = (cfg_dir / script_path).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"LeRobot train entry not found: {script_path}")

        command = [sys.executable, str(script_path)]

        dataset_cfg = config.get("dataset", {})
        if isinstance(dataset_cfg, dict):
            for key, value in dataset_cfg.items():
                command.append(_to_cli_flag(f"dataset.{key}", value))

        policy_cfg = config.get("policy", {})
        if isinstance(policy_cfg, dict):
            for key, value in policy_cfg.items():
                command.append(_to_cli_flag(f"policy.{key}", value))

        train_cfg = config.get("train", {})
        if isinstance(train_cfg, dict):
            for key, value in train_cfg.items():
                command.append(_to_cli_flag(key, value))

        extra_args_raw = lerobot_cfg.get("extra_args", [])
        if not isinstance(extra_args_raw, list):
            raise ValueError("LeRobot extra_args must be a list.")
        command.extend(str(x) for x in extra_args_raw)

        run_cwd_raw = lerobot_cfg.get("cwd", ".")
        run_cwd = Path(str(run_cwd_raw))
        if not run_cwd.is_absolute():
            run_cwd = (cfg_dir / run_cwd).resolve()
        subprocess.run(command, check=True, cwd=str(run_cwd))

        output_dir_raw = None
        if isinstance(train_cfg, dict):
            output_dir_raw = train_cfg.get("output_dir")
        output_dir = Path(str(output_dir_raw or "outputs/train"))
        if not output_dir.is_absolute():
            output_dir = (run_cwd / output_dir).resolve()

        return TrainResult(
            artifact_dir=output_dir,
            final_loss=float("nan"),
            epochs=0,
            num_samples=0,
        )

