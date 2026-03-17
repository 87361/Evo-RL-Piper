#!/usr/bin/env python
"""Unified training launcher for OpenPI and LeRobot policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_repo.train.orchestrator import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training from unified YAML config (openpi or lerobot policies)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(args.config)
    print(
        {
            "artifact_dir": str(result.artifact_dir),
            "final_loss": result.final_loss,
            "epochs": result.epochs,
            "num_samples": result.num_samples,
        }
    )


if __name__ == "__main__":
    main()

