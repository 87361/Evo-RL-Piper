#!/usr/bin/env python
"""Train with OpenPI backend adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_repo.train.orchestrator import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training using OpenPI backend.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_openpi.yaml"),
        help="Path to training config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(args.config)
    print("training complete")
    print(
        {
            "artifact_dir": str(result.artifact_dir),
            "epochs": result.epochs,
            "num_samples": result.num_samples,
            "final_loss": result.final_loss,
        }
    )


if __name__ == "__main__":
    main()

