#!/usr/bin/env python
"""Build minimal offline training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_repo.dataset_build.builder import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline dataset for training_repo v0.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_build.yaml"),
        help="Path to dataset build config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(args.config)
    print("dataset_build complete")
    print(result)


if __name__ == "__main__":
    main()

