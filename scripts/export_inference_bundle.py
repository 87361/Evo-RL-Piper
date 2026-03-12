#!/usr/bin/env python
"""Export self-contained inference bundle for deployment handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from training_repo.export.bundle import export_inference_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deployable inference bundle.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/export_inference.yaml"),
        help="Path to export config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_inference_bundle(args.config)
    print("export complete")
    print(
        {
            "output_dir": result["output_dir"],
            "obs_dim": result["obs_dim"],
            "action_dim": result["action_dim"],
            "artifact_weights_file": result["artifact_weights_file"],
            "files": result["files"],
        }
    )


if __name__ == "__main__":
    main()

