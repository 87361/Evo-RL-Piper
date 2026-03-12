#!/usr/bin/env python
"""Smoke check for minimal OpenPI-readable dataset package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from training_repo.backend_openpi.dataset_adapter import OpenPIDatasetAdapter
from training_repo.common.io import read_json, read_jsonl


def _load_step_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_file in sorted((dataset_root / "steps").glob("shard-*.jsonl")):
        rows.extend(read_jsonl(shard_file))
    return rows


def _assert_min_contract(dataset_root: Path) -> None:
    required_files = [
        dataset_root / "manifests" / "build_manifest.jsonl",
        dataset_root / "labels" / "sample_labels.jsonl",
        dataset_root / "meta" / "normalization_stats.json",
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file missing: {file_path}")

    build_rows = read_jsonl(dataset_root / "manifests" / "build_manifest.jsonl")
    label_rows = read_jsonl(dataset_root / "labels" / "sample_labels.jsonl")
    stats = read_json(dataset_root / "meta" / "normalization_stats.json")
    step_rows = _load_step_rows(dataset_root)
    if not step_rows:
        raise ValueError("No step shards found under steps/shard-*.jsonl")

    required_manifest_keys = {"sample_id", "episode_id", "t", "split", "bucket", "shard_id"}
    required_step_keys = {
        "sample_id",
        "episode_id",
        "t",
        "obs_image_refs",
        "obs_state",
        "action",
        "intervention_flag",
        "terminal",
    }
    required_label_keys = {"sample_id", "sample_type", "label_source"}
    required_stats_keys = {"obs_state", "action"}

    if not build_rows:
        raise ValueError("build_manifest.jsonl is empty")
    if not label_rows:
        raise ValueError("sample_labels.jsonl is empty")

    if not required_manifest_keys.issubset(build_rows[0].keys()):
        raise ValueError("build_manifest.jsonl does not match minimal OpenPI contract")
    if not required_step_keys.issubset(step_rows[0].keys()):
        raise ValueError("steps shard rows do not match minimal OpenPI contract")
    if not required_label_keys.issubset(label_rows[0].keys()):
        raise ValueError("sample_labels.jsonl does not match minimal OpenPI contract")
    if not required_stats_keys.issubset(stats.keys()):
        raise ValueError("normalization_stats.json does not match minimal OpenPI contract")

    build_ids = {row["sample_id"] for row in build_rows}
    step_ids = {row["sample_id"] for row in step_rows}
    label_ids = {row["sample_id"] for row in label_rows}
    if build_ids != step_ids or build_ids != label_ids:
        raise ValueError("sample_id sets mismatch across build_manifest/steps/labels")

    label_by_id = {row["sample_id"]: row for row in label_rows}
    for row in build_rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Invalid split value: {row['split']}")
        if row["bucket"] not in {"correct", "interaction", "incorrect"}:
            raise ValueError(f"Invalid bucket value: {row['bucket']}")
        if row["bucket"] != label_by_id[row["sample_id"]]["sample_type"]:
            raise ValueError(
                f"bucket and sample_type mismatch for sample_id={row['sample_id']}"
            )

    sample_row = step_rows[0]
    if len(stats["obs_state"]["mean"]) != len(sample_row["obs_state"]):
        raise ValueError("obs_state stats dim mismatch")
    if len(stats["action"]["mean"]) != len(sample_row["action"]):
        raise ValueError("action stats dim mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check for OpenPI-readable dataset package.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Built dataset root path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    _assert_min_contract(dataset_root)

    adapter = OpenPIDatasetAdapter(dataset_root)
    train_samples = adapter.load_split("train")
    if not train_samples:
        raise ValueError("No train samples found. Check val_ratio or source episodes.")

    print("dataset_openpi_smoke passed")
    print(
        {
            "dataset_root": str(dataset_root),
            "num_train_samples": len(train_samples),
            "stats_keys": sorted(adapter.stats.keys()),
        }
    )


if __name__ == "__main__":
    main()
