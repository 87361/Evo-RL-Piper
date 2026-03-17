#!/usr/bin/env python
"""Convert training_repo dataset package to a minimal LeRobot v3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    create_empty_dataset_info,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from training_repo.common.io import read_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert training_repo dataset to LeRobot v3 format.")
    parser.add_argument("--input-root", type=Path, required=True, help="training_repo dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="LeRobot dataset output root.")
    parser.add_argument("--fps", type=int, default=10, help="Dataset fps for timestamp conversion.")
    return parser.parse_args()


def _build_features(obs_dim: int, action_dim: int, fps: int) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [obs_dim],
            "names": [f"state_{i}" for i in range(obs_dim)],
            "fps": fps,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": [obs_dim],
            "names": [f"obs_{i}" for i in range(obs_dim)],
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"act_{i}" for i in range(action_dim)],
            "fps": fps,
        },
    }
    for key, value in DEFAULT_FEATURES.items():
        features[key] = {
            "dtype": value["dtype"],
            "shape": list(value["shape"]),
            "names": value["names"],
            "fps": fps,
        }
    return features


def convert(input_root: Path, output_root: Path, fps: int) -> dict:
    step_files = sorted((input_root / "steps").glob("shard-*.jsonl"))
    if not step_files:
        raise FileNotFoundError(f"No step shard files found under: {input_root / 'steps'}")

    episode_manifest = read_jsonl(input_root / "manifests" / "episode_manifest.jsonl")
    build_manifest = read_jsonl(input_root / "manifests" / "build_manifest.jsonl")
    stats = read_json(input_root / "meta" / "normalization_stats.json")

    if not episode_manifest:
        raise ValueError("episode_manifest is empty.")
    if not build_manifest:
        raise ValueError("build_manifest is empty.")

    # sample_id -> split
    sample_split = {row["sample_id"]: row["split"] for row in build_manifest}
    episode_task = {row["episode_id"]: row["task_id"] for row in episode_manifest}
    episode_ids = sorted(episode_task.keys())
    ep_to_idx = {ep: i for i, ep in enumerate(episode_ids)}

    task_names = sorted(set(episode_task.values()))
    task_df = pd.DataFrame({"task_index": list(range(len(task_names)))}, index=task_names)
    task_to_idx = {task: idx for idx, task in enumerate(task_names)}

    rows: list[dict] = []
    for shard in step_files:
        for step in read_jsonl(shard):
            ep_id = step["episode_id"]
            rows.append(
                {
                    "observation.state": np.asarray(step["obs_state"], dtype=np.float32),
                    "observation.environment_state": np.asarray(step["obs_state"], dtype=np.float32),
                    "action": np.asarray(step["action"], dtype=np.float32),
                    "timestamp": np.float32(float(step["t"]) / float(fps)),
                    "frame_index": np.int64(int(step["t"])),
                    "episode_index": np.int64(ep_to_idx[ep_id]),
                    "index": np.int64(0),  # filled after global sort
                    "task_index": np.int64(task_to_idx[episode_task[ep_id]]),
                    "_split": sample_split[step["sample_id"]],
                    "_episode_id": ep_id,
                    "_t": int(step["t"]),
                }
            )

    # Stable global order and global index.
    rows.sort(key=lambda x: (int(x["episode_index"]), x["_t"]))
    for i, row in enumerate(rows):
        row["index"] = np.int64(i)

    # LeRobot expects one parquet file with all frame-level rows.
    data_rows = []
    for row in rows:
        data_rows.append(
            {
                "observation.state": row["observation.state"],
                "observation.environment_state": row["observation.environment_state"],
                "action": row["action"],
                "timestamp": row["timestamp"],
                "frame_index": row["frame_index"],
                "episode_index": row["episode_index"],
                "index": row["index"],
                "task_index": row["task_index"],
            }
        )
    data_df = pd.DataFrame(data_rows)
    data_path = output_root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_parquet(data_path, index=False)

    # Build episode metadata.
    ep_bounds: dict[int, list[int]] = {}
    ep_split: dict[int, str] = {}
    for row in rows:
        ep_idx = int(row["episode_index"])
        global_idx = int(row["index"])
        ep_bounds.setdefault(ep_idx, [global_idx, global_idx + 1])
        ep_bounds[ep_idx][1] = global_idx + 1
        ep_split.setdefault(ep_idx, row["_split"])

    ep_records = []
    for ep_idx in range(len(episode_ids)):
        start, end = ep_bounds[ep_idx]
        ep_id = episode_ids[ep_idx]
        ep_records.append(
            {
                "episode_index": ep_idx,
                "tasks": [episode_task[ep_id]],
                "length": end - start,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": end,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )

    write_episodes(Dataset.from_list(ep_records), output_root)
    write_tasks(task_df, output_root)

    obs_stats = stats["obs_state"]
    act_stats = stats["action"]
    write_stats(
        {
            "observation.state": {
                "mean": np.asarray(obs_stats["mean"], dtype=np.float32),
                "std": np.asarray(obs_stats["std"], dtype=np.float32),
                "min": np.asarray(obs_stats["mean"], dtype=np.float32)
                - np.asarray(obs_stats["std"], dtype=np.float32),
                "max": np.asarray(obs_stats["mean"], dtype=np.float32)
                + np.asarray(obs_stats["std"], dtype=np.float32),
            },
            "observation.environment_state": {
                "mean": np.asarray(obs_stats["mean"], dtype=np.float32),
                "std": np.asarray(obs_stats["std"], dtype=np.float32),
                "min": np.asarray(obs_stats["mean"], dtype=np.float32)
                - np.asarray(obs_stats["std"], dtype=np.float32),
                "max": np.asarray(obs_stats["mean"], dtype=np.float32)
                + np.asarray(obs_stats["std"], dtype=np.float32),
            },
            "action": {
                "mean": np.asarray(act_stats["mean"], dtype=np.float32),
                "std": np.asarray(act_stats["std"], dtype=np.float32),
                "min": np.asarray(act_stats["mean"], dtype=np.float32)
                - np.asarray(act_stats["std"], dtype=np.float32),
                "max": np.asarray(act_stats["mean"], dtype=np.float32)
                + np.asarray(act_stats["std"], dtype=np.float32),
            },
        },
        output_root,
    )

    splits = {"train": "0:0", "val": "0:0"}
    train_eps = sorted(i for i, split in ep_split.items() if split == "train")
    val_eps = sorted(i for i, split in ep_split.items() if split == "val")
    if train_eps:
        splits["train"] = f"{train_eps[0]}:{train_eps[-1] + 1}"
    if val_eps:
        splits["val"] = f"{val_eps[0]}:{val_eps[-1] + 1}"

    features = _build_features(
        obs_dim=len(stats["obs_state"]["mean"]),
        action_dim=len(stats["action"]["mean"]),
        fps=fps,
    )
    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=fps,
        features=features,
        use_videos=False,
        robot_type="training_repo_converted",
    )
    info["total_episodes"] = len(episode_ids)
    info["total_frames"] = len(rows)
    info["total_tasks"] = len(task_names)
    info["splits"] = splits
    write_info(info, output_root)

    return {
        "output_root": str(output_root),
        "episodes": len(episode_ids),
        "frames": len(rows),
        "tasks": task_names,
        "train_split_range": splits["train"],
        "val_split_range": splits["val"],
    }


def main() -> None:
    args = parse_args()
    result = convert(args.input_root, args.output_root, args.fps)
    print("convert_training_repo_to_lerobot complete")
    print(result)


if __name__ == "__main__":
    main()
