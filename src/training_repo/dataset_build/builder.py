"""Build minimal offline dataset artifacts for training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from training_repo.common.hash_utils import stable_config_hash
from training_repo.common.io import read_yaml, write_json, write_jsonl
from training_repo.common.schema import ALL_SAMPLE_TYPES
from training_repo.dataset_build.config import DatasetBuildConfig
from training_repo.ingest.loader import load_raw_episodes
from training_repo.relabel.apo import relabel_episode_steps


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    # Use config file directory as anchor for repo-relative paths.
    return (config_path.parent / path).resolve()


def _load_config(config_path: Path) -> DatasetBuildConfig:
    raw_cfg = read_yaml(config_path)
    raw_data_root = _resolve_path(str(raw_cfg["raw_data_root"]), config_path)
    output_root = _resolve_path(str(raw_cfg["output_root"]), config_path)
    return DatasetBuildConfig(
        raw_data_root=str(raw_data_root),
        output_root=str(output_root),
        schema_version=str(raw_cfg.get("schema_version", "v0.1.0")),
        val_ratio=float(raw_cfg.get("val_ratio", 0.1)),
        split_mode=str(raw_cfg.get("split_mode", "episode")).lower(),
        pre_intervention_k=int(raw_cfg.get("pre_intervention_k", 3)),
        shard_size=int(raw_cfg.get("shard_size", 2048)),
        random_seed=int(raw_cfg.get("random_seed", 42)),
    )


def _split_episodes(
    episodes: list[dict[str, Any]],
    val_ratio: float,
    random_seed: int,
    split_mode: str,
) -> dict[str, str]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if split_mode != "episode":
        raise ValueError(f"Unsupported split_mode: {split_mode}. Only 'episode' is supported in v0.")
    ids = [ep["episode_id"] for ep in episodes]
    rng = random.Random(random_seed)
    rng.shuffle(ids)
    val_count = int(len(ids) * val_ratio)
    val_ids = set(ids[:val_count])
    return {episode_id: ("val" if episode_id in val_ids else "train") for episode_id in ids}


def _compute_stats(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_records:
        raise ValueError("No train records available to compute stats.")

    obs_state = np.asarray([r["obs_state"] for r in train_records], dtype=np.float64)
    actions = np.asarray([r["action"] for r in train_records], dtype=np.float64)

    obs_std = obs_state.std(axis=0)
    action_std = actions.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    action_std[action_std < 1e-8] = 1.0

    return {
        "obs_state": {
            "mean": obs_state.mean(axis=0).tolist(),
            "std": obs_std.tolist(),
            "shape": [int(obs_state.shape[1])],
        },
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": action_std.tolist(),
            "shape": [int(actions.shape[1])],
        },
    }


def build_dataset(config_path: Path) -> dict[str, Any]:
    cfg = _load_config(config_path)
    if cfg.shard_size <= 0:
        raise ValueError("shard_size must be > 0")

    output_root = Path(cfg.output_root)

    episodes = load_raw_episodes(Path(cfg.raw_data_root))
    split_map = _split_episodes(episodes, cfg.val_ratio, cfg.random_seed, cfg.split_mode)

    episode_manifest: list[dict[str, Any]] = []
    build_manifest: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    expected_obs_dim: int | None = None
    expected_action_dim: int | None = None

    for episode in episodes:
        episode_manifest.append(
            {
                "episode_id": episode["episode_id"],
                "task_id": episode["task_id"],
                "source_path": episode["source_path"],
                "num_steps": episode["num_steps"],
                "success": episode["success"],
            }
        )

        relabeled_steps = relabel_episode_steps(episode, cfg.pre_intervention_k)
        for relabeled in relabeled_steps:
            sample_type = relabeled["sample_type"]
            if sample_type not in ALL_SAMPLE_TYPES:
                raise ValueError(f"Unsupported sample_type: {sample_type}")

            shard_id = len(step_records) // cfg.shard_size
            split = split_map[episode["episode_id"]]
            canonical_sample_id = f'{relabeled["episode_id"]}:{relabeled["t"]}'
            sample_id = str(relabeled["sample_id"])
            if sample_id != canonical_sample_id:
                raise ValueError(
                    "sample_id must follow '<episode_id>:<t>' format. "
                    f"Expected {canonical_sample_id}, got {sample_id}."
                )
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id in build result: {sample_id}")
            seen_sample_ids.add(sample_id)

            obs_dim = len(relabeled["obs_state"])
            action_dim = len(relabeled["action"])
            if expected_obs_dim is None:
                expected_obs_dim = obs_dim
            elif obs_dim != expected_obs_dim:
                raise ValueError(
                    f"Inconsistent obs_state dim for {sample_id}: "
                    f"expected {expected_obs_dim}, got {obs_dim}."
                )
            if expected_action_dim is None:
                expected_action_dim = action_dim
            elif action_dim != expected_action_dim:
                raise ValueError(
                    f"Inconsistent action dim for {sample_id}: "
                    f"expected {expected_action_dim}, got {action_dim}."
                )

            step_records.append(
                {
                    "sample_id": sample_id,
                    "episode_id": relabeled["episode_id"],
                    "t": relabeled["t"],
                    "obs_image_refs": relabeled["obs_image_refs"],
                    "obs_state": relabeled["obs_state"],
                    "action": relabeled["action"],
                    "intervention_flag": relabeled["intervention_flag"],
                    "terminal": relabeled["terminal"],
                }
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "sample_type": relabeled["sample_type"],
                    "label_source": relabeled["label_source"],
                }
            )
            build_manifest.append(
                {
                    "sample_id": sample_id,
                    "episode_id": relabeled["episode_id"],
                    "t": relabeled["t"],
                    "split": split,
                    "bucket": relabeled["sample_type"],
                    "shard_id": shard_id,
                }
            )

    steps_root = output_root / "steps"
    total_shards = (len(step_records) + cfg.shard_size - 1) // cfg.shard_size
    for shard_id in range(total_shards):
        shard_start = shard_id * cfg.shard_size
        shard_end = min(shard_start + cfg.shard_size, len(step_records))
        write_jsonl(steps_root / f"shard-{shard_id:05d}.jsonl", step_records[shard_start:shard_end])

    write_jsonl(output_root / "manifests" / "episode_manifest.jsonl", episode_manifest)
    write_jsonl(output_root / "manifests" / "build_manifest.jsonl", build_manifest)
    write_jsonl(output_root / "labels" / "sample_labels.jsonl", labels)

    train_sample_ids = {row["sample_id"] for row in build_manifest if row["split"] == "train"}
    train_records = [record for record in step_records if record["sample_id"] in train_sample_ids]
    normalization_stats = _compute_stats(train_records)
    write_json(output_root / "meta" / "normalization_stats.json", normalization_stats)

    dataset_meta = {
        "schema_version": cfg.schema_version,
        "action_dim": len(step_records[0]["action"]) if step_records else 0,
        "obs_spec": {
            "obs_image_refs": "dict[str, str]",
            "obs_state_dim": len(step_records[0]["obs_state"]) if step_records else 0,
        },
        "normalization_stats": "meta/normalization_stats.json",
        "build_config_hash": stable_config_hash(
            {
                "raw_data_root": cfg.raw_data_root,
                "schema_version": cfg.schema_version,
                "val_ratio": cfg.val_ratio,
                "split_mode": cfg.split_mode,
                "pre_intervention_k": cfg.pre_intervention_k,
                "shard_size": cfg.shard_size,
                "random_seed": cfg.random_seed,
            }
        ),
    }
    write_json(output_root / "meta" / "dataset_meta.json", dataset_meta)

    return {
        "output_root": str(output_root),
        "episodes": len(episode_manifest),
        "samples": len(step_records),
        "shards": total_shards,
    }

