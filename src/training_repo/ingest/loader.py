"""Load raw offline episodes into normalized in-memory records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training_repo.common.io import read_json


REQUIRED_EPISODE_KEYS = {"episode_id", "task_id", "success", "steps"}
REQUIRED_STEP_KEYS = {"t", "obs_image_refs", "obs_state", "action", "intervention_flag", "terminal"}


def _validate_episode_shape(episode: dict[str, Any], path: Path) -> None:
    missing_keys = REQUIRED_EPISODE_KEYS.difference(episode.keys())
    if missing_keys:
        raise ValueError(f"{path} missing episode keys: {sorted(missing_keys)}")
    if not isinstance(episode["steps"], list) or not episode["steps"]:
        raise ValueError(f"{path} steps must be a non-empty list")

    for idx, step in enumerate(episode["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"{path} step[{idx}] must be an object")
        missing_step_keys = REQUIRED_STEP_KEYS.difference(step.keys())
        if missing_step_keys:
            raise ValueError(f"{path} step[{idx}] missing keys: {sorted(missing_step_keys)}")


def _normalize_episode(episode: dict[str, Any], source_path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for step in sorted(episode["steps"], key=lambda s: int(s["t"])):
        steps.append(
            {
                "t": int(step["t"]),
                "obs_image_refs": dict(step["obs_image_refs"]),
                "obs_state": list(step["obs_state"]),
                "action": list(step["action"]),
                "intervention_flag": bool(step["intervention_flag"]),
                "terminal": bool(step["terminal"]),
            }
        )

    return {
        "episode_id": str(episode["episode_id"]),
        "task_id": str(episode["task_id"]),
        "source_path": str(source_path),
        "num_steps": len(steps),
        "success": bool(episode["success"]),
        "steps": steps,
    }


def load_raw_episodes(raw_data_root: Path) -> list[dict[str, Any]]:
    if not raw_data_root.exists():
        raise FileNotFoundError(f"raw_data_root does not exist: {raw_data_root}")

    episode_paths = sorted(raw_data_root.rglob("*.json"))
    if not episode_paths:
        raise ValueError(f"No episode json files found under: {raw_data_root}")

    episodes: list[dict[str, Any]] = []
    for path in episode_paths:
        episode = read_json(path)
        if not isinstance(episode, dict):
            raise ValueError(f"{path} must contain a JSON object")
        _validate_episode_shape(episode, path)
        episodes.append(_normalize_episode(episode, path))

    return episodes

