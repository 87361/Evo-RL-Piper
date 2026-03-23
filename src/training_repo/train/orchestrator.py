"""Config-driven train orchestration entrypoint for training_repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training_repo.common.io import read_yaml
from training_repo.train.backend_factory import create_backend
from training_repo.train.interfaces import TrainResult

_SECTION_CONFIG_KEYS: dict[str, str] = {
    "policy": "policy_config",
    "dataset": "dataset_config",
    "train": "train_config",
    "lerobot": "lerobot_config",
    "openpi": "openpi_config",
}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(config_dir: Path, raw_path: str) -> Path:
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = (config_dir / config_path).resolve()
    return config_path


def _resolve_section_configs(cfg: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = dict(cfg)
    for section_name, section_config_key in _SECTION_CONFIG_KEYS.items():
        section_config_path_raw = resolved.get(section_config_key)
        if section_config_path_raw is None:
            continue
        if not isinstance(section_config_path_raw, str) or not section_config_path_raw.strip():
            raise ValueError(f"`{section_config_key}` must be a non-empty string path.")

        section_config_path = _resolve_config_path(config_dir, section_config_path_raw)
        section_from_file = read_yaml(section_config_path)
        if not isinstance(section_from_file, dict):
            raise ValueError(
                f"Section config `{section_config_key}` must be a mapping: {section_config_path}"
            )

        section_inline = resolved.get(section_name, {})
        if section_inline is None:
            section_inline = {}
        if not isinstance(section_inline, dict):
            raise ValueError(f"`{section_name}` section must be a mapping when provided.")

        resolved[section_name] = _deep_merge_dict(section_from_file, section_inline)

    return resolved


def run_training(config_path: Path) -> TrainResult:
    cfg = read_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    cfg = _resolve_section_configs(cfg, config_path.parent.resolve())
    cfg["__config_dir__"] = str(config_path.parent.resolve())
    backend = create_backend(cfg)
    return backend.train(cfg)

