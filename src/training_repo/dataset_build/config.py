"""Configuration types for dataset build."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetBuildConfig:
    raw_data_root: str
    output_root: str
    schema_version: str
    val_ratio: float
    split_mode: str
    pre_intervention_k: int
    shard_size: int
    random_seed: int

