"""OpenPI dataset adapter over built dataset artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training_repo.common.io import read_json, read_jsonl


def _load_step_table(dataset_root: Path) -> dict[str, dict[str, Any]]:
    step_table: dict[str, dict[str, Any]] = {}
    for shard_file in sorted((dataset_root / "steps").glob("shard-*.jsonl")):
        for row in read_jsonl(shard_file):
            step_table[row["sample_id"]] = row
    return step_table


class OpenPIDatasetAdapter:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self._step_table = _load_step_table(dataset_root)
        self._index_rows = read_jsonl(dataset_root / "manifests" / "build_manifest.jsonl")
        self._label_rows = read_jsonl(dataset_root / "labels" / "sample_labels.jsonl")
        self._stats = read_json(dataset_root / "meta" / "normalization_stats.json")
        self._label_by_id = {row["sample_id"]: row for row in self._label_rows}

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats

    def load_split(self, split: str) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for row in self._index_rows:
            if row["split"] != split:
                continue
            sample_id = row["sample_id"]
            step = self._step_table[sample_id]
            label = self._label_by_id[sample_id]
            samples.append(
                {
                    "sample_id": sample_id,
                    "episode_id": row["episode_id"],
                    "t": row["t"],
                    "bucket": row["bucket"],
                    "obs_image_refs": step["obs_image_refs"],
                    "obs_state": step["obs_state"],
                    "action": step["action"],
                    "intervention_flag": step["intervention_flag"],
                    "terminal": step["terminal"],
                    "sample_type": label["sample_type"],
                    "label_source": label["label_source"],
                }
            )
        return samples

