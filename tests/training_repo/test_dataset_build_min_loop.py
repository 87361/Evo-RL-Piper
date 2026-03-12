from __future__ import annotations

import shutil
from pathlib import Path

from training_repo.backend_openpi.dataset_adapter import OpenPIDatasetAdapter
from training_repo.common.io import read_json, read_jsonl, write_yaml
from training_repo.dataset_build.builder import build_dataset


def _build_fixture_dataset(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests" / "fixtures" / "training_repo" / "raw_episodes"
    raw_root = tmp_path / "raw_episodes"
    shutil.copytree(fixture_root, raw_root)

    config_dir = tmp_path / "configs"
    config_path = config_dir / "dataset_build.yaml"
    write_yaml(
        config_path,
        {
            "raw_data_root": "../raw_episodes",
            "output_root": "../processed/openpi_v0",
            "schema_version": "v0.1.0",
            "val_ratio": 0.5,
            "split_mode": "episode",
            "pre_intervention_k": 2,
            "shard_size": 2,
            "random_seed": 42,
        },
    )
    build_dataset(config_path)
    return (config_dir / "../processed/openpi_v0").resolve()


def test_relabel_outputs_three_buckets_and_pre_intervention_label(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    labels = read_jsonl(dataset_root / "labels" / "sample_labels.jsonl")
    sample_types = {row["sample_type"] for row in labels}
    label_sources = {row["label_source"] for row in labels}

    assert sample_types == {"correct", "interaction", "incorrect"}
    assert "pre_intervention_relabel" in label_sources


def test_split_is_episode_level(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    build_rows = read_jsonl(dataset_root / "manifests" / "build_manifest.jsonl")

    episode_splits: dict[str, set[str]] = {}
    for row in build_rows:
        episode_splits.setdefault(row["episode_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in episode_splits.values())
    assert {row["split"] for row in build_rows}.issubset({"train", "val"})


def test_sample_id_unique_globally_and_cross_file_sets_match(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    build_rows = read_jsonl(dataset_root / "manifests" / "build_manifest.jsonl")
    label_rows = read_jsonl(dataset_root / "labels" / "sample_labels.jsonl")

    step_rows = []
    for shard_file in sorted((dataset_root / "steps").glob("shard-*.jsonl")):
        step_rows.extend(read_jsonl(shard_file))

    build_ids = [row["sample_id"] for row in build_rows]
    label_ids = [row["sample_id"] for row in label_rows]
    step_ids = [row["sample_id"] for row in step_rows]

    assert len(build_ids) == len(set(build_ids))
    assert len(label_ids) == len(set(label_ids))
    assert len(step_ids) == len(set(step_ids))
    assert set(build_ids) == set(label_ids) == set(step_ids)


def test_openpi_min_contract_and_adapter_readable(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    build_rows = read_jsonl(dataset_root / "manifests" / "build_manifest.jsonl")
    label_rows = read_jsonl(dataset_root / "labels" / "sample_labels.jsonl")
    stats = read_json(dataset_root / "meta" / "normalization_stats.json")

    required_manifest_keys = {"sample_id", "episode_id", "t", "split", "bucket", "shard_id"}
    required_label_keys = {"sample_id", "sample_type", "label_source"}
    required_stats_keys = {"obs_state", "action"}

    assert required_manifest_keys.issubset(build_rows[0].keys())
    assert required_label_keys.issubset(label_rows[0].keys())
    assert required_stats_keys.issubset(stats.keys())

    label_by_id = {row["sample_id"]: row for row in label_rows}
    for row in build_rows:
        assert row["bucket"] == label_by_id[row["sample_id"]]["sample_type"]
        assert row["split"] in {"train", "val"}
        assert row["bucket"] in {"correct", "interaction", "incorrect"}

    adapter = OpenPIDatasetAdapter(dataset_root)
    train_samples = adapter.load_split("train")
    val_samples = adapter.load_split("val")

    assert train_samples
    assert val_samples
    sample = train_samples[0]
    required_adapter_keys = {
        "sample_id",
        "episode_id",
        "t",
        "bucket",
        "obs_image_refs",
        "obs_state",
        "action",
        "intervention_flag",
        "terminal",
        "sample_type",
        "label_source",
    }
    assert required_adapter_keys.issubset(sample.keys())


def test_stats_shape_matches_step_vectors(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    stats = read_json(dataset_root / "meta" / "normalization_stats.json")
    first_shard = sorted((dataset_root / "steps").glob("shard-*.jsonl"))[0]
    first_row = read_jsonl(first_shard)[0]

    assert len(stats["obs_state"]["mean"]) == len(first_row["obs_state"])
    assert len(stats["obs_state"]["std"]) == len(first_row["obs_state"])
    assert len(stats["action"]["mean"]) == len(first_row["action"])
    assert len(stats["action"]["std"]) == len(first_row["action"])
