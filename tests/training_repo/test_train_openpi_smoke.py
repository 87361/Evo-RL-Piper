from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from training_repo.common.io import read_json, write_yaml
from training_repo.dataset_build.builder import build_dataset
from training_repo.train.orchestrator import run_training


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


def _snapshot_dataset_files(dataset_root: Path) -> dict[str, str]:
    tracked_files: list[Path] = []
    tracked_files.extend(sorted((dataset_root / "manifests").glob("*.jsonl")))
    tracked_files.extend(sorted((dataset_root / "labels").glob("*.jsonl")))
    tracked_files.extend(sorted((dataset_root / "meta").glob("*.json")))
    tracked_files.extend(sorted((dataset_root / "steps").glob("shard-*.jsonl")))

    snapshot: dict[str, str] = {}
    for file_path in tracked_files:
        rel_path = str(file_path.relative_to(dataset_root))
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        snapshot[rel_path] = digest
    return snapshot


def test_openpi_train_min_smoke_and_no_dataset_mutation(tmp_path: Path) -> None:
    dataset_root = _build_fixture_dataset(tmp_path)
    before_snapshot = _snapshot_dataset_files(dataset_root)

    config_dir = tmp_path / "configs"
    train_config_path = config_dir / "train_openpi_smoke.yaml"
    write_yaml(
        train_config_path,
        {
            "backend": "openpi",
            "dataset_root": str(dataset_root),
            "artifact_dir": str((tmp_path / "artifacts" / "openpi_v0" / "run_smoke").resolve()),
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "ratio_correct": 0,
            "ratio_incorrect": 1,
            "ratio_interaction": 1,
            "random_seed": 42,
        },
    )

    result = run_training(train_config_path)
    artifact_dir = result.artifact_dir

    assert (artifact_dir / "model.npz").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "train_config_snapshot.json").exists()

    metrics = read_json(artifact_dir / "metrics.json")
    assert isinstance(metrics["train_loss_history"], list)
    assert len(metrics["train_loss_history"]) == 1
    assert isinstance(metrics["final_train_loss"], (int, float))

    after_snapshot = _snapshot_dataset_files(dataset_root)
    assert before_snapshot == after_snapshot
