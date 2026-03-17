from __future__ import annotations

from pathlib import Path

from training_repo.common.io import write_yaml
from training_repo.train.backend_factory import create_backend
from training_repo.train.orchestrator import run_training


def test_factory_infers_openpi_from_legacy_config() -> None:
    cfg = {
        "backend": "openpi_torch",
        "config_name": "pi0_aloha_sim",
        "openpi_root": "third_party/openpi",
    }
    backend = create_backend(cfg)
    assert backend.__class__.__name__ == "OpenPIBackend"


def test_factory_infers_lerobot_from_policy_type() -> None:
    cfg = {
        "policy": {"type": "act"},
        "dataset": {"repo_id": "demo"},
    }
    backend = create_backend(cfg)
    assert backend.__class__.__name__ == "LerobotBackend"


def test_run_training_dispatches_openpi_command(tmp_path: Path, monkeypatch) -> None:
    openpi_root = tmp_path / "third_party" / "openpi"
    script_path = openpi_root / "scripts" / "train_pytorch.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    config_path = tmp_path / "train_policy_openpi.yaml"
    write_yaml(
        config_path,
        {
            "backend": "openpi",
            "openpi": {
                "backend": "openpi_torch",
                "config_name": "pi0_aloha_sim",
                "openpi_root": "third_party/openpi",
                "extra_args": ["--overwrite"],
            },
        },
    )

    captured: dict[str, object] = {}

    def _fake_run(command, check, cwd):  # noqa: ANN001
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("training_repo.backends.openpi_backend.subprocess.run", _fake_run)
    result = run_training(config_path)

    assert captured["check"] is True
    assert str(script_path) in captured["command"]
    assert "pi0_aloha_sim" in captured["command"]
    assert captured["cwd"] == str(openpi_root)
    assert result.artifact_dir.name == "openpi"


def test_run_training_dispatches_lerobot_command(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "src" / "lerobot" / "scripts" / "lerobot_train.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    config_path = tmp_path / "train_policy_act.yaml"
    write_yaml(
        config_path,
        {
            "backend": "lerobot",
            "policy": {"type": "act", "device": "cuda"},
            "dataset": {"repo_id": "pipeline_ab/A", "root": "/tmp/dataset"},
            "train": {"steps": 10, "batch_size": 4, "job_name": "act_smoke"},
            "lerobot": {"script_path": "src/lerobot/scripts/lerobot_train.py", "extra_args": []},
        },
    )

    captured: dict[str, object] = {}

    def _fake_run(command, check, cwd):  # noqa: ANN001
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("training_repo.backends.lerobot_backend.subprocess.run", _fake_run)
    result = run_training(config_path)

    argv = captured["command"]
    assert captured["check"] is True
    assert str(script_path) in argv
    assert "--policy.type=act" in argv
    assert "--dataset.repo_id=pipeline_ab/A" in argv
    assert "--steps=10" in argv
    assert result.artifact_dir.name == "train"

