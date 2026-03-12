from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from training_repo.common.io import write_yaml
from training_repo.export.bundle import export_inference_bundle


def _write_mock_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weight=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        bias=np.array([0.5, -0.5], dtype=np.float32),
        obs_mean=np.array([0.0, 0.0], dtype=np.float32),
        obs_std=np.array([1.0, 1.0], dtype=np.float32),
        action_mean=np.array([0.0, 0.0], dtype=np.float32),
        action_std=np.array([1.0, 1.0], dtype=np.float32),
    )


def test_export_inference_bundle(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "openpi_v0" / "run_001"
    model_path = artifact_dir / "policy_v0_weights.npz"
    _write_mock_model(model_path)

    output_dir = tmp_path / "deploy" / "inference_bundle"
    cfg_path = tmp_path / "export.yaml"
    write_yaml(
        cfg_path,
        {
            "artifact_dir": str(artifact_dir),
            "artifact_weights_file": "policy_v0_weights.npz",
            "output_dir": str(output_dir),
            "tos_weights_path": "/mnt/tos/openpi_v0/run_001/policy_v0_weights.npz",
        },
    )

    result = export_inference_bundle(cfg_path)
    assert result["output_dir"] == str(output_dir)
    assert result["obs_dim"] == 2
    assert result["action_dim"] == 2
    assert result["artifact_weights_file"] == "policy_v0_weights.npz"

    expected_files = {
        "__init__.py",
        "inference_runner.py",
        "model_spec.yaml",
        "postprocess.py",
        "preprocess.py",
        "README_deploy.md",
        "requirements.txt",
        "weights_locator.yaml",
    }
    assert expected_files.issubset({p.name for p in output_dir.iterdir()})
    readme = (output_dir / "README_deploy.md").read_text(encoding="utf-8")
    assert "self-contained for runtime" in readme
    assert "NOT runtime prerequisites" in readme
    assert "pyproject.toml` + `uv.lock" in readme
    assert "uses only local `requirements.txt`" in readme

    sys.path.insert(0, str(output_dir))
    try:
        from inference_runner import load_policy, predict_action

        policy = load_policy(bundle_dir=str(output_dir), weight_path=str(model_path))
        action = predict_action(policy, np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(
            action,
            np.array([4.5, 5.5], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        with pytest.raises(ValueError, match="obs_state must be a 1D numeric array, not a dict"):
            predict_action(policy, {"obs_state": np.array([1.0, 1.0], dtype=np.float32)})

        with pytest.raises(ValueError, match="obs_state dimension mismatch"):
            predict_action(policy, np.array([1.0, 1.0, 1.0], dtype=np.float32))

        bad_model_path = artifact_dir / "bad_weights.npz"
        np.savez(bad_model_path, wrong=np.array([1.0], dtype=np.float32))
        with pytest.raises(ValueError, match="Invalid weights file"):
            load_policy(bundle_dir=str(output_dir), weight_path=str(bad_model_path))
    finally:
        sys.path.remove(str(output_dir))
