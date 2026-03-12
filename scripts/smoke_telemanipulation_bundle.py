#!/usr/bin/env python
"""Smoke test for simulated TeleManipulation bundle integration."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated TeleManipulation integration smoke test.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("deploy/inference_bundle"),
        help="Path to exported inference bundle directory.",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("tmp/telemanipulation_smoke"),
        help="Root path for simulated TeleManipulation workspace.",
    )
    return parser.parse_args()


def _assert_bundle_files(bundle_dir: Path) -> None:
    required = {
        "__init__.py",
        "inference_runner.py",
        "model_spec.yaml",
        "postprocess.py",
        "preprocess.py",
        "README_deploy.md",
        "requirements.txt",
        "weights_locator.yaml",
    }
    existing = {p.name for p in bundle_dir.iterdir()}
    missing = sorted(required - existing)
    if missing:
        raise FileNotFoundError(f"Inference bundle missing required files: {missing}")


def _copy_bundle(bundle_dir: Path, sim_root: Path) -> Path:
    target_bundle = sim_root / "deploy" / "inference_bundle"
    if target_bundle.exists():
        shutil.rmtree(target_bundle)
    target_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_dir, target_bundle)
    return target_bundle


def _write_mock_weights(path: Path, obs_dim: int, action_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weight=np.ones((obs_dim, action_dim), dtype=np.float32),
        bias=np.zeros(action_dim, dtype=np.float32),
        obs_mean=np.zeros(obs_dim, dtype=np.float32),
        obs_std=np.ones(obs_dim, dtype=np.float32),
        action_mean=np.zeros(action_dim, dtype=np.float32),
        action_std=np.ones(action_dim, dtype=np.float32),
    )


def _expect_value_error(fn, expected_substring: str) -> None:
    try:
        fn()
    except ValueError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(
                f"Expected error containing `{expected_substring}`, got: {exc}"
            ) from exc
        return
    raise AssertionError(f"Expected ValueError containing `{expected_substring}`.")


def main() -> None:
    args = _parse_args()
    bundle_dir = args.bundle_dir.resolve()
    sim_root = args.sim_root.resolve()

    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"Bundle directory does not exist: {bundle_dir}. "
            "Run scripts/export_inference_bundle.py first."
        )

    _assert_bundle_files(bundle_dir)
    sim_bundle = _copy_bundle(bundle_dir, sim_root)

    with (sim_bundle / "model_spec.yaml").open("r", encoding="utf-8") as f:
        model_spec = yaml.safe_load(f)
    obs_dim = int(model_spec["input"]["obs_state_dim"])
    action_dim = int(model_spec["output"]["action_dim"])

    mock_weight_path = sim_root / "weights" / "smoke_valid_weights.npz"
    _write_mock_weights(mock_weight_path, obs_dim, action_dim)

    bad_weight_path = sim_root / "weights" / "smoke_invalid_weights.npz"
    np.savez(bad_weight_path, wrong=np.array([1.0], dtype=np.float32))

    sys.path.insert(0, str(sim_bundle))
    try:
        from inference_runner import load_policy, predict_action

        policy = load_policy(bundle_dir=str(sim_bundle), weight_path=str(mock_weight_path))
        action = predict_action(policy, np.zeros(obs_dim, dtype=np.float32))
        if action.shape != (action_dim,):
            raise AssertionError(
                f"Expected action shape {(action_dim,)}, got {tuple(action.shape)}."
            )

        _expect_value_error(
            lambda: predict_action(policy, {"obs_state": np.zeros(obs_dim, dtype=np.float32)}),
            "obs_state must be a 1D numeric array, not a dict",
        )
        _expect_value_error(
            lambda: predict_action(policy, np.zeros(obs_dim + 1, dtype=np.float32)),
            "obs_state dimension mismatch",
        )
        _expect_value_error(
            lambda: load_policy(bundle_dir=str(sim_bundle), weight_path=str(bad_weight_path)),
            "Invalid weights file",
        )
    finally:
        sys.path.remove(str(sim_bundle))

    print("telemanipulation bundle smoke passed")
    print(
        {
            "sim_bundle": str(sim_bundle),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "checks": [
                "basic load + predict",
                "dict obs_state clear error",
                "shape mismatch clear error",
                "invalid weights clear error",
            ],
        }
    )


if __name__ == "__main__":
    main()
