"""Minimal OpenPI backend implementation for offline training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training_repo.backend_openpi.dataset_adapter import OpenPIDatasetAdapter
from training_repo.backend_openpi.sampler import ProportionalBucketSampler, SamplerConfig
from training_repo.common.io import write_json
from training_repo.train.interfaces import TrainResult, TrainingBackend


class OpenPIBackend(TrainingBackend):
    def train(self, config: dict[str, Any]) -> TrainResult:
        dataset_root = Path(config["dataset_root"])
        artifact_dir = Path(config["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)

        adapter = OpenPIDatasetAdapter(dataset_root)
        train_samples = adapter.load_split("train")
        if not train_samples:
            raise ValueError("No train samples found. Build dataset first.")

        obs_mean = np.asarray(adapter.stats["obs_state"]["mean"], dtype=np.float64)
        obs_std = np.asarray(adapter.stats["obs_state"]["std"], dtype=np.float64)
        action_mean = np.asarray(adapter.stats["action"]["mean"], dtype=np.float64)
        action_std = np.asarray(adapter.stats["action"]["std"], dtype=np.float64)

        x_all = np.asarray([sample["obs_state"] for sample in train_samples], dtype=np.float64)
        y_all = np.asarray([sample["action"] for sample in train_samples], dtype=np.float64)
        x_all = (x_all - obs_mean) / obs_std
        y_all = (y_all - action_mean) / action_std

        sample_types = [sample["sample_type"] for sample in train_samples]
        sampler = ProportionalBucketSampler(
            sample_types,
            SamplerConfig(
                batch_size=int(config.get("batch_size", 64)),
                ratio_correct=int(config.get("ratio_correct", 1)),
                ratio_incorrect=int(config.get("ratio_incorrect", 1)),
                ratio_interaction=int(config.get("ratio_interaction", 1)),
                random_seed=int(config.get("random_seed", 42)),
            ),
        )

        obs_dim = x_all.shape[1]
        action_dim = y_all.shape[1]
        lr = float(config.get("learning_rate", 1e-2))
        epochs = int(config.get("epochs", 10))
        steps_per_epoch = int(config.get("steps_per_epoch", 100))

        rng = np.random.default_rng(int(config.get("random_seed", 42)))
        w = rng.normal(loc=0.0, scale=0.02, size=(obs_dim, action_dim))
        b = np.zeros(action_dim, dtype=np.float64)

        losses: list[float] = []
        for _ in range(epochs):
            epoch_losses: list[float] = []
            for batch_indices in sampler.iter_batches(steps_per_epoch):
                xb = x_all[batch_indices]
                yb = y_all[batch_indices]
                pred = xb @ w + b
                diff = pred - yb
                loss = float(np.mean(diff**2))
                epoch_losses.append(loss)

                grad_w = (xb.T @ diff) * (2.0 / xb.shape[0])
                grad_b = np.mean(2.0 * diff, axis=0)
                w -= lr * grad_w
                b -= lr * grad_b

            losses.append(float(np.mean(epoch_losses)))

        np.savez(
            artifact_dir / "model.npz",
            weight=w.astype(np.float32),
            bias=b.astype(np.float32),
            obs_mean=obs_mean.astype(np.float32),
            obs_std=obs_std.astype(np.float32),
            action_mean=action_mean.astype(np.float32),
            action_std=action_std.astype(np.float32),
        )

        metrics = {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "num_train_samples": len(train_samples),
            "train_loss_history": losses,
            "final_train_loss": losses[-1] if losses else None,
        }
        write_json(artifact_dir / "metrics.json", metrics)
        write_json(artifact_dir / "train_config_snapshot.json", config)

        return TrainResult(
            artifact_dir=artifact_dir,
            final_loss=float(losses[-1]),
            epochs=epochs,
            num_samples=len(train_samples),
        )

