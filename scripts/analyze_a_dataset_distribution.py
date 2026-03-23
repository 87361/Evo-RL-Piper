from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch

import openpi.training.config as training_config
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms


matplotlib.use("Agg")


def _collate_fn(items):
    return {k: np.stack([np.asarray(x[k]) for x in items], axis=0) for k in items[0]}


def _make_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=_collate_fn,
    )


def _concat_batches(batches: list[np.ndarray]) -> np.ndarray:
    if not batches:
        raise ValueError("No batch data collected.")
    return np.concatenate(batches, axis=0)


def _plot_per_dim_hist(raw: np.ndarray, norm: np.ndarray, title: str, out_path: Path):
    dims = raw.shape[1]
    cols = min(4, dims)
    rows = math.ceil(dims / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= dims:
            ax.axis("off")
            continue
        ax.hist(raw[:, i], bins=80, alpha=0.45, density=True, label="raw")
        ax.hist(norm[:, i], bins=80, alpha=0.45, density=True, label="norm")
        ax.set_title(f"dim {i}")
        if i == 0:
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_box(values: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(10, values.shape[1] * 0.45), 5))
    ax.boxplot([values[:, i] for i in range(values.shape[1])], showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("dimension")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_corr_heatmap(values: np.ndarray, title: str, out_path: Path):
    corr = np.corrcoef(values, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_transformed_datasets(cfg: training_config.TrainConfig):
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    base_dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    raw_dataset = data_loader.TransformedDataset(
        base_dataset,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            *data_cfg.model_transforms.inputs,
        ],
    )
    norm_dataset = data_loader.transform_dataset(base_dataset, data_cfg, skip_norm_stats=False)
    return data_cfg, raw_dataset, norm_dataset


def main():
    parser = argparse.ArgumentParser(description="Analyze A dataset normalized distributions and clusters.")
    parser.add_argument("--config-name", default="pi05_aloha_wbcd_lora")
    parser.add_argument(
        "--output-dir",
        default="/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/artifacts/analysis",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cluster-max-samples", type=int, default=20000)
    parser.add_argument("--silhouette-max-samples", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, only analyze first N samples.")
    parser.add_argument(
        "--features-npz",
        default="",
        help="Optional precomputed feature file. If set, skip dataset loading and reuse this file.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = None
    if args.features_npz:
        loaded = np.load(args.features_npz)
        raw_state = loaded["raw_state"].astype(np.float32)
        norm_state = loaded["norm_state"].astype(np.float32)
        raw_action_seq = loaded["raw_actions"].astype(np.float32)
        norm_action_seq = loaded["norm_actions"].astype(np.float32)
        sample_index = loaded["sample_index"].astype(np.int64)
        print(f"Loaded features from {args.features_npz}", flush=True)
    else:
        cfg = training_config.get_config(args.config_name)
        data_cfg, raw_dataset, norm_dataset = _build_transformed_datasets(cfg)

        raw_loader = _make_loader(raw_dataset, args.batch_size, args.num_workers)
        norm_loader = _make_loader(norm_dataset, args.batch_size, args.num_workers)

        raw_states, norm_states = [], []
        raw_actions, norm_actions = [], []
        sample_indices = []

        sample_count = 0
        for batch_idx, (raw_batch, norm_batch) in enumerate(zip(raw_loader, norm_loader, strict=True)):
            raw_states.append(raw_batch["state"])
            norm_states.append(norm_batch["state"])
            raw_actions.append(raw_batch["actions"])
            norm_actions.append(norm_batch["actions"])
            bsz = raw_batch["state"].shape[0]
            sample_indices.append(np.arange(sample_count, sample_count + bsz))
            sample_count += bsz
            if batch_idx % 50 == 0:
                print(f"Loaded batches={batch_idx + 1}, samples={sample_count}", flush=True)
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break

        raw_state = _concat_batches(raw_states).astype(np.float32)
        norm_state = _concat_batches(norm_states).astype(np.float32)
        raw_action_seq = _concat_batches(raw_actions).astype(np.float32)
        norm_action_seq = _concat_batches(norm_actions).astype(np.float32)
        sample_index = _concat_batches(sample_indices).astype(np.int64)

    # Collapse horizon to visualize per-action-dim distributions.
    raw_action = raw_action_seq.reshape(-1, raw_action_seq.shape[-1])
    norm_action = norm_action_seq.reshape(-1, norm_action_seq.shape[-1])

    np.savez_compressed(
        out_dir / "a_dataset_features.npz",
        raw_state=raw_state,
        norm_state=norm_state,
        raw_actions=raw_action_seq,
        norm_actions=norm_action_seq,
        sample_index=sample_index,
    )

    _plot_per_dim_hist(raw_state, norm_state, "State Distribution: raw vs norm", fig_dir / "state_hist_raw_vs_norm.png")
    _plot_per_dim_hist(raw_action, norm_action, "Action Distribution: raw vs norm", fig_dir / "action_hist_raw_vs_norm.png")

    _plot_box(raw_state, "State Raw Boxplot (per dim)", fig_dir / "state_box_raw.png")
    _plot_box(norm_state, "State Norm Boxplot (per dim)", fig_dir / "state_box_norm.png")
    _plot_box(raw_action, "Action Raw Boxplot (per dim)", fig_dir / "action_box_raw.png")
    _plot_box(norm_action, "Action Norm Boxplot (per dim)", fig_dir / "action_box_norm.png")

    _plot_corr_heatmap(norm_state, "Norm State Correlation", fig_dir / "state_corr_norm.png")
    _plot_corr_heatmap(norm_action, "Norm Action Correlation", fig_dir / "action_corr_norm.png")
    _plot_corr_heatmap(
        np.concatenate([norm_state, norm_action_seq.mean(axis=1)], axis=1),
        "Norm (State + MeanAction) Correlation",
        fig_dir / "state_action_corr_norm.png",
    )

    # Clustering on flattened action sequences.
    X = norm_action_seq.reshape(norm_action_seq.shape[0], -1).astype(np.float32)
    rng = np.random.default_rng(42)
    cluster_n = min(args.cluster_max_samples, X.shape[0])
    cluster_idx = np.sort(rng.choice(X.shape[0], size=cluster_n, replace=False))
    X_cluster = X[cluster_idx]

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cluster)

    sil_n = min(args.silhouette_max_samples, X_cluster.shape[0])
    sil_idx = np.sort(rng.choice(X_cluster.shape[0], size=sil_n, replace=False))
    X_sil = X_cluster[sil_idx]

    k_values = list(range(2, 11))
    silhouette_scores = []
    labels_by_k = {}
    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
        labels = model.fit_predict(X_cluster)
        labels_by_k[k] = labels
        sil_model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
        sil_labels = sil_model.fit_predict(X_sil)
        silhouette_scores.append(float(silhouette_score(X_sil, sil_labels)))

    best_k = k_values[int(np.argmax(np.asarray(silhouette_scores)))]
    best_labels = labels_by_k[best_k]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, silhouette_scores, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_title("KMeans silhouette scan")
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_silhouette_scan.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, s=6, alpha=0.65, cmap="tab10")
    ax.set_title(f"PCA(2D) + KMeans labels (k={best_k})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_kmeans_scatter.png", dpi=180)
    plt.close(fig)

    cluster_counts = np.bincount(best_labels, minlength=best_k)
    cluster_ratio = (cluster_counts / cluster_counts.sum()).tolist()

    metrics = {
        "config_name": args.config_name,
        "repo_id": None if data_cfg is None else data_cfg.repo_id,
        "num_samples": int(norm_action_seq.shape[0]),
        "action_horizon": int(norm_action_seq.shape[1]),
        "action_dim": int(norm_action_seq.shape[2]),
        "state_dim": int(norm_state.shape[1]),
        "cluster_max_samples": int(cluster_n),
        "silhouette_max_samples": int(sil_n),
        "silhouette_scan": {str(k): v for k, v in zip(k_values, silhouette_scores, strict=True)},
        "best_k": int(best_k),
        "best_cluster_counts": cluster_counts.tolist(),
        "best_cluster_ratio": cluster_ratio,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    (out_dir / "cluster_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved features: {out_dir / 'a_dataset_features.npz'}")
    print(f"Saved figures: {fig_dir}")
    print(f"Saved metrics: {out_dir / 'cluster_metrics.json'}")


if __name__ == "__main__":
    main()
