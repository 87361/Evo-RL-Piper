#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
import torch

from _gemma_probe_utils import (
    build_inputs,
    choose_episode_indices,
    evenly_sample_indices,
    load_dataset,
    load_probe_context,
    resolve_head_camera_key,
    resolve_state_key,
    split_tail_by_gripper_close,
    pil_from_frame,
    write_json,
)

matplotlib.use("Agg")

PHASE_PRE = "pre_contact"
PHASE_POST = "post_contact"
PROBE_PROMPT = "请理解当前T恤状态并判断是否已经发生接触或拨开。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma2B probing C: hidden-state phase clustering.")
    parser.add_argument("--config-name", default="pi05_aloha_wbcd_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", default=None, help="Override dataset repo id (e.g. pipeline_ab/A).")
    parser.add_argument("--processor-id", default="google/paligemma2-3b-pt-224")
    parser.add_argument(
        "--output-dir",
        default="/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/artifacts/gemma2b_probe/C_phase",
    )
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--frames-per-phase", type=int, default=2)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tsne-perplexity", type=float, default=20.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _pool_feature(last_hidden: np.ndarray, input_ids: list[int], image_token_id: int, attention_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visual_pos = [i for i, tid in enumerate(input_ids) if tid == image_token_id and attention_mask[i]]
    text_pos = [i for i, m in enumerate(attention_mask.tolist()) if m and i not in visual_pos]
    if not text_pos:
        text_pos = [i for i, m in enumerate(attention_mask.tolist()) if m]
    if not visual_pos:
        visual_pos = text_pos
    text_pool = last_hidden[text_pos].mean(axis=0)
    visual_pool = last_hidden[visual_pos].mean(axis=0)
    return text_pool, visual_pool


def _safe_metrics(X: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if X.shape[0] < 4 or len(np.unique(y)) < 2:
        return None, None
    sil = float(silhouette_score(X, y))
    db = float(davies_bouldin_score(X, y))
    return sil, db


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_probe_context(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        processor_id=args.processor_id,
        repo_id_override=args.repo_id,
        device=args.device,
    )
    dataset = load_dataset(ctx.repo_id, video_backend=args.video_backend)
    cam_key = resolve_head_camera_key(dataset)
    state_key = resolve_state_key(dataset)
    episode_indices = choose_episode_indices(
        dataset.num_episodes,
        max_episodes=args.max_episodes,
        stride=args.episode_stride,
    )
    if not episode_indices:
        raise ValueError("No episodes selected.")

    image_token_id = int(ctx.model.config.image_token_index)

    phase_labels: list[str] = []
    y: list[int] = []
    features_joint: list[np.ndarray] = []
    features_text: list[np.ndarray] = []
    features_visual: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for ep_idx in episode_indices:
        pre_indices, post_indices, t_close = split_tail_by_gripper_close(dataset, ep_idx, state_key)
        phase_to_indices = {
            PHASE_PRE: evenly_sample_indices(pre_indices, args.frames_per_phase),
            PHASE_POST: evenly_sample_indices(post_indices, args.frames_per_phase),
        }
        for phase_name, indices in phase_to_indices.items():
            for frame_idx in indices:
                item = dataset[frame_idx]
                pil_image = pil_from_frame(item[cam_key])

                inputs = build_inputs(ctx.processor, pil_image, PROBE_PROMPT, ctx.device)
                with torch.no_grad():
                    out = ctx.model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                last_hidden = out.hidden_states[-1][0].detach().float().cpu().numpy()
                input_ids = inputs["input_ids"][0].detach().cpu().tolist()
                attn_mask = inputs["attention_mask"][0].detach().cpu().numpy().astype(bool)
                text_pool, visual_pool = _pool_feature(last_hidden, input_ids, image_token_id, attn_mask)
                joint = np.concatenate([text_pool, visual_pool], axis=0)

                features_text.append(text_pool.astype(np.float32))
                features_visual.append(visual_pool.astype(np.float32))
                features_joint.append(joint.astype(np.float32))
                phase_labels.append(phase_name)
                y.append(0 if phase_name == PHASE_PRE else 1)
                meta_rows.append(
                    {
                        "episode_index": int(ep_idx),
                        "frame_index": int(frame_idx),
                        "phase": phase_name,
                        "close_index": int(t_close),
                    }
                )
        print(
            f"[C] episode={ep_idx:04d} close_idx={t_close} pre={len(pre_indices)} post={len(post_indices)}",
            flush=True,
        )

    if not features_joint:
        raise RuntimeError("No feature extracted. Check camera/state keys and dataset content.")

    X = np.stack(features_joint, axis=0)
    X_text = np.stack(features_text, axis=0)
    X_visual = np.stack(features_visual, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)

    sil_joint, db_joint = _safe_metrics(X, y_arr)
    sil_text, db_text = _safe_metrics(X_text, y_arr)
    sil_visual, db_visual = _safe_metrics(X_visual, y_arr)

    rng = np.random.default_rng(args.random_seed)
    y_shuffle = rng.permutation(y_arr)
    sil_shuffle, db_shuffle = _safe_metrics(X, y_shuffle)

    n_samples = int(X.shape[0])
    # sklearn requires perplexity < n_samples.
    perplexity = min(args.tsne_perplexity, max(2.0, float(n_samples - 1) / 3.0))
    if perplexity >= n_samples:
        perplexity = max(1.0, float(n_samples) - 1.0)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.random_seed,
    )
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    pre_mask = y_arr == 0
    post_mask = y_arr == 1
    ax.scatter(emb[pre_mask, 0], emb[pre_mask, 1], s=14, alpha=0.75, label=PHASE_PRE)
    ax.scatter(emb[post_mask, 0], emb[post_mask, 1], s=14, alpha=0.75, label=PHASE_POST)
    ax.set_title("t-SNE of Gemma2B hidden states (pre/post contact)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tsne_phase_scatter.png", dpi=180)
    plt.close(fig)

    umap_path = None
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=args.random_seed)
        emb_u = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb_u[pre_mask, 0], emb_u[pre_mask, 1], s=14, alpha=0.75, label=PHASE_PRE)
        ax.scatter(emb_u[post_mask, 0], emb_u[post_mask, 1], s=14, alpha=0.75, label=PHASE_POST)
        ax.set_title("UMAP of Gemma2B hidden states (pre/post contact)")
        ax.legend()
        fig.tight_layout()
        umap_path = fig_dir / "umap_phase_scatter.png"
        fig.savefig(umap_path, dpi=180)
        plt.close(fig)
    except Exception:
        umap_path = None

    np.savez_compressed(
        out_dir / "phase_features.npz",
        X_joint=X.astype(np.float32),
        X_text=X_text.astype(np.float32),
        X_visual=X_visual.astype(np.float32),
        y=y_arr,
        phase_labels=np.asarray(phase_labels),
    )

    metrics = {
        "config_name": args.config_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "repo_id": ctx.repo_id,
        "camera_key": cam_key,
        "state_key": state_key,
        "num_samples": int(X.shape[0]),
        "num_pre": int(pre_mask.sum()),
        "num_post": int(post_mask.sum()),
        "silhouette_joint": sil_joint,
        "davies_bouldin_joint": db_joint,
        "silhouette_text": sil_text,
        "davies_bouldin_text": db_text,
        "silhouette_visual": sil_visual,
        "davies_bouldin_visual": db_visual,
        "silhouette_joint_shuffled_label": sil_shuffle,
        "davies_bouldin_joint_shuffled_label": db_shuffle,
        "tsne_perplexity": perplexity,
        "tsne_scatter": str((fig_dir / "tsne_phase_scatter.png").resolve()),
        "umap_scatter": None if umap_path is None else str(umap_path.resolve()),
    }
    write_json(out_dir / "phase_cluster_metrics.json", metrics)
    write_json(out_dir / "phase_sample_index.json", {"samples": meta_rows})
    print(f"Saved C probing outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
