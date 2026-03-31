#!/usr/bin/env python3
"""Open-loop deployment evaluation for LeRobot policies (ACT, Diffusion, etc.).

Loads a trained LeRobot checkpoint and dataset, feeds each frame's
observation to the policy, extracts the predicted action sequence (chunk),
and compares a specified index of the predicted chunk with the ground-truth
action from the dataset.

Outputs:
  - Per-episode and overall MSE / MAE / cosine similarity
  - Detailed per-dimension error statistics
  - Trajectory plots saved to an output directory
  - Console summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import multiprocessing

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop evaluation for LeRobot.")
    p.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Path to the checkpoint directory (e.g. outputs/train/.../checkpoints/060000).",
    )
    p.add_argument("--repo-id", required=True, help="LeRobot dataset repo id or directory name.")
    p.add_argument("--dataset-root", required=True, type=Path, help="Path to dataset root where repo is stored.")
    p.add_argument("--episodes", type=str, default="0", help="Comma-separated episode indices, or 'all'.")
    p.add_argument("--device", default="cuda", help="PyTorch device.")
    p.add_argument("--max-steps-per-episode", type=int, default=None, help="Max frames per episode.")
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "tmp" / "open_loop_eval_lerobot",
        help="Directory to save plots and results.",
    )
    p.add_argument("--action-chunk-index", type=int, default=0, help="Which index in the predicted action chunk to compare.")
    p.add_argument("--left-arm-dims", type=str, default="7:14", help="Slice notation for left arm dims (default 7:14, i.e. dim7-dim13).")
    return p.parse_args()

def parse_dim_slice(s: str) -> slice:
    parts = s.split(":")
    return slice(*[int(x) if x else None for x in parts])

def compute_metrics(gt: np.ndarray, pred: np.ndarray, dim_names: list[str] | None = None) -> dict:
    err = pred - gt
    mse_per_dim = np.mean(err ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(err), axis=0)

    dots = np.sum(gt * pred, axis=1)
    norms_gt = np.linalg.norm(gt, axis=1) + 1e-8
    norms_pred = np.linalg.norm(pred, axis=1) + 1e-8
    cos_sim = dots / (norms_gt * norms_pred)

    result = {
        "mse_overall": float(np.mean(mse_per_dim)),
        "mae_overall": float(np.mean(mae_per_dim)),
        "cos_sim_mean": float(np.mean(cos_sim)),
        "cos_sim_std": float(np.std(cos_sim)),
        "mse_per_dim": mse_per_dim.tolist(),
        "mae_per_dim": mae_per_dim.tolist(),
        "infer_time_mean_ms": None,
    }
    if dim_names:
        result["dim_names"] = dim_names
    return result

def save_trajectory_plots(
    gt: np.ndarray, pred: np.ndarray,
    episode_index: int, output_dir: Path,
    left_arm_slice: slice,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    T = gt.shape[0]
    ndim = gt.shape[1]

    dim_labels = [f"left_j{i}" for i in range(1, 7)] + ["left_grip"] + \
                 [f"right_j{i}" for i in range(1, 7)] + ["right_grip"]
    if ndim < len(dim_labels):
        dim_labels = dim_labels[:ndim]

    # Full plot
    fig, axes = plt.subplots(ndim, 1, figsize=(14, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for d in range(ndim):
        ax = axes[d]
        ax.plot(gt[:, d], label="GT", alpha=0.8, linewidth=1.5)
        ax.plot(pred[:, d], label="Pred", alpha=0.8, linewidth=1.5, linestyle="--")
        ax.set_ylabel(dim_labels[d] if d < len(dim_labels) else f"dim{d}", fontsize=8)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Episode {episode_index} GT vs Pred", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"ep{episode_index}_all_dims.png", dpi=120)
    plt.close(fig)

    # Left arm plot
    left_gt = gt[:, left_arm_slice]
    left_pred = pred[:, left_arm_slice]
    left_ndim = left_gt.shape[1]
    left_labels = dim_labels[left_arm_slice.start:left_arm_slice.stop] if left_arm_slice.stop <= len(dim_labels) else [f"dim{i}" for i in range(left_arm_slice.start, left_arm_slice.start + left_ndim)]

    fig2, axes2 = plt.subplots(left_ndim, 1, figsize=(14, 2 * left_ndim), sharex=True)
    if left_ndim == 1:
        axes2 = [axes2]
    for d in range(left_ndim):
        ax = axes2[d]
        ax.plot(left_gt[:, d], label="GT", alpha=0.8, linewidth=1.5)
        ax.plot(left_pred[:, d], label="Pred", alpha=0.8, linewidth=1.5, linestyle="--")
        label = left_labels[d] if d < len(left_labels) else f"dim{left_arm_slice.start + d}"
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8)
    axes2[-1].set_xlabel("Step")
    fig2.suptitle(f"Episode {episode_index} Left Arm GT vs Pred", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"ep{episode_index}_left_arm.png", dpi=120)
    plt.close(fig2)

    # Error plot
    err = pred - gt
    fig3, axes3 = plt.subplots(ndim, 1, figsize=(14, 2 * ndim), sharex=True)
    if ndim == 1:
        axes3 = [axes3]
    for d in range(ndim):
        ax = axes3[d]
        ax.plot(err[:, d], color="red", alpha=0.7, linewidth=1)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_ylabel(dim_labels[d] if d < len(dim_labels) else f"dim{d}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes3[-1].set_xlabel("Step")
    fig3.suptitle(f"Episode {episode_index} Prediction Error (Pred - GT)", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(output_dir / f"ep{episode_index}_error.png", dpi=120)
    plt.close(fig3)

def evaluate_episode(
    policy,
    preprocessor,
    dataset,
    episode_index: int,
    action_chunk_idx: int,
    max_steps: int | None,
    device: str
) -> dict:
    ep = dataset.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    indices = list(range(start, end))
    if max_steps is not None:
        indices = indices[:max_steps]

    gt_actions_list = []
    pred_actions_list = []
    infer_times = []

    for step_i, idx in enumerate(indices):
        item = dataset[idx]

        # Extract gt action (1D unnormalized tensor typically).
        # Depending on policy action delta_timestamps, item['action'] might be a scalar step or sequence.
        gt_a = item["action"]
        if isinstance(gt_a, torch.Tensor):
            gt_a = gt_a.numpy()
        # if the dataset provides a sequence of actions due to delta_timestamps, take the first one as GT
        if gt_a.ndim == 2:
            gt_a = gt_a[0]

        # Prepare batch for policy
        batch = {}
        # Only take input features (observations) expected by preprocessor/policy
        for k, v in item.items():
            if k in policy.config.input_features:
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
        
        t0 = time.monotonic()
        with torch.inference_mode():
            obs = preprocessor(batch)
            
            # Use predict_action_chunk if available (ACT/Diffusion usually have this)
            if hasattr(policy, "predict_action_chunk"):
                action_pred = policy.predict_action_chunk(obs)  # usually returns [B, chunk_size, act_dim]
            else:
                # Fallback to select_action
                action_pred = policy.select_action(obs)

        dt = time.monotonic() - t0
        infer_times.append(dt)

        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.cpu().numpy()

        if action_pred.ndim == 3:  # [B, chunk_size, act_dim]
            pred_a = action_pred[0, action_chunk_idx]
        elif action_pred.ndim == 2: # [B, act_dim]
            pred_a = action_pred[0]
        else:
            pred_a = action_pred

        # Both gt_a and pred_a are ~ [14]
        gt_actions_list.append(gt_a[:14])
        pred_actions_list.append(pred_a[:14])

        if step_i % 20 == 0:
            print(f"  episode {episode_index} step {step_i}/{len(indices)}: infer {dt*1000:.0f}ms")

    return {
        "gt_actions": np.array(gt_actions_list),
        "pred_actions": np.array(pred_actions_list),
        "infer_times": np.array(infer_times),
    }

def main():
    # Force single thread to avoid multiprocessing lockups from matplotlib
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    # Append 'pretrained_model' if passing directory directly
    ckpt_dir = args.checkpoint_dir
    if (ckpt_dir / "pretrained_model").exists():
        ckpt_dir = ckpt_dir / "pretrained_model"

    print(f"=== Open-loop Eval ===")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Device: {args.device}")
    print(f"Left arm dims: {args.left_arm_dims}")
    print()

    print("Loading policy...")
    cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(ckpt_dir))
    policy.eval()
    policy.to(args.device)
    print(f"Policy loaded (type: {cfg.type}).\n")

    print("Loading processors...")
    # NOTE: make_pre_post_processors will load from config/pretrained_path by default.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": args.device}}
    )

    print("Loading dataset...")
    # Read datset delta_timestamps if configured
    dt = getattr(policy.config, "dataset_kwargs", {}).get("delta_timestamps", None)
    if hasattr(policy.config, "training"):
        dt = dt or getattr(policy.config.training, "delta_timestamps", None)
    
    # Needs to import from lerobot.common.datasets.lerobot_dataset or lerobot.datasets.lerobot_dataset
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.dataset_root), delta_timestamps=dt)
    n_episodes = len(dataset.meta.episodes)
    print(f"Dataset has {n_episodes} episodes, {len(dataset)} total frames.\n")

    if args.episodes == "all":
        episode_indices = list(range(n_episodes))
    else:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",")]

    left_slice = parse_dim_slice(args.left_arm_dims)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assuming checkpoint_dir has format `checkpoints/060000`, fetch step logic
    step_num = ckpt_dir.parent.name if ckpt_dir.name == "pretrained_model" else ckpt_dir.name
    output_dir = args.output_dir / f"step_{step_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_gt = []
    all_pred = []

    for ep_idx in episode_indices:
        print(f"\n--- Episode {ep_idx} ---")
        ep_result = evaluate_episode(
            policy, preprocessor, dataset, ep_idx,
            args.action_chunk_index, args.max_steps_per_episode, args.device
        )

        gt = ep_result["gt_actions"]
        pred = ep_result["pred_actions"]

        metrics = compute_metrics(gt, pred)
        metrics["infer_time_mean_ms"] = float(np.mean(ep_result["infer_times"]) * 1000)
        metrics["n_steps"] = int(gt.shape[0])

        left_metrics = compute_metrics(gt[:, left_slice], pred[:, left_slice])
        metrics["left_arm_mse"] = left_metrics["mse_overall"]
        metrics["left_arm_mae"] = left_metrics["mae_overall"]
        metrics["left_arm_cos_sim"] = left_metrics["cos_sim_mean"]
        metrics["left_arm_mse_per_dim"] = left_metrics["mse_per_dim"]

        all_results[f"episode_{ep_idx}"] = metrics
        all_gt.append(gt)
        all_pred.append(pred)

        print(f"  Steps: {metrics['n_steps']}")
        print(f"  Overall MSE: {metrics['mse_overall']:.6f}  MAE: {metrics['mae_overall']:.6f}")
        print(f"  Left-arm MSE: {metrics['left_arm_mse']:.6f}  MAE: {metrics['left_arm_mae']:.6f}")
        print(f"  Cosine sim: {metrics['cos_sim_mean']:.4f} ± {metrics['cos_sim_std']:.4f}")
        print(f"  Infer time: {metrics['infer_time_mean_ms']:.1f} ms/step")

        save_trajectory_plots(gt, pred, ep_idx, output_dir, left_slice)

    if len(all_gt) > 0:
        agg_gt = np.concatenate(all_gt, axis=0)
        agg_pred = np.concatenate(all_pred, axis=0)
        agg_metrics = compute_metrics(agg_gt, agg_pred)
        left_agg_metrics = compute_metrics(agg_gt[:, left_slice], agg_pred[:, left_slice])
        agg_metrics["left_arm_mse"] = left_agg_metrics["mse_overall"]
        agg_metrics["left_arm_mae"] = left_agg_metrics["mae_overall"]
        agg_metrics["left_arm_cos_sim"] = left_agg_metrics["cos_sim_mean"]
        agg_metrics["n_total_steps"] = int(agg_gt.shape[0])
        agg_metrics["n_episodes"] = len(episode_indices)
        all_results["aggregate"] = agg_metrics

        print(f"\n{'='*60}")
        print(f"AGGREGATE ({agg_metrics['n_episodes']} episodes, {agg_metrics['n_total_steps']} steps)")
        print(f"  Overall MSE: {agg_metrics['mse_overall']:.6f}  MAE: {agg_metrics['mae_overall']:.6f}")
        print(f"  Left-arm MSE: {agg_metrics['left_arm_mse']:.6f}  MAE: {agg_metrics['left_arm_mae']:.6f}")
        print(f"  Cosine sim: {agg_metrics['cos_sim_mean']:.4f} ± {float(agg_metrics.get('cos_sim_std', 0)):.4f}")
        print(f"  Per-dim MSE: {[f'{v:.6f}' for v in agg_metrics['mse_per_dim']]}")
        print(f"{'='*60}\n")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
