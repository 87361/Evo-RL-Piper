#!/usr/bin/env python3
"""Open-loop deployment evaluation for PI05 policy.

Loads a trained checkpoint and a LeRobot dataset, feeds each frame's
observation (images + state) to the policy, and compares the predicted
action with the ground-truth action from the dataset.

Outputs:
  - Per-episode and overall MSE / MAE / cosine similarity
  - Detailed per-dimension error statistics (for diagnosing left-arm dim7-13)
  - Trajectory plots saved to an output directory
  - Console summary

No ROS2 or simulation required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = REPO_ROOT / "third_party" / "openpi" / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

import openpi.policies.policy_config as policy_config
import openpi.training.config as training_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop evaluation of PI05 policy.")
    p.add_argument(
        "--checkpoint-dir", type=Path,
        default=REPO_ROOT / "third_party/openpi/checkpoints/pi05_aloha_wbcd_lora/evorl_pi05_lora_A5_prompt_260318",
        help="Path to the training run directory (contains step subdirs like 5000/, 10000/, ...).",
    )
    p.add_argument(
        "--step", type=int, default=None,
        help="Checkpoint step to load (e.g. 18000). If None, uses the latest available step.",
    )
    p.add_argument("--config-name", default="pi05_aloha_wbcd_lora")
    p.add_argument("--repo-id", default="pipeline_ab/A", help="LeRobot dataset repo id.")
    p.add_argument("--episodes", type=str, default="0", help="Comma-separated episode indices to evaluate, or 'all'.")
    p.add_argument("--prompt", default="pick and place", help="Task prompt fed to policy.")
    p.add_argument("--device", default="cuda", help="PyTorch device for policy.")
    p.add_argument("--max-steps-per-episode", type=int, default=None, help="Max frames per episode (for quick tests).")
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "tmp" / "open_loop_eval",
        help="Directory to save plots and results.",
    )
    p.add_argument("--action-chunk-index", type=int, default=0, help="Which index in the action chunk to compare (0=first predicted step).")
    p.add_argument("--left-arm-dims", type=str, default="7:14", help="Slice notation for left arm dims (default 7:14, i.e. dim7-dim13).")
    return p.parse_args()


def resolve_checkpoint_step(run_dir: Path, requested_step: int | None) -> Path:
    """Find the checkpoint subdirectory for the given step."""
    step_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step directories found in {run_dir}")
    if requested_step is not None:
        target = run_dir / str(requested_step)
        if not target.exists():
            available = [d.name for d in step_dirs]
            raise FileNotFoundError(f"Step {requested_step} not found. Available: {available}")
        return target
    # Use latest
    return step_dirs[-1]


def parse_dim_slice(s: str) -> slice:
    parts = s.split(":")
    return slice(*[int(x) if x else None for x in parts])


def load_dataset(repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=repo_id, delta_timestamps=None)


def episode_span(dataset, episode_index: int) -> tuple[int, int]:
    epi = getattr(dataset, "episode_data_index", None)
    if isinstance(epi, dict) and "from" in epi and "to" in epi:
        start = int(epi["from"][episode_index])
        end = int(epi["to"][episode_index])
        return start, end
    ep = dataset.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    return start, end


def resolve_cam_key(camera_keys: list[str], preferred: list[str], tag: str) -> str:
    for want in preferred:
        if want in camera_keys:
            return want
    for key in camera_keys:
        if tag in key.lower():
            return key
    raise ValueError(f"Cannot resolve {tag} camera key from {camera_keys}")


def to_chw_uint8(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.01:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.shape[0] != 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def evaluate_episode(
    policy,
    dataset,
    episode_index: int,
    cam_keys: dict[str, str],
    prompt: str,
    action_chunk_idx: int,
    max_steps: int | None,
) -> dict:
    """Run open-loop evaluation on a single episode. Returns dict of arrays."""
    start, end = episode_span(dataset, episode_index)
    indices = list(range(start, end))
    if max_steps is not None:
        indices = indices[:max_steps]

    gt_actions_list = []
    pred_actions_list = []
    states_list = []
    infer_times = []

    for step_i, idx in enumerate(indices):
        item = dataset[idx]

        # Extract state
        # The repack_transforms in config maps "agent_pos" -> "state"
        state = np.asarray(item.get("agent_pos", item.get("observation.state", item.get("state", None))))
        if state is None:
            raise ValueError(f"Cannot find state key in dataset item. Keys: {list(item.keys())}")
        state = state.astype(np.float32)

        # Extract images
        images = {
            "cam_high": to_chw_uint8(item[cam_keys["head"]]),
            "cam_left_wrist": to_chw_uint8(item[cam_keys["left"]]),
            "cam_right_wrist": to_chw_uint8(item[cam_keys["right"]]),
        }

        # Extract ground truth action
        gt_action = np.asarray(item.get("action", item.get("actions", None)))
        if gt_action is None:
            raise ValueError(f"Cannot find action key. Keys: {list(item.keys())}")
        gt_action = gt_action.astype(np.float32)

        # Build observation dict matching the repack_transforms output
        obs = {
            "images": images,
            "state": state,
            "prompt": prompt,
        }

        t0 = time.monotonic()
        out = policy.infer(obs)
        dt = time.monotonic() - t0
        infer_times.append(dt)

        pred_actions = np.asarray(out["actions"], dtype=np.float32)
        # pred_actions shape: [action_horizon, action_dim]
        pred_action = pred_actions[action_chunk_idx, :14]

        gt_actions_list.append(gt_action[:14])
        pred_actions_list.append(pred_action)
        states_list.append(state[:14])

        if step_i % 20 == 0:
            print(f"  episode {episode_index} step {step_i}/{len(indices)}: infer {dt*1000:.0f}ms")

    return {
        "gt_actions": np.array(gt_actions_list),
        "pred_actions": np.array(pred_actions_list),
        "states": np.array(states_list),
        "infer_times": np.array(infer_times),
    }


def compute_metrics(gt: np.ndarray, pred: np.ndarray, dim_names: list[str] | None = None) -> dict:
    """Compute per-dimension and overall metrics."""
    err = pred - gt
    mse_per_dim = np.mean(err ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(err), axis=0)

    # Cosine similarity per timestep
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
        "infer_time_mean_ms": None,  # filled later
    }
    if dim_names:
        result["dim_names"] = dim_names
    return result


def save_trajectory_plots(
    gt: np.ndarray, pred: np.ndarray, states: np.ndarray,
    episode_index: int, output_dir: Path,
    left_arm_slice: slice,
):
    """Save trajectory comparison plots."""
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

    # --- Full 14-dim plot ---
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
    fig.suptitle(f"Episode {episode_index} — All dims (GT vs Pred)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"ep{episode_index}_all_dims.png", dpi=120)
    plt.close(fig)

    # --- Left arm only (dim7-dim13) ---
    left_gt = gt[:, left_arm_slice]
    left_pred = pred[:, left_arm_slice]
    left_ndim = left_gt.shape[1]
    left_labels = dim_labels[left_arm_slice.start:left_arm_slice.stop] if left_arm_slice.stop <= len(dim_labels) else [f"dim{i}" for i in range(left_arm_slice.start, left_arm_slice.start + left_ndim)]

    # Override: user says left arm = dim7-dim13, let's label accordingly
    # But in the config, state is [left_6joints + left_gripper + right_6joints + right_gripper]
    # dim0-6 = left arm (including gripper at dim6)
    # dim7-13 = right arm (including gripper at dim13)
    # Wait - user says left arm is dim7-dim13. Let me just use the user's specification.

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
    fig2.suptitle(f"Episode {episode_index} — Left Arm (dim{left_arm_slice.start}-{left_arm_slice.stop - 1}) GT vs Pred", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"ep{episode_index}_left_arm.png", dpi=120)
    plt.close(fig2)

    # --- Error plot ---
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
    fig3.suptitle(f"Episode {episode_index} — Prediction Error (Pred - GT)", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(output_dir / f"ep{episode_index}_error.png", dpi=120)
    plt.close(fig3)

    print(f"  Saved plots to {output_dir}/ep{episode_index}_*.png")


def main():
    args = parse_args()

    # Resolve checkpoint
    ckpt_dir = resolve_checkpoint_step(args.checkpoint_dir, args.step)
    step_num = int(ckpt_dir.name)
    print(f"=== Open-loop Eval ===")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Step: {step_num}")
    print(f"Config: {args.config_name}")
    print(f"Dataset: {args.repo_id}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print(f"Left arm dims: {args.left_arm_dims}")
    print()

    # Load policy
    print("Loading policy...")
    cfg = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        cfg,
        ckpt_dir,
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )
    print("Policy loaded.\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.repo_id)
    camera_keys = list(dataset.meta.camera_keys)
    cam_keys = {
        "head": resolve_cam_key(camera_keys, ["observation.images.head_cam", "head_cam", "cam_high"], "head"),
        "left": resolve_cam_key(camera_keys, ["observation.images.left_wrist_cam", "left_wrist_cam"], "left"),
        "right": resolve_cam_key(camera_keys, ["observation.images.right_wrist_cam", "right_wrist_cam"], "right"),
    }
    print(f"Camera keys: {cam_keys}")
    n_episodes = len(dataset.meta.episodes)
    print(f"Dataset has {n_episodes} episodes, {len(dataset)} total frames.\n")

    # Resolve episodes
    if args.episodes == "all":
        episode_indices = list(range(n_episodes))
    else:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",")]

    left_slice = parse_dim_slice(args.left_arm_dims)

    output_dir = args.output_dir / f"step_{step_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_gt = []
    all_pred = []

    for ep_idx in episode_indices:
        print(f"\n--- Episode {ep_idx} ---")
        ep_result = evaluate_episode(
            policy, dataset, ep_idx, cam_keys, args.prompt,
            args.action_chunk_index, args.max_steps_per_episode,
        )

        gt = ep_result["gt_actions"]
        pred = ep_result["pred_actions"]
        states = ep_result["states"]

        metrics = compute_metrics(gt, pred)
        metrics["infer_time_mean_ms"] = float(np.mean(ep_result["infer_times"]) * 1000)
        metrics["n_steps"] = int(gt.shape[0])

        # Left arm specific metrics
        left_gt = gt[:, left_slice]
        left_pred = pred[:, left_slice]
        left_metrics = compute_metrics(left_gt, left_pred)
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

        # Save plots
        save_trajectory_plots(gt, pred, states, ep_idx, output_dir, left_slice)

    # Aggregate metrics
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

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
