#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch

from _gemma_probe_utils import (
    build_inputs,
    choose_episode_indices,
    find_subsequence_positions,
    load_dataset,
    load_probe_context,
    normalized_entropy,
    resolve_head_camera_key,
    sample_last_frame,
    write_json,
)

matplotlib.use("Agg")

PROMPT = "请观察这件T恤，重点关注中间与可拨开路径。"
KEYWORDS = ("T恤", "中间", "拨开", "shirt", "middle", "open")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma2B probing B: keyword attention visualization.")
    parser.add_argument("--config-name", default="pi05_aloha_wbcd_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", default=None, help="Override dataset repo id (e.g. pipeline_ab/A).")
    parser.add_argument("--processor-id", default="google/paligemma2-3b-pt-224")
    parser.add_argument(
        "--output-dir",
        default="/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/artifacts/gemma2b_probe/B_attention",
    )
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _gridify(values: np.ndarray) -> np.ndarray:
    n = len(values)
    side = int(math.ceil(math.sqrt(n)))
    out = np.zeros((side * side,), dtype=np.float32)
    out[:n] = values.astype(np.float32)
    return out.reshape(side, side)


def _overlay_heatmap(image: Image.Image, heat: np.ndarray, out_path: Path) -> None:
    img = np.asarray(image).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), resample=Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0
    color = cm.get_cmap("jet")(heat_arr)[..., :3]
    alpha = 0.45
    mixed = np.clip((1 - alpha) * img + alpha * color, 0.0, 1.0)
    out = (mixed * 255.0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)


def _keyword_positions(processor, input_ids: list[int]) -> list[int]:
    tok = processor.tokenizer
    hits: set[int] = set()
    for word in KEYWORDS:
        ids = tok.encode(word, add_special_tokens=False)
        if not ids:
            continue
        for pos in find_subsequence_positions(input_ids, ids):
            hits.add(pos)
    return sorted(hits)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_probe_context(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        processor_id=args.processor_id,
        repo_id_override=args.repo_id,
        device=args.device,
    )
    dataset = load_dataset(ctx.repo_id, video_backend=args.video_backend)
    cam_key = resolve_head_camera_key(dataset)
    episode_indices = choose_episode_indices(
        dataset.num_episodes,
        max_episodes=args.max_episodes,
        stride=args.episode_stride,
    )
    if not episode_indices:
        raise ValueError("No episodes selected.")

    image_token_id = int(ctx.model.config.image_token_index)
    rows: list[dict] = []
    all_entropy = []
    all_top10 = []
    all_peak = []

    for ep_idx in episode_indices:
        image, frame_idx = sample_last_frame(dataset, ep_idx, cam_key)
        inputs = build_inputs(ctx.processor, image, PROMPT, ctx.device)
        with torch.no_grad():
            out = ctx.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )

        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        visual_pos = [i for i, tid in enumerate(input_ids) if tid == image_token_id]
        key_pos = _keyword_positions(ctx.processor, input_ids)
        if not key_pos:
            # fallback: use all valid non-image prompt tokens
            mask = inputs["attention_mask"][0].detach().cpu().numpy().astype(bool)
            key_pos = [i for i, m in enumerate(mask.tolist()) if m and i not in visual_pos]

        if not visual_pos or not key_pos:
            print(f"[B] skip episode={ep_idx} no visual/key tokens", flush=True)
            continue

        attn = out.attentions[-1][0].detach().float().cpu().numpy()  # [heads, q, k]
        score = attn[:, key_pos, :][:, :, visual_pos].mean(axis=(0, 1))  # [n_visual]
        score = np.clip(score, 0.0, None)
        if score.sum() <= 1e-8:
            score = np.ones_like(score)
        score = score / score.sum()

        entropy = normalized_entropy(score)
        peak = float(score.max())
        topk = float(np.sort(score)[-max(1, len(score) // 10) :].sum())
        all_entropy.append(entropy)
        all_peak.append(peak)
        all_top10.append(topk)

        heat_grid = _gridify(score)
        heat_grid = (heat_grid - heat_grid.min()) / (heat_grid.max() - heat_grid.min() + 1e-8)
        img_out = heat_dir / f"episode_{ep_idx:06d}_frame_{frame_idx:06d}.png"
        _overlay_heatmap(image, heat_grid, img_out)

        row = {
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "camera_key": cam_key,
            "num_visual_tokens": len(visual_pos),
            "num_keyword_positions": len(key_pos),
            "entropy_norm": entropy,
            "peak_weight": peak,
            "top10pct_mass": topk,
            "heatmap_path": str(img_out.resolve()),
        }
        rows.append(row)
        print(
            f"[B] episode={ep_idx:04d} entropy={entropy:.4f} peak={peak:.4f} top10={topk:.4f}",
            flush=True,
        )

    with (out_dir / "attention_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "repo_id": ctx.repo_id,
        "camera_key": cam_key,
        "num_samples": len(rows),
        "mean_entropy_norm": float(np.mean(all_entropy)) if rows else None,
        "mean_peak_weight": float(np.mean(all_peak)) if rows else None,
        "mean_top10pct_mass": float(np.mean(all_top10)) if rows else None,
    }
    write_json(out_dir / "attention_metrics.json", summary)
    print(f"Saved B probing outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
