#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _gemma_probe_utils import (
    choose_episode_indices,
    generate_answer,
    load_dataset,
    load_probe_context,
    parse_xy,
    resolve_head_camera_key,
    sample_last_frame,
    write_json,
)


Q1 = "描述这件T恤的物理状态，中间的褶皱在哪里？"
Q2 = "如果我要从中间拨开它，我的夹爪应该移动到图像坐标的哪个位置？请输出 x=..., y=..."
SEMANTIC_KEYWORDS = ("褶皱", "中间", "中央", "fold", "wrinkle", "middle", "center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma2B probing A: VLM semantic prompting on A dataset.")
    parser.add_argument("--config-name", default="pi05_aloha_wbcd_lora")
    parser.add_argument("--checkpoint-dir", required=True, help="OpenPI checkpoint directory.")
    parser.add_argument("--repo-id", default=None, help="Override dataset repo id (e.g. pipeline_ab/A).")
    parser.add_argument("--processor-id", default="google/paligemma2-3b-pt-224")
    parser.add_argument(
        "--output-dir",
        default="/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/artifacts/gemma2b_probe/A_vlm",
    )
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--question-1", default=Q1)
    parser.add_argument("--question-2", default=Q2)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _semantic_hit(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in SEMANTIC_KEYWORDS)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("No episodes selected. Check --max-episodes/--episode-stride.")

    rows: list[dict] = []
    semantic_hit = 0
    parse_ok = 0
    in_bounds = 0

    for ep_idx in episode_indices:
        image, frame_idx = sample_last_frame(dataset, ep_idx, cam_key)
        image_w, image_h = image.size

        a1 = generate_answer(
            ctx.model,
            ctx.processor,
            image=image,
            prompt=args.question_1,
            device=ctx.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if _semantic_hit(a1):
            semantic_hit += 1

        a2 = generate_answer(
            ctx.model,
            ctx.processor,
            image=image,
            prompt=args.question_2,
            device=ctx.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        xy, xy_in_bounds = parse_xy(a2, image_w, image_h)
        ok = xy is not None
        if ok:
            parse_ok += 1
        if xy_in_bounds:
            in_bounds += 1

        row = {
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "camera_key": cam_key,
            "question_1": args.question_1,
            "answer_1": a1,
            "semantic_hit": _semantic_hit(a1),
            "question_2": args.question_2,
            "answer_2": a2,
            "parsed_xy": None if xy is None else [float(xy[0]), float(xy[1])],
            "parse_ok": ok,
            "xy_in_bounds": bool(xy_in_bounds),
            "image_w": image_w,
            "image_h": image_h,
        }
        rows.append(row)
        print(
            f"[A] episode={ep_idx:04d} frame={frame_idx} semantic_hit={row['semantic_hit']} "
            f"parse_ok={ok} in_bounds={xy_in_bounds}",
            flush=True,
        )

    jsonl_path = out_dir / "vlm_prompt_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "vlm_prompt_results.csv"
    fields = [
        "episode_index",
        "frame_index",
        "camera_key",
        "question_1",
        "answer_1",
        "semantic_hit",
        "question_2",
        "answer_2",
        "parsed_xy",
        "parse_ok",
        "xy_in_bounds",
        "image_w",
        "image_h",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n = len(rows)
    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "repo_id": ctx.repo_id,
        "camera_key": cam_key,
        "num_samples": n,
        "semantic_hit_rate": semantic_hit / n,
        "coordinate_parse_rate": parse_ok / n,
        "coordinate_in_bounds_rate": in_bounds / n,
        "output_jsonl": str(jsonl_path.resolve()),
        "output_csv": str(csv_path.resolve()),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Saved A probing outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
