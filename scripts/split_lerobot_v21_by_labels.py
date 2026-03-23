#!/usr/bin/env python
"""Split a LeRobot v2.1 dataset into per-label subsets by CSV labels.

Generalized N-category split: creates one output directory per label.
Backward compatible with A/B usage via --task-map.

Can also be imported and called from other code (split_dataset function).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training_repo.common.io import read_json, read_jsonl, write_json, write_jsonl


def _episode_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", value.strip())
    return int(match.group(1)) if match else None


def _load_label_map(csv_path: Path) -> dict[int, str]:
    by_num: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            episode_id = str(row.get("episode_id", "")).strip()
            label = str(row.get("label", "")).strip()
            if not episode_id or not label:
                continue
            num = _episode_number(episode_id)
            if num is not None:
                by_num[num] = label
    return by_num


def _video_keys(info: dict) -> list[str]:
    keys: list[str] = []
    for key, feature in info.get("features", {}).items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            keys.append(key)
    return sorted(keys)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _total_chunks(total_episodes: int, chunks_size: int) -> int:
    if total_episodes <= 0:
        return 0
    return (total_episodes - 1) // chunks_size + 1


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _rewrite_episode_parquet(
    src_path: Path,
    dst_path: Path,
    *,
    new_episode_index: int,
    new_task_index: int,
    new_global_index_start: int,
) -> int:
    table = pq.read_table(src_path)
    num_rows = table.num_rows
    if num_rows <= 0:
        raise ValueError(f"Empty episode parquet: {src_path}")

    ep_col = table.schema.get_field_index("episode_index")
    idx_col = table.schema.get_field_index("index")
    task_col = table.schema.get_field_index("task_index")
    if ep_col < 0 or idx_col < 0 or task_col < 0:
        raise ValueError(f"Missing required columns in {src_path}")

    table = table.set_column(ep_col, "episode_index",
                             pa.array([new_episode_index] * num_rows, type=pa.int64()))
    table = table.set_column(idx_col, "index",
                             pa.array(range(new_global_index_start, new_global_index_start + num_rows), type=pa.int64()))
    table = table.set_column(task_col, "task_index",
                             pa.array([new_task_index] * num_rows, type=pa.int64()))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)
    return num_rows


def _update_episode_stats_row(
    stats_row: dict,
    *,
    new_episode_index: int,
    new_global_index_start: int,
    episode_len: int,
) -> dict:
    row = dict(stats_row)
    row["episode_index"] = new_episode_index
    stats = row.get("stats", {})

    if "episode_index" in stats:
        stats["episode_index"] = {
            "min": [new_episode_index], "max": [new_episode_index],
            "mean": [float(new_episode_index)], "std": [0.0], "count": [episode_len],
        }
    if "task_index" in stats:
        stats["task_index"] = {
            "min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [episode_len],
        }
    if "index" in stats and episode_len > 0:
        start = new_global_index_start
        end = new_global_index_start + episode_len - 1
        mean = start + (episode_len - 1) / 2.0
        std = math.sqrt((episode_len ** 2 - 1) / 12.0) if episode_len > 1 else 0.0
        stats["index"] = {
            "min": [start], "max": [end], "mean": [float(mean)], "std": [float(std)],
            "count": [episode_len],
        }
    row["stats"] = stats
    return row


def _build_subset(
    *,
    subset_name: str,
    subset_indices: list[int],
    subset_task_name: str,
    src_root: Path,
    src_info: dict,
    episodes_by_idx: dict[int, dict],
    episodes_stats_by_idx: dict[int, dict],
    video_keys: list[str],
    dst_root: Path,
    require_all_videos: bool,
    dry_run: bool,
) -> dict:
    chunks_size = int(src_info.get("chunks_size", 1000))
    data_path_tmpl = str(src_info["data_path"])
    video_path_tmpl = str(src_info.get("video_path", ""))

    kept_indices: list[int] = []
    skipped_missing_video: list[int] = []
    skipped_missing_data: list[int] = []

    for ep_idx in subset_indices:
        episode_chunk = _episode_chunk(ep_idx, chunks_size)
        data_src = src_root / data_path_tmpl.format(episode_chunk=episode_chunk, episode_index=ep_idx)
        if not data_src.exists():
            skipped_missing_data.append(ep_idx)
            continue

        missing_video = False
        if video_path_tmpl:
            for key in video_keys:
                video_src = src_root / video_path_tmpl.format(
                    episode_chunk=episode_chunk, video_key=key, episode_index=ep_idx)
                if not video_src.exists():
                    missing_video = True
                    break

        if missing_video and require_all_videos:
            skipped_missing_video.append(ep_idx)
            continue
        kept_indices.append(ep_idx)

    total_frames_est = sum(int(episodes_by_idx[i]["length"]) for i in kept_indices)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

    if dry_run:
        return {
            "subset": subset_name,
            "kept_episodes": len(kept_indices),
            "skipped_missing_video": skipped_missing_video,
            "skipped_missing_data": skipped_missing_data,
            "total_frames_estimated": total_frames_est,
        }

    total_frames = 0
    subset_episodes = []
    subset_stats = []

    for old_ep_idx in kept_indices:
        new_ep_idx = old_to_new[old_ep_idx]
        old_chunk = _episode_chunk(old_ep_idx, chunks_size)
        new_chunk = _episode_chunk(new_ep_idx, chunks_size)

        data_src = src_root / data_path_tmpl.format(episode_chunk=old_chunk, episode_index=old_ep_idx)
        data_dst = dst_root / data_path_tmpl.format(episode_chunk=new_chunk, episode_index=new_ep_idx)
        episode_len = _rewrite_episode_parquet(
            data_src, data_dst,
            new_episode_index=new_ep_idx,
            new_task_index=0,
            new_global_index_start=total_frames,
        )
        total_frames += episode_len

        if video_path_tmpl:
            for key in video_keys:
                video_src = src_root / video_path_tmpl.format(
                    episode_chunk=old_chunk, video_key=key, episode_index=old_ep_idx)
                if video_src.exists():
                    video_dst = dst_root / video_path_tmpl.format(
                        episode_chunk=new_chunk, video_key=key, episode_index=new_ep_idx)
                    _copy_file(video_src, video_dst)

        item = dict(episodes_by_idx[old_ep_idx])
        item["episode_index"] = new_ep_idx
        item["tasks"] = [subset_task_name]
        item["length"] = episode_len
        subset_episodes.append(item)

    write_jsonl(dst_root / "meta" / "episodes.jsonl", subset_episodes)

    for old_ep_idx in kept_indices:
        if old_ep_idx not in episodes_stats_by_idx:
            continue
        new_ep_idx = old_to_new[old_ep_idx]
        episode_len = int(episodes_by_idx[old_ep_idx]["length"])
        global_start = sum(int(episodes_by_idx[i]["length"]) for i in kept_indices if old_to_new[i] < new_ep_idx)
        subset_stats.append(
            _update_episode_stats_row(
                episodes_stats_by_idx[old_ep_idx],
                new_episode_index=new_ep_idx,
                new_global_index_start=global_start,
                episode_len=episode_len,
            )
        )
    write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", subset_stats)
    write_jsonl(dst_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": subset_task_name}])

    info = dict(src_info)
    info["total_episodes"] = len(kept_indices)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    if "total_chunks" in info:
        info["total_chunks"] = _total_chunks(len(kept_indices), chunks_size)
    if "total_videos" in info:
        info["total_videos"] = len(kept_indices) * len(video_keys)
    info["splits"] = {"train": f"0:{len(kept_indices)}"}
    write_json(dst_root / "meta" / "info.json", info)

    return {
        "subset": subset_name,
        "task_name": subset_task_name,
        "kept_episodes": len(kept_indices),
        "skipped_missing_video": skipped_missing_video,
        "skipped_missing_data": skipped_missing_data,
        "total_frames": total_frames,
        "output_root": str(dst_root.resolve()),
    }


def split_dataset(
    *,
    src_root: Path,
    label_csv: Path,
    output_root: Path,
    task_map: dict[str, str] | None = None,
    target_labels: list[str] | None = None,
    drop_unlabeled: bool = True,
    require_all_videos: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Split a single dataset into per-label subsets. Importable API.

    Returns a list of result dicts, one per label subset.
    """
    info_path = src_root / "meta" / "info.json"
    episodes_path = src_root / "meta" / "episodes.jsonl"
    episodes_stats_path = src_root / "meta" / "episodes_stats.jsonl"

    for path in [info_path, episodes_path, episodes_stats_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required metadata file not found: {path}")

    src_info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    episodes_stats = read_jsonl(episodes_stats_path)
    episodes_by_idx = {int(row["episode_index"]): row for row in episodes}
    episodes_stats_by_idx = {int(row["episode_index"]): row for row in episodes_stats}
    video_keys = _video_keys(src_info)
    label_map = _load_label_map(label_csv)

    by_label: dict[str, list[int]] = {}
    unlabeled_count = 0
    for ep in sorted(episodes, key=lambda x: int(x["episode_index"])):
        ep_idx = int(ep["episode_index"])
        label = label_map.get(ep_idx, "")
        if not label:
            unlabeled_count += 1
            continue
        by_label.setdefault(label, []).append(ep_idx)

    if target_labels:
        filtered = {l: by_label[l] for l in target_labels if l in by_label}
        by_label = filtered

    if task_map is None:
        task_map = {}

    results = []
    for label in sorted(by_label.keys()):
        task_name = task_map.get(label, label)
        dst_root = output_root / label

        if dst_root.exists():
            if overwrite:
                if not dry_run:
                    shutil.rmtree(dst_root)
            elif not dry_run:
                raise FileExistsError(f"Output exists: {dst_root}. Use overwrite=True.")

        result = _build_subset(
            subset_name=label,
            subset_indices=by_label[label],
            subset_task_name=task_name,
            src_root=src_root,
            src_info=src_info,
            episodes_by_idx=episodes_by_idx,
            episodes_stats_by_idx=episodes_stats_by_idx,
            video_keys=video_keys,
            dst_root=dst_root,
            require_all_videos=require_all_videos,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LeRobot v2.1 dataset into per-label subsets.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output root. Creates one subdir per label.")
    parser.add_argument(
        "--task-map", nargs="*", default=[],
        help="Label->task_name as 'LABEL=task_name'. Unmapped labels use the label as task name.",
    )
    parser.add_argument("--labels", nargs="*", default=[],
                        help="Only split these labels. Default: all labels found.")
    parser.add_argument("--drop-unlabeled", action="store_true")
    parser.add_argument("--require-all-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_map: dict[str, str] = {}
    for item in args.task_map:
        if "=" not in item:
            raise ValueError(f"Invalid task-map entry: {item}")
        k, v = item.split("=", 1)
        task_map[k.strip()] = v.strip()

    results = split_dataset(
        src_root=args.src_root,
        label_csv=args.label_csv,
        output_root=args.output_root,
        task_map=task_map,
        target_labels=args.labels or None,
        drop_unlabeled=args.drop_unlabeled,
        require_all_videos=args.require_all_videos,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    for r in results:
        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n{prefix}Subset: {r['subset']}")
        for k, v in r.items():
            print(f"  {k}: {v}")

    print(f"\nDone. {len(results)} subsets.")


if __name__ == "__main__":
    main()
