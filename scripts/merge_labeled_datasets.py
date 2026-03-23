#!/usr/bin/env python
"""Merge episodes of the same label from multiple LeRobot v2.1 datasets.

Given N source datasets each with a label CSV, this script collects all
episodes sharing a label and builds one merged output dataset per label.

Usage:
    PYTHONPATH=src python scripts/merge_labeled_datasets.py \
      --sources \
        /path/to/dataset1::/path/to/dataset1/task_labels.csv \
        /path/to/dataset2::/path/to/dataset2/task_labels.csv \
      --output-root /path/to/merged_output \
      --task-map A=shirt_open_middle B=shirt_flatten \
      --drop-unlabeled \
      --require-all-videos \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
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


class SourceDataset:
    """One source dataset with its label map."""

    def __init__(self, src_root: Path, label_csv: Path):
        self.src_root = src_root
        self.label_csv = label_csv
        self.info = read_json(src_root / "meta" / "info.json")
        self.episodes = read_jsonl(src_root / "meta" / "episodes.jsonl")
        self.episodes_stats = read_jsonl(src_root / "meta" / "episodes_stats.jsonl")
        self.episodes_by_idx = {int(r["episode_index"]): r for r in self.episodes}
        self.stats_by_idx = {int(r["episode_index"]): r for r in self.episodes_stats}
        self.label_map = _load_label_map(label_csv)
        self.video_keys = _video_keys(self.info)
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_tmpl = str(self.info["data_path"])
        self.video_path_tmpl = str(self.info.get("video_path", ""))

    def data_path(self, ep_idx: int) -> Path:
        chunk = _episode_chunk(ep_idx, self.chunks_size)
        return self.src_root / self.data_path_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)

    def video_path(self, ep_idx: int, video_key: str) -> Path:
        chunk = _episode_chunk(ep_idx, self.chunks_size)
        return self.src_root / self.video_path_tmpl.format(
            episode_chunk=chunk, video_key=video_key, episode_index=ep_idx)

    def labeled_episodes(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for ep in sorted(self.episodes, key=lambda x: int(x["episode_index"])):
            ep_idx = int(ep["episode_index"])
            label = self.label_map.get(ep_idx, "")
            if label:
                result.setdefault(label, []).append(ep_idx)
        return result


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_merged_subset(
    *,
    label: str,
    task_name: str,
    sources: list[tuple[SourceDataset, list[int]]],
    dst_root: Path,
    require_all_videos: bool,
    dry_run: bool,
) -> dict:
    """Build one merged dataset for a single label from multiple sources."""

    ref_info = sources[0][0].info
    ref_video_keys = sources[0][0].video_keys
    chunks_size = int(ref_info.get("chunks_size", 1000))
    data_path_tmpl = str(ref_info["data_path"])
    video_path_tmpl = str(ref_info.get("video_path", ""))

    collected: list[tuple[SourceDataset, int]] = []
    skipped_video: list[str] = []
    skipped_data: list[str] = []

    for src, ep_indices in sources:
        for ep_idx in ep_indices:
            tag = f"{src.src_root.name}/ep_{ep_idx}"
            if not src.data_path(ep_idx).exists():
                skipped_data.append(tag)
                continue
            if require_all_videos and src.video_path_tmpl:
                missing = False
                for vk in ref_video_keys:
                    if not src.video_path(ep_idx, vk).exists():
                        missing = True
                        break
                if missing:
                    skipped_video.append(tag)
                    continue
            collected.append((src, ep_idx))

    if dry_run:
        return {
            "label": label,
            "task_name": task_name,
            "total_episodes": len(collected),
            "skipped_missing_data": skipped_data,
            "skipped_missing_video": skipped_video,
            "sources": [(str(s.src_root.name), idx) for s, idx in collected],
        }

    total_frames = 0
    merged_episodes = []
    merged_stats = []

    for new_ep_idx, (src, old_ep_idx) in enumerate(collected):
        old_chunk = _episode_chunk(old_ep_idx, src.chunks_size)
        new_chunk = _episode_chunk(new_ep_idx, chunks_size)

        data_src = src.data_path(old_ep_idx)
        data_dst = dst_root / data_path_tmpl.format(episode_chunk=new_chunk, episode_index=new_ep_idx)
        episode_len = _rewrite_episode_parquet(
            data_src, data_dst,
            new_episode_index=new_ep_idx,
            new_task_index=0,
            new_global_index_start=total_frames,
        )

        if video_path_tmpl:
            for vk in ref_video_keys:
                vsrc = src.video_path(old_ep_idx, vk)
                if vsrc.exists():
                    vdst = dst_root / video_path_tmpl.format(
                        episode_chunk=new_chunk, video_key=vk, episode_index=new_ep_idx)
                    _copy_file(vsrc, vdst)

        item = dict(src.episodes_by_idx.get(old_ep_idx, {}))
        item["episode_index"] = new_ep_idx
        item["tasks"] = [task_name]
        item["length"] = episode_len
        merged_episodes.append(item)

        if old_ep_idx in src.stats_by_idx:
            merged_stats.append(_update_episode_stats_row(
                src.stats_by_idx[old_ep_idx],
                new_episode_index=new_ep_idx,
                new_global_index_start=total_frames,
                episode_len=episode_len,
            ))

        total_frames += episode_len

    write_jsonl(dst_root / "meta" / "episodes.jsonl", merged_episodes)
    write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", merged_stats)
    write_jsonl(dst_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_name}])

    info = dict(ref_info)
    info["total_episodes"] = len(collected)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    if "total_chunks" in info:
        info["total_chunks"] = _total_chunks(len(collected), chunks_size)
    if "total_videos" in info:
        info["total_videos"] = len(collected) * len(ref_video_keys)
    info["splits"] = {"train": f"0:{len(collected)}"}
    write_json(dst_root / "meta" / "info.json", info)

    return {
        "label": label,
        "task_name": task_name,
        "output_root": str(dst_root),
        "total_episodes": len(collected),
        "total_frames": total_frames,
        "skipped_missing_data": skipped_data,
        "skipped_missing_video": skipped_video,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge same-label episodes from multiple LeRobot v2.1 datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sources", nargs="+", required=True,
        help="Source specs as 'dataset_root::label_csv' pairs. "
             "Example: /data/ds1::/data/ds1/task_labels.csv",
    )
    p.add_argument("--output-root", type=Path, required=True,
                   help="Output root. Creates one subdir per label.")
    p.add_argument(
        "--task-map", nargs="*", default=[],
        help="Label->task_name mappings as 'LABEL=task_name'. "
             "Example: A=shirt_open_middle B=shirt_flatten. "
             "Unmapped labels use the label itself as task name.",
    )
    p.add_argument("--labels", nargs="*", default=[],
                   help="Only merge these labels. Default: all labels found.")
    p.add_argument("--drop-unlabeled", action="store_true",
                   help="Drop episodes without labels in any CSV.")
    p.add_argument("--require-all-videos", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    task_map: dict[str, str] = {}
    for item in args.task_map:
        if "=" not in item:
            raise ValueError(f"Invalid task-map entry (need LABEL=task_name): {item}")
        k, v = item.split("=", 1)
        task_map[k.strip()] = v.strip()

    source_datasets: list[SourceDataset] = []
    for spec in args.sources:
        if "::" not in spec:
            raise ValueError(f"Invalid source spec (need 'root::csv'): {spec}")
        root_str, csv_str = spec.split("::", 1)
        root_path = Path(root_str.strip())
        csv_path = Path(csv_str.strip())
        if not root_path.exists():
            raise FileNotFoundError(f"Source root not found: {root_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Label CSV not found: {csv_path}")
        source_datasets.append(SourceDataset(root_path, csv_path))

    all_labels: dict[str, list[tuple[SourceDataset, list[int]]]] = {}
    for src in source_datasets:
        labeled = src.labeled_episodes()
        for label, ep_indices in labeled.items():
            all_labels.setdefault(label, []).append((src, ep_indices))

    target_labels = set(args.labels) if args.labels else set(all_labels.keys())

    print(f"Found labels: {sorted(all_labels.keys())}")
    print(f"Target labels: {sorted(target_labels)}")
    for label in sorted(all_labels.keys()):
        if label not in target_labels:
            continue
        counts = [(str(s.src_root.name), len(indices)) for s, indices in all_labels[label]]
        total = sum(c for _, c in counts)
        print(f"  {label}: {total} episodes from {len(counts)} sources {counts}")

    results = []
    for label in sorted(target_labels):
        if label not in all_labels:
            print(f"  WARNING: label '{label}' not found in any source, skipping")
            continue

        task_name = task_map.get(label, label)
        dst_root = args.output_root / label

        if dst_root.exists():
            if args.overwrite:
                if not args.dry_run:
                    shutil.rmtree(dst_root)
            elif not args.dry_run:
                raise FileExistsError(f"Output exists: {dst_root}. Use --overwrite.")

        result = build_merged_subset(
            label=label,
            task_name=task_name,
            sources=all_labels[label],
            dst_root=dst_root,
            require_all_videos=args.require_all_videos,
            dry_run=args.dry_run,
        )
        results.append(result)
        print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Result for label={label}:")
        for k, v in result.items():
            if k == "sources":
                print(f"  {k}: ({len(v)} entries)")
            else:
                print(f"  {k}: {v}")

    print(f"\nDone. {len(results)} label subsets processed.")


if __name__ == "__main__":
    main()
