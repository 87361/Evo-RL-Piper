#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

EP_RE = re.compile(r"(episode_\d+)\.mp4$")
DEFAULT_CATEGORIES = ["failed"]


def discover(dataset_root: Path) -> dict[str, dict[str, str]]:
    video_root = dataset_root / "videos"
    episodes_dir = dataset_root / "episodes"
    is_v2 = episodes_dir.exists() and episodes_dir.is_dir()
    
    grouped: dict[str, dict[str, str]] = {}
    if not video_root.exists():
        return grouped
        
    for p in sorted(video_root.rglob("episode_*.mp4")):
        m = EP_RE.search(p.name)
        if not m:
            continue
        ep = m.group(1)
        
        # [Phase 1: Source of Truth Replace]
        # Verify parquet exists for LeRobot v2.1. 
        # If it's merely a corrupted/dangling .mp4 without data, skip it instantly.
        if is_v2:
            pq_file = episodes_dir / f"{ep}.parquet"
            if not pq_file.exists():
                print(f"[WARN] Skipping dangling video {ep}: Missing {pq_file.name}")
                continue
                
        cam = p.parent.name
        grouped.setdefault(ep, {})[cam] = p.relative_to(video_root).as_posix()
        
    return grouped


def load_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ep = str(row.get("episode_id", "")).strip()
            if ep:
                rows[ep] = {
                    "episode_id": ep,
                    "label": str(row.get("label", "")).strip(),
                    "note": str(row.get("note", "")).strip(),
                    "updated_at": str(row.get("updated_at", "")).strip(),
                }
    return rows


def write_csv(csv_path: Path, rows: dict[str, dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode_id", "label", "note", "updated_at"]
        )
        writer.writeheader()
        for ep in sorted(rows.keys()):
            writer.writerow(rows[ep])
            
    # [Phase 1B: Auto-sync tasks to native meta/tasks.jsonl]
    # For compatibility with OpenPI and standard LeRobot Dataloaders that expect NLP instructions 
    try:
        dataset_root = csv_path.parent
        tasks_jsonl_path = dataset_root / "meta" / "tasks.jsonl"
        tasks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        unique_tasks = []
        for row in rows.values():
            label = str(row.get("label", "")).strip()
            if label and label not in unique_tasks:
                unique_tasks.append(label)
                
        # Format natively as {"task_index": int, "task": str}
        with tasks_jsonl_path.open("w", encoding="utf-8") as f:
            for idx, task in enumerate(unique_tasks):
                record = {"task_index": idx, "task": task}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to auto-sync tasks.jsonl: {e}")


def categories_path(label_csv: Path) -> Path:
    return label_csv.with_name(f"{label_csv.stem}_categories.json")


def load_categories(
    categories_json: Path, rows: dict[str, dict[str, str]]
) -> list[str]:
    categories: list[str] = []
    if categories_json.exists():
        data = json.loads(categories_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for item in data.get("categories", []):
                name = str(item).strip()
                if name and name not in categories:
                    categories.append(name)
    for item in DEFAULT_CATEGORIES:
        if item not in categories:
            categories.append(item)
    for row in rows.values():
        name = str(row.get("label", "")).strip()
        if name and name not in categories:
            categories.append(name)
    return categories


def write_categories(categories_json: Path, categories: list[str]) -> None:
    categories_json.parent.mkdir(parents=True, exist_ok=True)
    categories_json.write_text(
        json.dumps({"categories": categories}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_lerobot_datasets(root: Path) -> list[dict]:
    """Recursively find LeRobot v2.1 datasets under *root* (identified by meta/info.json)."""
    results: list[dict] = []
    if not root.exists():
        return results
    for info_path in sorted(root.rglob("meta/info.json")):
        ds_root = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        video_root = ds_root / "videos"
        label_csv = ds_root / "task_labels.csv"
        results.append({
            "name": ds_root.name,
            "dataset_root": str(ds_root),
            "video_root": str(video_root) if video_root.exists() else "",
            "label_csv": str(label_csv) if label_csv.exists() else "",
            "codebase_version": str(info.get("codebase_version", "?")),
            "total_episodes": int(info.get("total_episodes", 0)),
            "fps": int(info.get("fps", 0)),
        })
    return results


def _load_source_info(dataset_root: Path, label_csv: Path) -> dict:
    """Load metadata for a single merge source."""
    info_path = dataset_root / "meta" / "info.json"
    info: dict = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    label_map = load_csv(label_csv)
    label_stats: dict[str, int] = {}
    for row in label_map.values():
        lbl = str(row.get("label", "")).strip()
        if lbl:
            label_stats[lbl] = label_stats.get(lbl, 0) + 1
    return {
        "codebase_version": str(info.get("codebase_version", "?")),
        "total_episodes": int(info.get("total_episodes", 0)),
        "fps": int(info.get("fps", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "label_stats": label_stats,
    }


def save_subtask_segment(dataset_root: Path, episode_id: str, start_time: float, end_time: float, subtask: str):
    import pyarrow.parquet as pq
    import pyarrow as pa
    import math
    import json
    
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise Exception("meta/info.json not found")
        
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = info.get("fps", 30)
    
    start_frame = max(0, math.floor(start_time * fps))
    end_frame = math.ceil(end_time * fps)
    
    # 1. Ensure subtasks.jsonl holds this subtask
    subtasks_jsonl_path = dataset_root / "meta" / "subtasks.jsonl"
    subtasks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_subtasks = []
    if subtasks_jsonl_path.exists():
        with subtasks_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "subtask" in record:
                            existing_subtasks.append(record["subtask"])
                    except Exception:
                        pass
                    
    if subtask not in existing_subtasks:
        existing_subtasks.append(subtask)
        with subtasks_jsonl_path.open("a", encoding="utf-8") as f:
            record = {"subtask_index": len(existing_subtasks)-1, "subtask": subtask}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    subtask_idx = existing_subtasks.index(subtask)
    
    # 2. Modify Parquet
    ep_pq = dataset_root / "episodes" / f"{episode_id}.parquet"
    if not ep_pq.exists():
        raise Exception(f"Parquet file {ep_pq.name} not found. Ensure LeRobot V2.1 parity.")
        
    table = pq.read_table(ep_pq)
    num_rows = table.num_rows
    # Restrict end_frame to the total number of frames to avoid out-of-bounds mapping
    end_frame = min(num_rows, end_frame)
    if start_frame >= num_rows:
        raise Exception("Annotated start time exceeds video length.")
        
    col_names = table.column_names
    if "subtask_index" in col_names:
        subtasks_arr = table["subtask_index"].to_pylist()
        table = table.drop(["subtask_index"])
    else:
        # Default subtask index is usually 0 if there's only one subtask, or -1 if unlabeled.
        subtasks_arr = [-1] * num_rows
        
    for i in range(start_frame, end_frame):
        subtasks_arr[i] = subtask_idx
        
    subtasks_array = pa.array(subtasks_arr, type=pa.int64()) # Use int64 for pyarrow/pandas safety in LeRobot
    table = table.append_column("subtask_index", subtasks_array)
    
    # Overwrite
    pq.write_table(table, ep_pq)
