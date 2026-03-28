#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

EP_RE = re.compile(r"(episode_\d+)\.mp4$")
DEFAULT_CATEGORIES = ["A", "B", "uncertain"]


def discover(video_root: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for p in sorted(video_root.rglob("episode_*.mp4")):
        m = EP_RE.search(p.name)
        if not m:
            continue
        ep = m.group(1)
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
