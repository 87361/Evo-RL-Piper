"""Sample browsing, labeling, and progress tracking API (multi-dataset)."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from wbcd_claw.config import AppConfig, DatasetEntry

EP_RE = re.compile(r"(episode_\d+)\.mp4$")
DEFAULT_CATEGORIES = ["A", "B", "uncertain"]

router = APIRouter(prefix="/api/samples", tags=["samples"])

_state: dict = {}


class LabelPayload(BaseModel):
    episode_id: str
    label: str
    note: str = ""


class ProgressPayload(BaseModel):
    last_episode_id: str
    filter_mode: str = "all"


# ---------- data helpers ----------


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


def _categories_path(label_csv: Path) -> Path:
    return label_csv.with_name(f"{label_csv.stem}_categories.json")


def load_categories(categories_json: Path, rows: dict[str, dict[str, str]]) -> list[str]:
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


# ---------- progress persistence ----------

def _progress_path(label_csv: Path) -> Path:
    return label_csv.with_name(f"{label_csv.stem}_progress.json")


def load_progress(label_csv: Path) -> dict:
    pp = _progress_path(label_csv)
    if pp.exists():
        return json.loads(pp.read_text(encoding="utf-8"))
    return {}


def save_progress(label_csv: Path, data: dict) -> None:
    pp = _progress_path(label_csv)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- multi-dataset state ----------

def _ds(ds: str | None = None) -> dict:
    """Get state for a specific dataset, or the first one."""
    datasets = _state.get("datasets", {})
    if not datasets:
        raise RuntimeError("no datasets loaded")
    if ds and ds in datasets:
        return datasets[ds]
    return next(iter(datasets.values()))


def _ds_name(ds: str | None = None) -> str:
    datasets = _state.get("datasets", {})
    if ds and ds in datasets:
        return ds
    return next(iter(datasets.keys()))


# ---------- init ----------

def init_sample_state(config: AppConfig) -> None:
    datasets: dict[str, dict] = {}
    for entry in config.datasets:
        episodes = discover(entry.video_root)
        labels = load_csv(entry.label_csv)
        categories_json = _categories_path(entry.label_csv)
        categories = load_categories(categories_json, labels)
        progress = load_progress(entry.label_csv)
        datasets[entry.name] = {
            "entry": entry,
            "episodes": episodes,
            "labels": labels,
            "categories_json": categories_json,
            "categories": categories,
            "progress": progress,
        }
    _state["config"] = config
    _state["datasets"] = datasets


# ---------- routes ----------

@router.get("/datasets")
def list_datasets():
    ds_list = []
    for name, ds in _state.get("datasets", {}).items():
        labeled = sum(1 for r in ds["labels"].values() if r.get("label"))
        ds_list.append({
            "name": name,
            "episode_count": len(ds["episodes"]),
            "labeled_count": labeled,
            "unlabeled_count": len(ds["episodes"]) - labeled,
        })
    return {"ok": True, "datasets": ds_list}


@router.get("/meta")
def meta(ds: str = ""):
    d = _ds(ds or None)
    episodes = d["episodes"]
    labels = d["labels"]
    categories = d["categories"]
    labeled_count = sum(1 for r in labels.values() if r.get("label"))
    return {
        "dataset": _ds_name(ds or None),
        "episode_count": len(episodes),
        "labeled_count": labeled_count,
        "unlabeled_count": len(episodes) - labeled_count,
        "categories": categories,
    }


@router.get("/list")
def list_episodes(ds: str = "", q: str = "", lf: str = "all"):
    d = _ds(ds or None)
    episodes = d["episodes"]
    labels = d["labels"]
    items = []
    for ep in sorted(episodes.keys()):
        if q and q.lower() not in ep.lower():
            continue
        label = labels.get(ep, {}).get("label", "")
        if lf == "unlabeled" and label:
            continue
        if lf not in {"all", "unlabeled"} and label != lf:
            continue
        items.append({
            "episode_id": ep,
            "camera_count": len(episodes[ep]),
            "label": label,
        })
    return {"items": items}


@router.get("/episode/{episode_id}")
def get_episode(episode_id: str, ds: str = ""):
    d = _ds(ds or None)
    ds_name = _ds_name(ds or None)
    episodes = d["episodes"]
    labels = d["labels"]
    if episode_id not in episodes:
        return {"ok": False, "error": "episode not found"}
    videos = [
        {"camera": cam, "url": f"/media/{ds_name}/{rel}"}
        for cam, rel in sorted(episodes[episode_id].items())
    ]
    row = labels.get(episode_id, {})
    ep_list = sorted(episodes.keys())
    idx = ep_list.index(episode_id) if episode_id in ep_list else -1
    return {
        "ok": True,
        "episode_id": episode_id,
        "videos": videos,
        "label": row.get("label", ""),
        "note": row.get("note", ""),
        "index": idx,
        "total": len(ep_list),
    }


@router.post("/label")
def save_label(payload: LabelPayload, ds: str = ""):
    d = _ds(ds or None)
    entry: DatasetEntry = d["entry"]
    episodes = d["episodes"]
    labels = d["labels"]
    categories = d["categories"]
    if payload.episode_id not in episodes:
        return {"ok": False, "error": "episode not found"}
    normalized_label = payload.label.strip()
    if normalized_label and normalized_label not in categories:
        categories.append(normalized_label)
        write_categories(d["categories_json"], categories)
    labels[payload.episode_id] = {
        "episode_id": payload.episode_id,
        "label": normalized_label,
        "note": payload.note,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_csv(entry.label_csv, labels)
    return {"ok": True, "categories": categories}


@router.delete("/label/{episode_id}")
def delete_label(episode_id: str, ds: str = ""):
    d = _ds(ds or None)
    entry: DatasetEntry = d["entry"]
    episodes = d["episodes"]
    labels = d["labels"]
    if episode_id not in episodes:
        return {"ok": False, "error": "episode not found"}
    labels.pop(episode_id, None)
    write_csv(entry.label_csv, labels)
    return {"ok": True}


@router.get("/categories")
def list_categories(ds: str = ""):
    d = _ds(ds or None)
    return {"ok": True, "categories": d["categories"]}


@router.post("/categories")
def add_category(name: str, ds: str = ""):
    d = _ds(ds or None)
    categories = d["categories"]
    normalized = str(name).strip()
    if not normalized:
        return {"ok": False, "error": "empty"}
    if normalized not in categories:
        categories.append(normalized)
        write_categories(d["categories_json"], categories)
    return {"ok": True, "categories": categories}


@router.get("/progress")
def get_progress(ds: str = ""):
    d = _ds(ds or None)
    return {"ok": True, **d["progress"]}


@router.post("/progress")
def update_progress(payload: ProgressPayload, ds: str = ""):
    d = _ds(ds or None)
    entry: DatasetEntry = d["entry"]
    d["progress"] = {
        "last_episode_id": payload.last_episode_id,
        "filter_mode": payload.filter_mode,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_progress(entry.label_csv, d["progress"])
    return {"ok": True}
