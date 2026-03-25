#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pyarrow.parquet as pq

from data_ops import (
    _load_source_info,
    categories_path,
    discover,
    load_categories,
    load_csv,
    scan_lerobot_datasets,
    write_categories,
    write_csv,
)
from page import PAGE

class LabelPayload(BaseModel):
    episode_id: str
    label: str
    note: str = ""


class RenameCategoryPayload(BaseModel):
    old_name: str
    new_name: str
    apply_to_all_sources: bool = True


class SplitPayload(BaseModel):
    mode: str = "split"
    output_root: str
    task_map: dict[str, str] = {}
    labels: list[str] = []
    require_all_videos: bool = True
    overwrite: bool = True
    selected_sources: list[str] = []


class AddSourcePayload(BaseModel):
    name: str = ""
    dataset_root: str
    label_csv: str = ""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def app_factory(
    video_root: Path,
    label_csv: Path,
    categories_json_override: Path | None = None,
    merge_sources: list[dict] | None = None,
    datasets_scan_root: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Episode Review")
    episodes = discover(video_root)
    labels = load_csv(label_csv)
    categories_json = categories_json_override or categories_path(label_csv)
    categories = load_categories(categories_json, labels)
    dataset_root = video_root.parent
    info_path = dataset_root / "meta" / "info.json"
    info: dict = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    joint_names = info.get("features", {}).get("agent_pos", {}).get("names", [])
    app.mount("/media", StaticFiles(directory=str(video_root)), name="media")

    # Merge sources state: list of {name, dataset_root, label_csv, video_root, info_cache}
    sources_state: list[dict] = []
    # Always include the primary dataset as first source
    primary_name = dataset_root.name
    sources_state.append({
        "name": primary_name,
        "dataset_root": str(dataset_root),
        "label_csv": str(label_csv),
        "video_root": str(video_root),
        "is_primary": True,
    })
    if merge_sources:
        for ms in merge_sources:
            sources_state.append({
                "name": ms["name"],
                "dataset_root": ms["dataset_root"],
                "label_csv": ms["label_csv"],
                "video_root": ms.get("video_root", str(Path(ms["dataset_root"]) / "videos")),
                "is_primary": False,
            })
            vr = Path(ms.get("video_root", str(Path(ms["dataset_root"]) / "videos")))
            if vr.exists():
                mount_name = f"media_src_{ms['name']}"
                app.mount(f"/media-src/{ms['name']}", StaticFiles(directory=str(vr)), name=mount_name)

    scan_root = datasets_scan_root

    # Split/merge async job state
    split_job: dict = {"running": False, "result": None, "error": None}
    split_lock = threading.Lock()

    def _episode_idx(episode_id: str) -> int:
        return int(episode_id.split("_")[-1])

    def _parquet_path(episode_id: str) -> Path:
        episode_idx = _episode_idx(episode_id)
        chunk = episode_idx // chunks_size
        return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"

    def _check_video_quality(video_rel_path: str) -> dict:
        full_path = video_root / video_rel_path
        result = {
            "path": video_rel_path,
            "exists": full_path.exists(),
            "size_bytes": int(full_path.stat().st_size) if full_path.exists() else 0,
            "open_ok": False, "read_first_frame_ok": False,
            "fps": 0.0, "frame_count": 0, "width": 0, "height": 0,
            "warnings": [],
        }
        if not full_path.exists():
            result["warnings"].append("file_missing")
            return result
        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
            result["warnings"].append("open_failed")
            cap.release()
            return result
        result["open_ok"] = True
        result["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        result["read_first_frame_ok"] = bool(ok)
        cap.release()
        if result["fps"] <= 0: result["warnings"].append("fps_invalid")
        if result["frame_count"] <= 0: result["warnings"].append("frame_count_invalid")
        if result["width"] <= 0 or result["height"] <= 0: result["warnings"].append("resolution_invalid")
        if not result["read_first_frame_ok"]: result["warnings"].append("first_frame_decode_failed")
        if result["size_bytes"] <= 0: result["warnings"].append("size_zero")
        return result

    def _pick_thumb_cam(cams: dict[str, str]) -> str:
        for cam_key in cams:
            if "left_wrist" in cam_key:
                return cam_key
        return next(iter(cams.keys())) if cams else ""

    def _get_source_info(src: dict) -> dict:
        ds_root = Path(src["dataset_root"])
        lc = Path(src["label_csv"])
        si = _load_source_info(ds_root, lc)
        return {
            "name": src["name"],
            "dataset_root": src["dataset_root"],
            "label_csv": src["label_csv"],
            **si,
        }

    # ---- Routes ----

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return PAGE

    @app.get("/api/list")
    def list_episodes(q: str = "", lf: str = "all", page: int = 1, per_page: int = 12) -> dict:
        items = []
        for ep in sorted(episodes.keys()):
            if q and q.lower() not in ep.lower():
                continue
            label = labels.get(ep, {}).get("label", "")
            if lf == "unlabeled" and label:
                continue
            if lf not in {"all", "unlabeled"} and label != lf:
                continue
            cams = episodes[ep]
            thumb_cam = _pick_thumb_cam(cams)
            thumb = cams.get(thumb_cam, "")
            items.append({
                "episode_id": ep,
                "camera_count": len(cams),
                "label": label,
                "note": labels.get(ep, {}).get("note", ""),
                "thumb_url": f"media/{thumb}" if thumb else "",
                "thumb_cam": thumb_cam,
            })
        total = len(items)
        return {"items": items, "total": total, "page": page, "per_page": per_page}

    @app.get("/api/episode/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        if episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        videos = [
            {"camera": cam, "url": f"media/{rel}", "rel_path": rel}
            for cam, rel in sorted(episodes[episode_id].items())
        ]
        row = labels.get(episode_id, {})
        return {
            "ok": True, "episode_id": episode_id, "videos": videos,
            "label": row.get("label", ""), "note": row.get("note", ""),
        }

    @app.get("/api/episode/{episode_id}/joints")
    def get_joints(episode_id: str) -> dict:
        if episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        parquet_path = _parquet_path(episode_id)
        if not parquet_path.exists():
            return {"ok": False, "error": f"parquet not found: {parquet_path}"}
        table = pq.read_table(parquet_path, columns=["frame_index", "agent_pos"])
        frame_index = [int(v) for v in table.column("frame_index").to_pylist()]
        agent_pos = table.column("agent_pos").to_pylist()
        return {"ok": True, "episode_id": episode_id, "joint_names": joint_names,
                "frame_index": frame_index, "agent_pos": agent_pos}

    @app.get("/api/episode/{episode_id}/quality")
    def get_quality(episode_id: str) -> dict:
        if episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        checks = []
        for camera_name, rel_path in sorted(episodes[episode_id].items()):
            check = _check_video_quality(rel_path)
            check["camera"] = camera_name
            checks.append(check)
        return {"ok": True, "episode_id": episode_id, "checks": checks}

    @app.post("/api/label")
    def save(payload: LabelPayload) -> dict:
        if payload.episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        normalized_label = payload.label.strip()
        if normalized_label and normalized_label not in categories:
            categories.append(normalized_label)
            write_categories(categories_json, categories)
        labels[payload.episode_id] = {
            "episode_id": payload.episode_id,
            "label": normalized_label,
            "note": payload.note,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_csv(label_csv, labels)
        return {"ok": True, "categories": categories}

    @app.delete("/api/label/{episode_id}")
    def delete_label(episode_id: str) -> dict:
        if episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        removed = labels.pop(episode_id, None) is not None
        write_csv(label_csv, labels)
        return {"ok": True, "removed": removed}

    @app.get("/api/categories")
    def list_categories() -> dict:
        fresh = load_categories(categories_json, labels)
        for c in fresh:
            if c not in categories:
                categories.append(c)
        return {"ok": True, "categories": categories}

    @app.post("/api/categories")
    def add_category(name: str) -> dict:
        normalized = str(name).strip()
        if not normalized:
            return {"ok": False, "error": "category empty"}
        if normalized in categories:
            return {"ok": True, "categories": categories}
        categories.append(normalized)
        write_categories(categories_json, categories)
        return {"ok": True, "categories": categories}

    @app.delete("/api/categories/{name}")
    def delete_category(name: str, purge_labeled_rows: bool = False) -> dict:
        normalized = str(name).strip()
        if normalized not in categories:
            return {"ok": False, "error": "category not found"}
        if len(categories) <= 1:
            return {"ok": False, "error": "at least one category required"}
        categories.remove(normalized)
        affected = 0
        if purge_labeled_rows:
            to_delete = [ep for ep, row in labels.items()
                         if str(row.get("label", "")).strip() == normalized]
            for ep in to_delete:
                labels.pop(ep, None)
            affected = len(to_delete)
        else:
            for row in labels.values():
                if str(row.get("label", "")).strip() == normalized:
                    row["label"] = ""
                    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    affected += 1
        write_categories(categories_json, categories)
        write_csv(label_csv, labels)
        return {"ok": True, "categories": categories, "affected": affected}

    @app.get("/api/meta")
    def meta() -> dict:
        fresh = load_categories(categories_json, labels)
        for c in fresh:
            if c not in categories:
                categories.append(c)
        return {"episode_count": len(episodes), "label_csv": str(label_csv), "categories": categories}

    # ---- Dataset info ----

    @app.get("/api/dataset-info")
    def dataset_info() -> dict:
        label_stats: dict[str, int] = {}
        for row in labels.values():
            lbl = str(row.get("label", "")).strip()
            if lbl:
                label_stats[lbl] = label_stats.get(lbl, 0) + 1
        labeled_count = sum(label_stats.values())
        video_keys = sorted(
            k for k, v in info.get("features", {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        )
        return {
            "ok": True,
            "codebase_version": str(info.get("codebase_version", "?")),
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "fps": int(info.get("fps", 0)),
            "chunks_size": chunks_size,
            "video_keys": video_keys,
            "robot_type": str(info.get("robot_type", "")),
            "dataset_root": str(dataset_root),
            "label_csv": str(label_csv),
            "label_stats": label_stats,
            "total_episodes_with_video": len(episodes),
            "labeled_count": labeled_count,
            "unlabeled_count": len(episodes) - labeled_count,
        }

    # ---- Merge sources ----

    @app.get("/api/merge-sources")
    def get_merge_sources() -> dict:
        out = []
        all_labels: dict[str, int] = {}
        for src in sources_state:
            si = _get_source_info(src)
            out.append(si)
            for lbl, cnt in si["label_stats"].items():
                all_labels[lbl] = all_labels.get(lbl, 0) + cnt
        return {"ok": True, "sources": out, "all_labels": all_labels,
                "primary_name": primary_name, "scan_root": str(scan_root or "")}

    @app.get("/api/scan-datasets")
    def api_scan_datasets() -> dict:
        if not scan_root:
            return {"ok": False, "error": "no scan root configured"}
        existing_roots = {s["dataset_root"] for s in sources_state}
        all_ds = scan_lerobot_datasets(scan_root)
        available = [d for d in all_ds if d["dataset_root"] not in existing_roots]
        return {"ok": True, "datasets": available, "scan_root": str(scan_root)}

    @app.post("/api/merge-sources/add")
    def add_merge_source(payload: AddSourcePayload) -> dict:
        ds_root = Path(payload.dataset_root).resolve()
        if not ds_root.exists():
            return {"ok": False, "error": f"dataset root not found: {ds_root}"}
        if not (ds_root / "meta" / "info.json").exists():
            return {"ok": False, "error": f"not a LeRobot dataset (no meta/info.json): {ds_root}"}

        existing_roots = {s["dataset_root"] for s in sources_state}
        if str(ds_root) in existing_roots:
            return {"ok": False, "error": "already added"}

        name = payload.name.strip() or ds_root.name
        if any(s["name"] == name for s in sources_state):
            name = f"{name}_{len(sources_state)}"

        lc_path = Path(payload.label_csv) if payload.label_csv else ds_root / "task_labels.csv"
        vr_path = ds_root / "videos"

        sources_state.append({
            "name": name,
            "dataset_root": str(ds_root),
            "label_csv": str(lc_path),
            "video_root": str(vr_path),
            "is_primary": False,
        })
        if vr_path.exists():
            mount_name = f"media_src_{name}"
            app.mount(f"/media-src/{name}", StaticFiles(directory=str(vr_path)), name=mount_name)

        return {"ok": True, "name": name}

    @app.delete("/api/merge-sources/{name}")
    def remove_merge_source(name: str) -> dict:
        for i, src in enumerate(sources_state):
            if src["name"] == name:
                if src.get("is_primary"):
                    return {"ok": False, "error": "cannot remove primary dataset"}
                sources_state.pop(i)
                return {"ok": True}
        return {"ok": False, "error": "source not found"}

    # ---- Rename category across sources ----

    @app.post("/api/rename-category")
    def rename_category(payload: RenameCategoryPayload) -> dict:
        old = payload.old_name.strip()
        new = payload.new_name.strip()
        if not old or not new:
            return {"ok": False, "error": "empty name"}
        if old == new:
            return {"ok": False, "error": "same name"}

        targets = sources_state if payload.apply_to_all_sources else [sources_state[0]]
        details = []
        total_affected = 0

        for src in targets:
            lc = Path(src["label_csv"])
            rows = load_csv(lc)
            affected = 0
            for row in rows.values():
                if str(row.get("label", "")).strip() == old:
                    row["label"] = new
                    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    affected += 1
            if affected:
                write_csv(lc, rows)
            details.append({"source": src["name"], "affected": affected})
            total_affected += affected

        # Update primary in-memory labels
        for row in labels.values():
            if str(row.get("label", "")).strip() == old:
                row["label"] = new

        if old in categories:
            idx = categories.index(old)
            if new not in categories:
                categories[idx] = new
            else:
                categories.pop(idx)
            write_categories(categories_json, categories)

        return {"ok": True, "total_affected": total_affected, "details": details, "categories": categories}

    # ---- Split / Merge job ----

    @app.post("/api/split")
    def start_split(payload: SplitPayload) -> dict:
        with split_lock:
            if split_job["running"]:
                return {"ok": False, "error": "job already running"}
            split_job["running"] = True
            split_job["result"] = None
            split_job["error"] = None

        project_root = Path(__file__).resolve().parents[3]  # scripts/gui/episode_review/routes.py -> Evo-RL-Piper/

        if payload.mode == "merge":
            source_args = []
            selected = set(payload.selected_sources)
            for src in sources_state:
                if src["name"] in selected:
                    source_args.extend(["--sources", f"{src['dataset_root']}::{src['label_csv']}"])
            cmd = [
                sys.executable, str(project_root / "scripts" / "merge_labeled_datasets.py"),
                *source_args,
                "--output-root", payload.output_root,
            ]
        else:
            cmd = [
                sys.executable, str(project_root / "scripts" / "split_lerobot_v21_by_labels.py"),
                "--src-root", str(dataset_root),
                "--label-csv", str(label_csv),
                "--output-root", payload.output_root,
            ]

        for label, prompt in payload.task_map.items():
            cmd.extend(["--task-map", f"{label}={prompt}"])
        if payload.labels:
            cmd.extend(["--labels", *payload.labels])
        if payload.require_all_videos:
            cmd.append("--require-all-videos")
        if payload.overwrite:
            cmd.append("--overwrite")

        cmd_str = " ".join(cmd)

        def _run():
            env = {**__import__("os").environ, "PYTHONPATH": str(project_root / "src")}
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
            with split_lock:
                split_job["result"] = {
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-20000:] if len(proc.stdout) > 20000 else proc.stdout,
                    "stderr": proc.stderr[-10000:] if len(proc.stderr) > 10000 else proc.stderr,
                }
                split_job["running"] = False

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "command": cmd_str}

    @app.get("/api/split/status")
    def split_status() -> dict:
        with split_lock:
            return {
                "running": split_job["running"],
                "result": split_job["result"],
                "error": split_job["error"],
            }

    return app
