"""
WBCD Web Console — Main Server
A lightweight FastAPI server for mobile-first training management.
"""
import os
import re
import sys
import time
import signal
import subprocess
import uuid
import json
import platform
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

import psutil
import yaml
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import auth
import data_ops

# ──────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────
app = FastAPI(title="WBCD Web Console", docs_url=None, redoc_url=None)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(config.JOBS_DIR, exist_ok=True)
os.makedirs(config.TEMPLATES_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Track server start time
SERVER_START_TIME = time.time()

# Background task state for data ops
_DATA_OPS_JOBS: Dict[str, Dict] = {}
_DATA_OPS_LOCK = threading.Lock()

# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class LoginRequest(BaseModel):
    password: str


class ConfigUpdateRequest(BaseModel):
    template: str
    params: dict


class JobStartRequest(BaseModel):
    template: str
    params: dict = {}


class PipelineStartRequest(BaseModel):
    dataset_path: str
    exp_name: str
    config_name: str = "pi05_aloha_wbcd_lora"
    gpu_indices: list = []
    batch_size: int = 64
    fsdp_devices: int = 4
    num_train_steps: int = 20000
    save_interval: int = 1000
    min_range: float = 0.1
    resume: bool = False
    overwrite: bool = True
    wandb_enabled: bool = True
    skip_norm_stats: bool = False
    skip_postprocess: bool = False


class LabelPayload(BaseModel):
    dataset_root: str
    episode_id: str
    label: str
    note: str = ""


class CategoryPayload(BaseModel):
    dataset_root: str
    name: str


class DataOpPayload(BaseModel):
    mode: str  # 'split' or 'merge'
    output_root: str
    task_map: dict = {}
    labels: list = []
    selected_sources: list = [] # for merge
    dataset_root: str = "" # for split
    label_csv: str = "" # for split


# ──────────────────────────────────────────────
# Auth-free endpoints
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check — no auth required."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(time.time() - SERVER_START_TIME),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/login")
async def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    if req.password != config.AUTH_PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong password")
    token = auth.create_token()
    response = JSONResponse({"ok": True, "token": token})
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        max_age=config.JWT_EXPIRE_HOURS * 3600,
        samesite="lax",
    )
    return response


@app.post("/api/logout")
async def logout():
    """Clear the auth cookie."""
    response = JSONResponse({"ok": True})
    response.delete_cookie("token")
    return response


# ──────────────────────────────────────────────
# Auth dependency
# ──────────────────────────────────────────────
async def require_login(request: Request):
    return auth.require_auth(request)


# ──────────────────────────────────────────────
# System status
# ──────────────────────────────────────────────
@app.get("/api/system")
async def system_status(_=Depends(require_login)):
    """Return system metrics for the dashboard."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_count = psutil.cpu_count()
    load_1, load_5, load_15 = os.getloadavg()

    # GPU info (best-effort)
    gpu_info = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpu_info.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_total_mb": int(parts[4]),
                        "temperature": int(parts[5]),
                    })
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "uptime_seconds": int(time.time() - SERVER_START_TIME),
        "cpu": {
            "count": cpu_count,
            "percent": psutil.cpu_percent(interval=0.5),
            "load_1m": round(load_1, 2),
            "load_5m": round(load_5, 2),
            "load_15m": round(load_15, 2),
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "used_gb": round(disk.used / (1024**3), 1),
            "percent": round(disk.percent, 1),
        },
        "gpu": gpu_info,
    }

# ──────────────────────────────────────────────
# Dataset Review & Data Ops
# ──────────────────────────────────────────────

@app.get("/api/datasets/scan")
async def scan_datasets(_=Depends(require_login)):
    """Scan for LeRobot datasets in configured roots."""
    all_datasets = []
    for root in config.EPISODE_SCAN_ROOTS:
        if os.path.exists(root):
            all_datasets.extend(data_ops.scan_lerobot_datasets(Path(root)))
    
    # Attach categories to each dataset
    for ds in all_datasets:
        ds_path = Path(ds['dataset_root'])
        label_csv = ds_path / "task_labels.csv"
        cat_path = data_ops.categories_path(label_csv)
        labels = data_ops.load_csv(label_csv)
        ds['categories'] = data_ops.load_categories(cat_path, labels)
    
    # Sort by name
    all_datasets.sort(key=lambda x: x['name'])
    return {"datasets": all_datasets, "roots": config.EPISODE_SCAN_ROOTS}


@app.get("/api/dataset/meta")
async def get_dataset_meta(path: str, _=Depends(require_login)):
    """Get metadata and label stats for a dataset."""
    ds_path = Path(path)
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    label_csv = ds_path / "task_labels.csv"
    info = data_ops._load_source_info(ds_path, label_csv)
    
    # Also get categories
    labels_dict = data_ops.load_csv(label_csv)
    cat_path = data_ops.categories_path(label_csv)
    categories = data_ops.load_categories(cat_path, labels_dict)
    
    # Count episodes with video
    video_root = ds_path / "videos"
    episodes = data_ops.discover(video_root) if video_root.exists() else {}
    
    return {
        "path": path,
        "name": ds_path.name,
        "info": info,
        "categories": categories,
        "total_episodes_with_video": len(episodes),
    }


@app.get("/api/dataset/episodes")
async def list_episodes(path: str, q: str = "", lf: str = "all", page: int = 1, per_page: int = 10, _=Depends(require_login)):
    """List episodes for review with filtering and pagination."""
    ds_path = Path(path)
    video_root = ds_path / "videos"
    if not video_root.exists():
         return {"items": [], "total": 0}
         
    label_csv = ds_path / "task_labels.csv"
    episodes = data_ops.discover(video_root)
    labels = data_ops.load_csv(label_csv)
    
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
        # Pick head camera if available, otherwise first camera
        head_cam = next((c for c in cams if "head" in c.lower()), None)
        if not head_cam:
            head_cam = next(iter(cams.keys()), "")
        head_rel = cams.get(head_cam, "")
        head_video_url = f"/api/media-serve?path={path}/videos/{head_rel}" if head_rel else ""
        
        items.append({
            "episode_id": ep,
            "camera_count": len(cams),
            "label": label,
            "note": labels.get(ep, {}).get("note", ""),
            "head_video_url": head_video_url,
            "head_cam": head_cam,
        })
    
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "per_page": per_page
    }


@app.get("/api/dataset/episode/{episode_id}")
async def get_episode_details(path: str, episode_id: str, _=Depends(require_login)):
    """Get video URLs for an episode."""
    ds_path = Path(path)
    video_root = ds_path / "videos"
    episodes = data_ops.discover(video_root)
    
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
        
    label_csv = ds_path / "task_labels.csv"
    labels = data_ops.load_csv(label_csv)
    row = labels.get(episode_id, {})
    
    videos = []
    for cam, rel in sorted(episodes[episode_id].items()):
        # We'll serve these via a special media endpoint to avoid path traversal issues
        videos.append({
            "camera": cam,
            "url": f"/api/media-serve?path={path}/videos/{rel}"
        })
        
    return {
        "episode_id": episode_id,
        "videos": videos,
        "label": row.get("label", ""),
        "note": row.get("note", "")
    }


@app.get("/api/media-serve")
async def serve_media(path: str, _=Depends(require_login)):
    """Serve a video or image file from a dataset root."""
    # Security: check if it's under one of our allowed Roots
    # EPISODE_SCAN_ROOTS or DATASET_ROOTS
    if not (any(path.startswith(os.path.realpath(r)) for r in config.EPISODE_SCAN_ROOTS) or 
            any(path.startswith(os.path.realpath(r)) for r in config.DATASET_ROOTS)):
         # We check realpath to be safe
         real_path = os.path.realpath(path)
         if not (any(real_path.startswith(os.path.realpath(r)) for r in config.EPISODE_SCAN_ROOTS) or 
                 any(real_path.startswith(os.path.realpath(r)) for r in config.DATASET_ROOTS)):
             raise HTTPException(status_code=403, detail="Access denied")
             
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Ensure correct MIME type for video files
    ext = os.path.splitext(path)[1].lower()
    media_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    media_type = media_types.get(ext)
    return FileResponse(path, media_type=media_type)


@app.post("/api/dataset/label")
async def save_label(payload: LabelPayload, _=Depends(require_login)):
    """Save an episode label."""
    ds_path = Path(payload.dataset_root)
    label_csv = ds_path / "task_labels.csv"
    
    labels = data_ops.load_csv(label_csv)
    cat_path = data_ops.categories_path(label_csv)
    categories = data_ops.load_categories(cat_path, labels)
    
    normalized_label = payload.label.strip()
    if normalized_label and normalized_label not in categories:
        categories.append(normalized_label)
        data_ops.write_categories(cat_path, categories)
        
    labels[payload.episode_id] = {
        "episode_id": payload.episode_id,
        "label": normalized_label,
        "note": payload.note,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }
    data_ops.write_csv(label_csv, labels)
    return {"ok": True, "categories": categories}


@app.post("/api/dataset/category/add")
async def add_category(payload: CategoryPayload, _=Depends(require_login)):
    """Add a new category to a dataset."""
    ds_path = Path(payload.dataset_root)
    label_csv = ds_path / "task_labels.csv"
    cat_path = data_ops.categories_path(label_csv)
    labels = data_ops.load_csv(label_csv)
    categories = data_ops.load_categories(cat_path, labels)
    
    new_cat = payload.name.strip()
    if not new_cat:
        raise HTTPException(status_code=400, detail="Name empty")
    
    if new_cat not in categories:
        categories.append(new_cat)
        data_ops.write_categories(cat_path, categories)
        
    return {"ok": True, "categories": categories}


class GlobalCategorySyncPayload(BaseModel):
    name: str


@app.get("/api/categories/global")
async def get_global_categories(_=Depends(require_login)):
    """Get the union of all categories across all datasets."""
    all_cats = set()
    for root in config.EPISODE_SCAN_ROOTS:
        if not os.path.exists(root):
            continue
        for ds in data_ops.scan_lerobot_datasets(Path(root)):
            ds_path = Path(ds['dataset_root'])
            label_csv = ds_path / "task_labels.csv"
            cat_path = data_ops.categories_path(label_csv)
            labels = data_ops.load_csv(label_csv)
            cats = data_ops.load_categories(cat_path, labels)
            all_cats.update(cats)
    return {"categories": sorted(all_cats)}


@app.post("/api/categories/sync")
async def sync_category_to_all(payload: GlobalCategorySyncPayload, _=Depends(require_login)):
    """Sync a new category to ALL datasets so they share the same label vocabulary."""
    new_cat = payload.name.strip()
    if not new_cat:
        raise HTTPException(status_code=400, detail="Name empty")
    
    synced = 0
    for root in config.EPISODE_SCAN_ROOTS:
        if not os.path.exists(root):
            continue
        for ds in data_ops.scan_lerobot_datasets(Path(root)):
            ds_path = Path(ds['dataset_root'])
            label_csv = ds_path / "task_labels.csv"
            cat_path = data_ops.categories_path(label_csv)
            labels = data_ops.load_csv(label_csv)
            cats = data_ops.load_categories(cat_path, labels)
            if new_cat not in cats:
                cats.append(new_cat)
                data_ops.write_categories(cat_path, cats)
                synced += 1
    
    return {"ok": True, "synced_count": synced}


@app.post("/api/dataset/category/delete")
async def delete_category(payload: CategoryPayload, purge: bool = False, _=Depends(require_login)):
    """Delete a category and optionally purge related labels."""
    ds_path = Path(payload.dataset_root)
    label_csv = ds_path / "task_labels.csv"
    cat_path = data_ops.categories_path(label_csv)
    labels = data_ops.load_csv(label_csv)
    categories = data_ops.load_categories(cat_path, labels)
    
    name = payload.name.strip()
    if name not in categories:
         raise HTTPException(status_code=404, detail="Category not found")
         
    categories.remove(name)
    affected = 0
    if purge:
        to_delete = [ep for ep, row in labels.items() if row.get("label") == name]
        for ep in to_delete:
            del labels[ep]
        affected = len(to_delete)
    else:
        for row in labels.values():
            if row.get("label") == name:
                row["label"] = ""
                affected += 1
                
    data_ops.write_categories(cat_path, categories)
    data_ops.write_csv(label_csv, labels)
    return {"ok": True, "categories": categories, "affected": affected}


@app.post("/api/dataset/ops/run")
async def run_data_op(payload: DataOpPayload, _=Depends(require_login)):
    """Start a split or merge background job."""
    with _DATA_OPS_LOCK:
        for job in _DATA_OPS_JOBS.values():
            if job.get("status") == "running":
                raise HTTPException(status_code=400, detail="A data operation is already running")
                
    job_id = str(uuid.uuid4())[:8]
    
    # We find project root for the scripts
    # This server is at scripts/gui_phone/server.py
    # Scripts are at scripts/merge_labeled_datasets.py or scripts/split_lerobot_v21_by_labels.py
    project_root = Path(__file__).resolve().parents[2] 
    
    cmd = [sys.executable]
    if payload.mode == "merge":
        script = project_root / "scripts" / "merge_labeled_datasets.py"
        cmd.append(str(script))
        for src in payload.selected_sources:
             # src is name::root::csv
             cmd.extend(["--sources", src])
    else:
        script = project_root / "scripts" / "split_lerobot_v21_by_labels.py"
        cmd.append(str(script))
        cmd.extend(["--src-root", payload.dataset_root])
        cmd.extend(["--label-csv", payload.label_csv])
        
    cmd.extend(["--output-root", payload.output_root])
    
    for label, prompt in payload.task_map.items():
        cmd.extend(["--task-map", f"{label}={prompt}"])
    
    if payload.labels:
        cmd.extend(["--labels", *payload.labels])
        
    cmd.append("--overwrite")
    
    def _run_job(jid, command):
        _DATA_OPS_JOBS[jid] = {"status": "running", "stdout": "", "stderr": ""}
        env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            _DATA_OPS_JOBS[jid]["stdout"] += line
            # Keep only last 20k chars
            if len(_DATA_OPS_JOBS[jid]["stdout"]) > 20000:
                _DATA_OPS_JOBS[jid]["stdout"] = _DATA_OPS_JOBS[jid]["stdout"][-20000:]
                
        ret = proc.wait()
        _DATA_OPS_JOBS[jid]["status"] = "completed" if ret == 0 else "failed"
        _DATA_OPS_JOBS[jid]["exit_code"] = ret

    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/dataset/ops/status")
async def get_data_op_status(job_id: str, _=Depends(require_login)):
    """Get status of a data operation job."""
    if job_id not in _DATA_OPS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return _DATA_OPS_JOBS[job_id]


# ──────────────────────────────────────────────
# Existing Dataset browsing (File browser)
# ──────────────────────────────────────────────
def _is_under_whitelist(path: str) -> bool:
    """Check that the resolved path is under one of the whitelisted roots."""
    real = os.path.realpath(path)
    return any(real.startswith(os.path.realpath(root)) for root in config.DATASET_ROOTS)


@app.get("/api/datasets")
async def list_datasets_files(_=Depends(require_login)):
    """List whitelisted dataset roots."""
    result = []
    for root in config.DATASET_ROOTS:
        if os.path.isdir(root):
            try:
                count = sum(1 for _ in os.scandir(root))
                result.append({"path": root, "name": os.path.basename(root), "items": count})
            except PermissionError:
                result.append({"path": root, "name": os.path.basename(root), "items": -1})
    return result


@app.get("/api/dataset/items")
async def dataset_items(path: str, _=Depends(require_login)):
    """List items in a directory under whitelisted roots."""
    if not _is_under_whitelist(path):
        # Also check if it's under EPISODE_SCAN_ROOTS which we also consider safe
        real = os.path.realpath(path)
        if not any(real.startswith(os.path.realpath(root)) for root in config.EPISODE_SCAN_ROOTS):
            raise HTTPException(status_code=403, detail="Access denied: path not in whitelist")
            
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Directory not found")

    items = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name)):
            info = {
                "name": entry.name,
                "path": entry.path,
                "is_dir": entry.is_dir(),
            }
            if not entry.is_dir():
                try:
                    stat = entry.stat()
                    info["size"] = stat.st_size
                    info["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                except OSError:
                    pass
                # Check if it's a previewable image
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                    info["preview"] = True
            items.append(info)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return {"path": path, "parent": os.path.dirname(path), "items": items}


@app.get("/api/dataset/preview")
async def dataset_preview(path: str, _=Depends(require_login)):
    """Serve an image file preview (only from whitelisted directories)."""
    if not _is_under_whitelist(path):
         real = os.path.realpath(path)
         if not any(real.startswith(os.path.realpath(root)) for root in config.EPISODE_SCAN_ROOTS):
            raise HTTPException(status_code=403, detail="Access denied")
            
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        raise HTTPException(status_code=400, detail="Not a previewable file")
    return FileResponse(path)


# ──────────────────────────────────────────────
# Training config templates
# ──────────────────────────────────────────────
def _load_templates() -> dict:
    """Load training templates from YAML files."""
    templates = {}
    if not os.path.isdir(config.TEMPLATES_DIR):
        return templates
    for fname in os.listdir(config.TEMPLATES_DIR):
        if fname.endswith((".yaml", ".yml")):
            fpath = os.path.join(config.TEMPLATES_DIR, fname)
            try:
                with open(fpath) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    templates[fname.rsplit(".", 1)[0]] = data
            except Exception:
                continue
    return templates


@app.get("/api/configs/templates")
async def list_templates(_=Depends(require_login)):
    """List available training config templates."""
    templates = _load_templates()
    result = []
    for name, data in templates.items():
        result.append({
            "name": name,
            "description": data.get("description", ""),
            "editable_params": list(data.get("params", {}).keys()),
        })
    return result


@app.get("/api/configs/get")
async def get_template_config(template: str, _=Depends(require_login)):
    """Get the editable parameters of a template."""
    templates = _load_templates()
    if template not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    data = templates[template]
    return {
        "name": template,
        "description": data.get("description", ""),
        "command": data.get("command", ""),
        "params": data.get("params", {}),
    }


@app.post("/api/configs/update")
async def update_template_config(req: ConfigUpdateRequest, _=Depends(require_login)):
    """Update parameters in a template."""
    tpl_path = os.path.join(config.TEMPLATES_DIR, f"{req.template}.yaml")
    if not os.path.isfile(tpl_path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(tpl_path) as f:
        data = yaml.safe_load(f)
    if "params" not in data:
        data["params"] = {}
    # Only update known params
    allowed = set(data["params"].keys())
    for k, v in req.params.items():
        if k in allowed:
            data["params"][k] = v
    with open(tpl_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return {"ok": True, "params": data["params"]}


# ──────────────────────────────────────────────
# Job management (Subprocess training)
# ──────────────────────────────────────────────
_JOBS: dict[str, dict] = {}  # in-memory job registry


def _job_meta_path(job_id: str) -> str:
    return os.path.join(config.JOBS_DIR, f"{job_id}.json")


def _job_log_path(job_id: str) -> str:
    return os.path.join(config.JOBS_DIR, f"{job_id}.log")


def _save_job_meta(job_id: str, meta: dict):
    with open(_job_meta_path(job_id), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _load_all_jobs():
    """Load persisted job metadata from disk on startup."""
    if not os.path.exists(config.JOBS_DIR):
        return
    for fname in os.listdir(config.JOBS_DIR):
        if fname.endswith(".json"):
            job_id = fname[:-5]
            try:
                with open(os.path.join(config.JOBS_DIR, fname)) as f:
                    meta = json.load(f)
                _JOBS[job_id] = meta
            except Exception:
                continue


def _update_job_status(job_id: str):
    """Update a running job's status by polling the subprocess."""
    meta = _JOBS.get(job_id)
    if not meta or meta.get("status") != "running":
        return
    pid = meta.get("pid")
    if pid:
        try:
            proc = psutil.Process(pid)
            if proc.status() == psutil.STATUS_ZOMBIE or not proc.is_running():
                # Process finished
                ret = proc.wait(timeout=1)
                meta["status"] = "completed" if ret == 0 else "failed"
                meta["exit_code"] = ret
                meta["end_time"] = datetime.now(timezone.utc).isoformat()
                _save_job_meta(job_id, meta)
        except psutil.NoSuchProcess:
            meta["status"] = "completed"
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            _save_job_meta(job_id, meta)
        except Exception:
            pass


@app.post("/api/jobs/start")
async def start_job(req: JobStartRequest, _=Depends(require_login)):
    """Start a training job from a template."""
    templates = _load_templates()
    if req.template not in templates:
        raise HTTPException(status_code=404, detail="Template not found")

    tpl = templates[req.template]
    command = tpl.get("command", "")
    if not command:
        raise HTTPException(status_code=400, detail="Template has no command defined")

    # Merge params
    params = dict(tpl.get("params", {}))
    for k, v in req.params.items():
        if k in params:
            params[k] = v

    # Substitute params into command
    cmd = command
    for k, v in params.items():
        cmd = cmd.replace(f"{{{{{k}}}}}", str(v))

    job_id = str(uuid.uuid4())[:8]
    log_path = _job_log_path(job_id)

    # Start subprocess
    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.expanduser("~"),
            preexec_fn=os.setsid,
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to start job: {e}")

    meta = {
        "job_id": job_id,
        "template": req.template,
        "command": cmd,
        "params": params,
        "pid": proc.pid,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "exit_code": None,
    }
    _JOBS[job_id] = meta
    _save_job_meta(job_id, meta)

    return {"ok": True, "job_id": job_id, "status": "running"}


@app.get("/api/jobs/list")
async def list_jobs(_=Depends(require_login)):
    """List all jobs, sorted by start time descending."""
    # Update running jobs' status first
    for jid in list(_JOBS.keys()):
        _update_job_status(jid)
    jobs = sorted(_JOBS.values(), key=lambda j: j.get("start_time", ""), reverse=True)
    return jobs[:50]  # Limit to 50 most recent


@app.get("/api/jobs/status")
async def job_status(job_id: str, _=Depends(require_login)):
    """Get status of a specific job."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    _update_job_status(job_id)
    return _JOBS[job_id]


@app.get("/api/jobs/log")
async def job_log(job_id: str, offset: int = 0, lines: int = 200, _=Depends(require_login)):
    """Get log tail for a job."""
    log_path = _job_log_path(job_id)
    if not os.path.isfile(log_path):
        return {"job_id": job_id, "lines": [], "offset": 0, "total_size": 0}

    file_size = os.path.getsize(log_path)
    result_lines = []

    with open(log_path, "r", errors="replace") as f:
        if offset > 0:
            f.seek(min(offset, file_size))
            result_lines = f.readlines()[-lines:]
        else:
            # default: tail
            all_lines = f.readlines()
            result_lines = all_lines[-lines:]

    return {
        "job_id": job_id,
        "lines": result_lines,
        "offset": file_size,
        "total_size": file_size,
    }


@app.post("/api/jobs/stop")
async def stop_job(job_id: str, _=Depends(require_login)):
    """Stop a running job."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    meta = _JOBS[job_id]
    if meta.get("status") != "running":
        return {"ok": False, "detail": "Job is not running"}
    pid = meta.get("pid")
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(0.5)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
            meta["status"] = "stopped"
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            _save_job_meta(job_id, meta)
        except ProcessLookupError:
            meta["status"] = "completed"
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            _save_job_meta(job_id, meta)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "status": meta["status"]}


# ──────────────────────────────────────────────
# Training Pipeline (3-step: norm_stats → postprocess → train)
# Mirrors train_pipeline_gui.py logic; auto-postprocess without user confirm
# ──────────────────────────────────────────────

STEP_NAMES = ["compute_norm_stats", "postprocess_norm_stats", "train"]

_pipeline: Dict[str, any] = {
    "running": False,
    "current_step": "",
    "current_step_idx": -1,
    "steps": {name: {"status": "pending", "logs": ""} for name in STEP_NAMES},
    "config": {},
    "process": None,
    "pid": None,
    "pgid": None,
    "cancelled": False,
    "wandb_url": "",
}
_pipeline_lock = threading.Lock()


def _pipe_append_log(step: str, text: str):
    with _pipeline_lock:
        _pipeline["steps"][step]["logs"] += text


def _pipe_set_status(step: str, st: str):
    with _pipeline_lock:
        _pipeline["steps"][step]["status"] = st


def _parse_norm_stats_path_from_log(step_logs: str):
    m = re.search(r"Writing stats to:\s*(.+)", step_logs)
    if m:
        return Path(m.group(1).strip()) / "norm_stats.json"
    return None


def _norm_stats_path_fallback(config_name: str):
    openpi_root = config.OPENPI_ROOT
    assets_dir = openpi_root / "assets" / config_name
    candidates = list(assets_dir.rglob("norm_stats.json"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return assets_dir / "norm_stats.json"


def _pipe_run_subprocess(cmd, step, cwd, env, timeout=7200):
    _pipe_append_log(step, f"$ {' '.join(str(c) for c in cmd)}\n")
    _pipe_append_log(step, f"  cwd: {cwd}\n\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd, env=env, text=True, bufsize=1,
        preexec_fn=os.setsid,
    )
    _pipeline["process"] = proc
    _pipeline["pid"] = proc.pid
    try:
        _pipeline["pgid"] = os.getpgid(proc.pid)
    except OSError:
        _pipeline["pgid"] = proc.pid
    _pipe_append_log(step, f"  pid: {_pipeline['pid']}, pgid: {_pipeline['pgid']}\n")

    def _reader():
        for line in proc.stdout:
            _pipe_append_log(step, line)
            if step == "train" and "wandb" in line.lower() and "http" in line:
                m = re.search(r'(https?://\S+)', line)
                if m:
                    _pipeline["wandb_url"] = m.group(1)

    reader_t = threading.Thread(target=_reader, daemon=True)
    reader_t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _pipe_append_log(step, f"\n--- TIMEOUT after {timeout}s ---\n")
        _pipeline["process"] = None
        _pipeline["pid"] = None
        _pipeline["pgid"] = None
        return -1
    reader_t.join(timeout=5)
    _pipeline["process"] = None
    _pipeline["pid"] = None
    _pipeline["pgid"] = None
    return proc.returncode or 0


def _run_pipeline_thread(req: PipelineStartRequest):
    openpi_root = config.OPENPI_ROOT
    project_root = config.PROJECT_ROOT

    cuda_visible = ",".join(str(i) for i in req.gpu_indices) if req.gpu_indices else ""
    env = {
        **os.environ,
        "HF_ENDPOINT": "https://hf-mirror.com",
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
    }
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key

    try:
        # --- Step 1: compute_norm_stats ---
        step = "compute_norm_stats"
        _pipeline["current_step"] = step
        _pipeline["current_step_idx"] = 0

        if req.skip_norm_stats:
            _pipe_set_status(step, "skipped")
            _pipe_append_log(step, "--- Skipped by user ---\n")
        elif _pipeline["cancelled"]:
            _pipe_set_status(step, "cancelled")
        else:
            _pipe_set_status(step, "running")
            cmd = [
                "uv", "run",
                "scripts/compute_norm_stats.py",
                "--config-name", req.config_name,
            ]
            rc = _pipe_run_subprocess(cmd, step, str(openpi_root), env, timeout=3600)
            if _pipeline["cancelled"]:
                _pipe_set_status(step, "cancelled")
                return
            if rc != 0:
                _pipe_set_status(step, "failed")
                _pipe_append_log(step, f"\n--- Exit code: {rc} ---\n")
                _pipeline["running"] = False
                return
            _pipe_set_status(step, "completed")

        # --- Step 2: postprocess_norm_stats (AUTOMATIC, no user confirmation) ---
        step = "postprocess_norm_stats"
        _pipeline["current_step"] = step
        _pipeline["current_step_idx"] = 1

        if req.skip_postprocess:
            _pipe_set_status(step, "skipped")
            _pipe_append_log(step, "--- Skipped by user ---\n")
        elif _pipeline["cancelled"]:
            _pipe_set_status(step, "cancelled")
        else:
            _pipe_set_status(step, "running")
            step1_logs = _pipeline["steps"]["compute_norm_stats"]["logs"]
            parsed = _parse_norm_stats_path_from_log(step1_logs)
            if parsed and parsed.exists():
                ns_path = str(parsed)
                _pipe_append_log(step, f"Resolved norm_stats from step-1 log: {ns_path}\n")
            else:
                ns_path = str(_norm_stats_path_fallback(req.config_name))
                _pipe_append_log(step, f"Resolved norm_stats via fallback (newest file): {ns_path}\n")
            postprocess_script = str(project_root / "scripts" / "postprocess_norm_stats.py")

            # Dry-run first to check if clamping is needed
            dry_cmd = [
                sys.executable, postprocess_script,
                "--input", ns_path,
                "--output", ns_path,
                "--min-range", str(req.min_range),
                "--dry-run",
            ]
            _pipe_append_log(step, "--- Dry-run preview ---\n")
            rc = _pipe_run_subprocess(dry_cmd, step, str(project_root), env, timeout=60)

            if _pipeline["cancelled"]:
                _pipe_set_status(step, "cancelled")
                return

            dry_log = _pipeline["steps"][step]["logs"]
            if "No dimensions needed clamping" in dry_log:
                _pipe_append_log(step, "\n--- No changes needed, auto-skipping ---\n")
                _pipe_set_status(step, "completed")
            else:
                # Auto-apply without user confirmation
                _pipe_append_log(step, "\n--- Clamping needed, auto-applying (phone mode) ---\n")
                real_cmd = [
                    sys.executable, postprocess_script,
                    "--input", ns_path,
                    "--output", ns_path,
                    "--min-range", str(req.min_range),
                ]
                rc = _pipe_run_subprocess(real_cmd, step, str(project_root), env, timeout=60)
                if _pipeline["cancelled"]:
                    _pipe_set_status(step, "cancelled")
                    return
                if rc != 0:
                    _pipe_set_status(step, "failed")
                    _pipeline["running"] = False
                    return
                _pipe_set_status(step, "completed")

        # --- Step 3: train ---
        step = "train"
        _pipeline["current_step"] = step
        _pipeline["current_step_idx"] = 2

        if _pipeline["cancelled"]:
            _pipe_set_status(step, "cancelled")
            return

        _pipe_set_status(step, "running")
        cmd = [
            "uv", "run", "--active", "--no-sync",
            "scripts/train.py", req.config_name,
            f"--project-name=EvoRL-Piper",
            f"--exp-name={req.exp_name}",
            f"--batch-size={req.batch_size}",
            f"--fsdp-devices={req.fsdp_devices}",
            f"--num-train-steps={req.num_train_steps}",
            f"--save-interval={req.save_interval}",
        ]
        if req.wandb_enabled:
            cmd.append("--wandb-enabled")
        else:
            cmd.append("--no-wandb-enabled")
        if req.overwrite:
            cmd.append("--overwrite")
        if req.resume:
            cmd.append("--resume")

        rc = _pipe_run_subprocess(cmd, step, str(openpi_root), env, timeout=86400)
        if _pipeline["cancelled"]:
            _pipe_set_status(step, "cancelled")
            return
        if rc != 0:
            _pipe_set_status(step, "failed")
            _pipe_append_log(step, f"\n--- Exit code: {rc} ---\n")
        else:
            _pipe_set_status(step, "completed")
            _pipe_append_log(step, "\n--- Training completed successfully ---\n")

    except Exception as exc:
        step = _pipeline["current_step"] or "train"
        _pipe_append_log(step, f"\n--- PIPELINE ERROR: {exc} ---\n")
        _pipe_set_status(step, "failed")
    finally:
        _pipeline["running"] = False
        _pipeline["current_step"] = ""
        _pipeline["current_step_idx"] = -1


# Pipeline API endpoints


@app.get("/api/pipeline/defaults")
async def pipeline_defaults(_=Depends(require_login)):
    """Return default pipeline config & available config names."""
    return {
        "config_names": config.TRAIN_CONFIG_NAMES,
        "config_name": "pi05_aloha_wbcd_lora",
        "batch_size": 64,
        "fsdp_devices": 4,
        "num_train_steps": 20000,
        "save_interval": 1000,
        "min_range": 0.1,
        "openpi_root": str(config.OPENPI_ROOT),
    }


@app.get("/api/pipeline/datasets")
async def pipeline_list_datasets(_=Depends(require_login)):
    """Scan for LeRobot datasets available for training."""
    all_ds = []
    for root in config.EPISODE_SCAN_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for info_path in sorted(root_path.rglob("meta/info.json")):
            ds_root = info_path.parent.parent
            info = json.loads(info_path.read_text(encoding="utf-8"))
            tasks = []
            tasks_path = ds_root / "meta" / "tasks.jsonl"
            if tasks_path.exists():
                with tasks_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                tasks.append(json.loads(line).get("task", ""))
                            except Exception:
                                pass
            all_ds.append({
                "path": str(ds_root),
                "name": ds_root.name,
                "total_episodes": info.get("total_episodes", 0),
                "total_frames": info.get("total_frames", 0),
                "fps": info.get("fps", 0),
                "tasks": tasks,
            })
    return {"datasets": all_ds}


@app.get("/api/pipeline/gpu")
async def pipeline_gpu_status(_=Depends(require_login)):
    """Return per-GPU status with memory free info for selection."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_pct": int(parts[5]),
                    })
    except Exception:
        pass
    return {"gpus": gpus}


@app.post("/api/pipeline/start")
async def pipeline_start(req: PipelineStartRequest, _=Depends(require_login)):
    """Start the 3-step training pipeline."""
    if _pipeline["running"]:
        raise HTTPException(status_code=400, detail="Pipeline already running")

    for name in STEP_NAMES:
        _pipeline["steps"][name] = {"status": "pending", "logs": ""}
    _pipeline["running"] = True
    _pipeline["cancelled"] = False
    _pipeline["wandb_url"] = ""
    _pipeline["current_step"] = ""
    _pipeline["current_step_idx"] = -1
    _pipeline["config"] = req.dict()

    threading.Thread(target=_run_pipeline_thread, args=(req,), daemon=True).start()
    return {"ok": True}


@app.get("/api/pipeline/status")
async def pipeline_status(_=Depends(require_login)):
    """Return current pipeline execution status."""
    with _pipeline_lock:
        steps_out = {}
        for name in STEP_NAMES:
            s = _pipeline["steps"][name]
            steps_out[name] = {
                "status": s["status"],
                "logs": s["logs"][-20000:] if len(s["logs"]) > 20000 else s["logs"],
            }
        pgid = _pipeline.get("pgid")
        kill_cmd = f"kill -- -{pgid}" if pgid else ""
        return {
            "running": _pipeline["running"],
            "current_step": _pipeline["current_step"],
            "current_step_idx": _pipeline["current_step_idx"],
            "steps": steps_out,
            "wandb_url": _pipeline["wandb_url"],
            "cancelled": _pipeline["cancelled"],
            "config": _pipeline.get("config", {}),
            "pid": _pipeline.get("pid"),
            "pgid": pgid,
            "kill_cmd": kill_cmd,
        }


@app.post("/api/pipeline/cancel")
async def pipeline_cancel(_=Depends(require_login)):
    """Cancel the running pipeline."""
    _pipeline["cancelled"] = True
    proc = _pipeline.get("process")
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return {"ok": True}


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    _load_all_jobs()


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Use config for host/port
    print(f"🚀 WBCD Web Console starting on {config.HOST}:{config.PORT}")
    print(f"   Local test: http://127.0.0.1:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
