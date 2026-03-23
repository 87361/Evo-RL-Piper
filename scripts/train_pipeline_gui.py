#!/usr/bin/env python
"""Training pipeline GUI for pi05 / OpenPI models.

Three-step pipeline: compute_norm_stats -> postprocess_norm_stats -> train,
wrapped in a web UI with real-time GPU monitoring and log streaming.

Architecture mirrors review_episode_tasks_gui.py: FastAPI + embedded HTML.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def query_gpu_status() -> list[dict]:
    """Parse nvidia-smi CSV output into a list of GPU dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
                "utilization_pct": int(parts[5]),
            })
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dataset scanner
# ---------------------------------------------------------------------------

def scan_datasets(root: Path) -> list[dict]:
    """Find LeRobot v2.1 datasets under root (look for meta/info.json)."""
    datasets: list[dict] = []
    if not root.exists():
        return datasets
    for info_path in sorted(root.rglob("meta/info.json")):
        ds_root = info_path.parent.parent
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tasks: list[str] = []
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
        datasets.append({
            "path": str(ds_root),
            "name": ds_root.name,
            "total_episodes": info.get("total_episodes", 0),
            "total_frames": info.get("total_frames", 0),
            "fps": info.get("fps", 0),
            "tasks": tasks,
        })
    return datasets


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PipelineStartRequest(BaseModel):
    dataset_path: str
    exp_name: str
    config_name: str = "pi05_aloha_wbcd_lora"
    gpu_indices: list[int] = []
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

STEP_NAMES = ["compute_norm_stats", "postprocess_norm_stats", "train"]


def app_factory(
    openpi_root: Path,
    project_root: Path,
    datasets_scan_root: Path | None = None,
    default_dataset: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Training Pipeline")

    openpi_root = openpi_root.resolve()
    project_root = project_root.resolve()
    venv_activate = (project_root.parent / "openpi" / ".venv" / "bin" / "activate")

    pipeline: dict[str, Any] = {
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
        "waiting_confirm": False,
        "dry_run_output": "",
    }
    _lock = threading.Lock()

    def _append_log(step: str, text: str):
        with _lock:
            pipeline["steps"][step]["logs"] += text

    def _set_step_status(step: str, status: str):
        with _lock:
            pipeline["steps"][step]["status"] = status

    def _parse_norm_stats_path_from_log(step_logs: str) -> Path | None:
        """Parse the output directory from compute_norm_stats stdout.

        compute_norm_stats.py prints 'Writing stats to: <dir>' before saving.
        """
        m = re.search(r"Writing stats to:\s*(.+)", step_logs)
        if m:
            return Path(m.group(1).strip()) / "norm_stats.json"
        return None

    def _norm_stats_path_fallback(config_name: str) -> Path:
        """Fallback: pick the most-recently-modified norm_stats.json under the
        config assets dir.  Used when step 1 was skipped and no log is available.
        """
        assets_dir = openpi_root / "assets" / config_name
        candidates = list(assets_dir.rglob("norm_stats.json"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        return assets_dir / "norm_stats.json"

    # --- API endpoints ---

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/api/gpu")
    def gpu_status():
        return {"gpus": query_gpu_status()}

    @app.get("/api/datasets")
    def list_datasets():
        root = datasets_scan_root
        if root and root.exists():
            return {"datasets": scan_datasets(root)}
        return {"datasets": []}

    @app.get("/api/defaults")
    def get_defaults():
        return {
            "dataset_path": default_dataset or "",
            "config_name": "pi05_aloha_wbcd_lora",
            "batch_size": 64,
            "fsdp_devices": 4,
            "num_train_steps": 20000,
            "save_interval": 1000,
            "min_range": 0.1,
            "openpi_root": str(openpi_root),
        }

    @app.get("/api/pipeline/status")
    def pipeline_status():
        with _lock:
            steps_out = {}
            for name in STEP_NAMES:
                s = pipeline["steps"][name]
                steps_out[name] = {
                    "status": s["status"],
                    "logs": s["logs"][-20000:] if len(s["logs"]) > 20000 else s["logs"],
                }
            pgid = pipeline.get("pgid")
            kill_cmd = f"kill -- -{pgid}" if pgid else ""

            return {
                "running": pipeline["running"],
                "current_step": pipeline["current_step"],
                "current_step_idx": pipeline["current_step_idx"],
                "steps": steps_out,
                "wandb_url": pipeline["wandb_url"],
                "waiting_confirm": pipeline["waiting_confirm"],
                "dry_run_output": pipeline["dry_run_output"],
                "cancelled": pipeline["cancelled"],
                "pid": pipeline.get("pid"),
                "pgid": pgid,
                "kill_cmd": kill_cmd,
            }

    @app.post("/api/pipeline/start")
    def pipeline_start(req: PipelineStartRequest):
        if pipeline["running"]:
            return {"ok": False, "error": "Pipeline already running"}

        for name in STEP_NAMES:
            pipeline["steps"][name] = {"status": "pending", "logs": ""}
        pipeline["running"] = True
        pipeline["cancelled"] = False
        pipeline["wandb_url"] = ""
        pipeline["waiting_confirm"] = False
        pipeline["dry_run_output"] = ""
        pipeline["current_step"] = ""
        pipeline["current_step_idx"] = -1
        pipeline["config"] = req.dict()

        threading.Thread(target=_run_pipeline, args=(req,), daemon=True).start()
        return {"ok": True}

    @app.post("/api/pipeline/confirm-postprocess")
    def confirm_postprocess():
        if pipeline["waiting_confirm"]:
            pipeline["waiting_confirm"] = False
            return {"ok": True}
        return {"ok": False, "error": "Not waiting for confirmation"}

    @app.post("/api/pipeline/skip")
    def skip_step():
        step = pipeline["current_step"]
        if step and pipeline["waiting_confirm"]:
            pipeline["waiting_confirm"] = False
            _set_step_status(step, "skipped")
            pipeline["steps"][step]["logs"] += "\n--- Skipped by user ---\n"
            return {"ok": True, "skipped": step}
        return {"ok": False, "error": "Cannot skip current step"}

    @app.post("/api/pipeline/cancel")
    def cancel_pipeline():
        pipeline["cancelled"] = True
        pipeline["waiting_confirm"] = False
        proc = pipeline.get("process")
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"ok": True}

    # --- Pipeline runner ---

    def _run_subprocess(cmd: list[str], step: str, cwd: str, env: dict,
                        timeout: int = 7200) -> int:
        """Run a command, stream output to step logs. Returns exit code."""
        _append_log(step, f"$ {' '.join(cmd)}\n")
        _append_log(step, f"  cwd: {cwd}\n\n")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd, env=env, text=True, bufsize=1,
                preexec_fn=os.setsid,
            )
            pipeline["process"] = proc
            pipeline["pid"] = proc.pid
            try:
                pipeline["pgid"] = os.getpgid(proc.pid)
            except OSError:
                pipeline["pgid"] = proc.pid
            _append_log(step, f"  pid: {pipeline['pid']}, pgid: {pipeline['pgid']}\n")

            def _reader():
                assert proc.stdout is not None
                for line in proc.stdout:
                    _append_log(step, line)
                    if step == "train" and "wandb" in line.lower() and "http" in line:
                        m = re.search(r'(https?://\S+)', line)
                        if m:
                            pipeline["wandb_url"] = m.group(1)

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()
            proc.wait(timeout=timeout)
            reader_thread.join(timeout=5)
            pipeline["process"] = None
            pipeline["pid"] = None
            pipeline["pgid"] = None
            return proc.returncode or 0
        except subprocess.TimeoutExpired:
            proc.kill()
            pipeline["process"] = None
            pipeline["pid"] = None
            pipeline["pgid"] = None
            _append_log(step, f"\n--- TIMEOUT after {timeout}s ---\n")
            return -1
        except Exception as exc:
            _append_log(step, f"\n--- ERROR: {exc} ---\n")
            pipeline["process"] = None
            pipeline["pid"] = None
            pipeline["pgid"] = None
            return -1

    def _run_pipeline(req: PipelineStartRequest):
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
            pipeline["current_step"] = step
            pipeline["current_step_idx"] = 0

            if req.skip_norm_stats:
                _set_step_status(step, "skipped")
                _append_log(step, "--- Skipped by user ---\n")
            elif pipeline["cancelled"]:
                _set_step_status(step, "cancelled")
            else:
                _set_step_status(step, "running")
                venv_python = str(openpi_root / ".venv" / "bin" / "python")
                uv_bin = "uv"
                cmd = [
                    uv_bin, "run",
                    "scripts/compute_norm_stats.py",
                    "--config-name", req.config_name,
                ]
                rc = _run_subprocess(cmd, step, str(openpi_root), env, timeout=3600)
                if pipeline["cancelled"]:
                    _set_step_status(step, "cancelled")
                    return
                if rc != 0:
                    _set_step_status(step, "failed")
                    _append_log(step, f"\n--- Exit code: {rc} ---\n")
                    pipeline["running"] = False
                    return
                _set_step_status(step, "completed")

            # --- Step 2: postprocess_norm_stats ---
            step = "postprocess_norm_stats"
            pipeline["current_step"] = step
            pipeline["current_step_idx"] = 1

            if req.skip_postprocess:
                _set_step_status(step, "skipped")
                _append_log(step, "--- Skipped by user ---\n")
            elif pipeline["cancelled"]:
                _set_step_status(step, "cancelled")
            else:
                _set_step_status(step, "running")
                step1_logs = pipeline["steps"]["compute_norm_stats"]["logs"]
                parsed = _parse_norm_stats_path_from_log(step1_logs)
                if parsed and parsed.exists():
                    ns_path = str(parsed)
                    _append_log(step, f"Resolved norm_stats from step-1 log: {ns_path}\n")
                else:
                    ns_path = str(_norm_stats_path_fallback(req.config_name))
                    _append_log(step, f"Resolved norm_stats via fallback (newest file): {ns_path}\n")
                postprocess_script = str(project_root / "scripts" / "postprocess_norm_stats.py")

                dry_cmd = [
                    sys.executable, postprocess_script,
                    "--input", ns_path,
                    "--output", ns_path,
                    "--min-range", str(req.min_range),
                    "--dry-run",
                ]
                _append_log(step, "--- Dry-run preview ---\n")
                rc = _run_subprocess(dry_cmd, step, str(project_root), env, timeout=60)

                if pipeline["cancelled"]:
                    _set_step_status(step, "cancelled")
                    return

                dry_log = pipeline["steps"][step]["logs"]
                if "No dimensions needed clamping" in dry_log:
                    _append_log(step, "\n--- No changes needed, auto-skipping ---\n")
                    _set_step_status(step, "completed")
                else:
                    pipeline["dry_run_output"] = dry_log
                    pipeline["waiting_confirm"] = True
                    _append_log(step, "\n--- Waiting for user confirmation... ---\n")

                    while pipeline["waiting_confirm"] and not pipeline["cancelled"]:
                        time.sleep(0.5)

                    if pipeline["cancelled"]:
                        _set_step_status(step, "cancelled")
                        return

                    if pipeline["steps"][step]["status"] == "skipped":
                        pass
                    else:
                        _append_log(step, "\n--- Executing postprocess ---\n")
                        real_cmd = [
                            sys.executable, postprocess_script,
                            "--input", ns_path,
                            "--output", ns_path,
                            "--min-range", str(req.min_range),
                        ]
                        rc = _run_subprocess(real_cmd, step, str(project_root), env, timeout=60)
                        if pipeline["cancelled"]:
                            _set_step_status(step, "cancelled")
                            return
                        if rc != 0:
                            _set_step_status(step, "failed")
                            pipeline["running"] = False
                            return
                        _set_step_status(step, "completed")

            # --- Step 3: train ---
            step = "train"
            pipeline["current_step"] = step
            pipeline["current_step_idx"] = 2

            if pipeline["cancelled"]:
                _set_step_status(step, "cancelled")
                return

            _set_step_status(step, "running")
            uv_bin = "uv"
            cmd = [
                uv_bin, "run", "--active", "--no-sync",
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

            rc = _run_subprocess(cmd, step, str(openpi_root), env, timeout=86400)
            if pipeline["cancelled"]:
                _set_step_status(step, "cancelled")
                return
            if rc != 0:
                _set_step_status(step, "failed")
                _append_log(step, f"\n--- Exit code: {rc} ---\n")
            else:
                _set_step_status(step, "completed")
                _append_log(step, "\n--- Training completed successfully ---\n")

        except Exception as exc:
            step = pipeline["current_step"] or "train"
            _append_log(step, f"\n--- PIPELINE ERROR: {exc} ---\n")
            _set_step_status(step, "failed")
        finally:
            pipeline["running"] = False
            pipeline["current_step"] = ""
            pipeline["current_step_idx"] = -1

    return app


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/>
<title>Training Pipeline</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f17;color:#e0e0e0;font-family:'Segoe UI','PingFang SC',Arial,sans-serif;font-size:14px;min-height:100vh}

.header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:16px 24px;border-bottom:1px solid #333;display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:20px;font-weight:600;color:#e0e0e0}
.header .subtitle{font-size:12px;color:#888;margin-top:2px}

.container{max-width:1400px;margin:0 auto;padding:16px 24px}

/* GPU panel */
.section{margin-bottom:20px}
.section-title{font-size:15px;font-weight:600;color:#ccc;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.section-title .refresh-hint{font-size:11px;color:#666;font-weight:400}

.gpu-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.gpu-card{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;cursor:pointer;transition:all .15s;position:relative}
.gpu-card:hover{border-color:#4fc3f7}
.gpu-card.selected{border-color:#4fc3f7;background:#1a2a3e}
.gpu-card.selected::after{content:'✓';position:absolute;top:8px;right:10px;color:#4fc3f7;font-size:16px;font-weight:700}
.gpu-card .gpu-idx{font-size:18px;font-weight:700;color:#4fc3f7}
.gpu-card .gpu-name{font-size:11px;color:#999;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gpu-card .gpu-mem{margin-top:8px;font-size:13px}
.gpu-card .mem-bar{height:6px;background:#333;border-radius:3px;margin-top:4px;overflow:hidden}
.gpu-card .mem-bar-fill{height:100%;border-radius:3px;transition:width .3s}
.mem-bar-fill.green{background:#4caf50}
.mem-bar-fill.yellow{background:#ff9800}
.mem-bar-fill.red{background:#f44336}
.gpu-card .gpu-util{font-size:11px;color:#888;margin-top:4px}
.gpu-card .gpu-free{font-size:13px;font-weight:600;margin-top:4px}
.gpu-free.green{color:#4caf50}
.gpu-free.yellow{color:#ff9800}
.gpu-free.red{color:#f44336}

.selected-gpus{margin-top:8px;font-size:13px;color:#aaa}
.selected-gpus span{color:#4fc3f7;font-weight:600}

/* Config form */
.config-section{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:20px}
.form-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.form-row label{min-width:120px;font-size:13px;color:#aaa;text-align:right;flex-shrink:0}
.form-row input[type=text],.form-row input[type=number],.form-row select{background:#222;color:#eee;border:1px solid #444;padding:7px 10px;border-radius:4px;font-size:13px;flex:1;max-width:500px}
.form-row input:focus,.form-row select:focus{outline:none;border-color:#4fc3f7}
.form-row .hint{font-size:11px;color:#666;margin-left:4px}

.toggle-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.toggle-label{font-size:13px;color:#aaa;min-width:120px;text-align:right}
.toggle{position:relative;width:40px;height:22px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:#444;border-radius:11px;transition:.2s}
.toggle .slider::before{content:'';position:absolute;width:16px;height:16px;left:3px;bottom:3px;background:#888;border-radius:50%;transition:.2s}
.toggle input:checked+.slider{background:#4fc3f7}
.toggle input:checked+.slider::before{transform:translateX(18px);background:#fff}

.advanced-toggle{cursor:pointer;color:#4fc3f7;font-size:13px;margin-bottom:12px;display:inline-flex;align-items:center;gap:4px;user-select:none}
.advanced-toggle:hover{color:#81d4fa}
.advanced-panel{display:none;border-top:1px solid #333;padding-top:12px;margin-top:4px}
.advanced-panel.open{display:block}

.btn-group{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}
button{background:#333;color:#eee;border:1px solid #555;padding:8px 16px;border-radius:4px;cursor:pointer;font-size:13px;transition:all .15s}
button:hover{background:#444}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-primary{background:#1976d2;border-color:#1976d2;color:#fff;font-size:15px;padding:10px 32px;font-weight:600}
.btn-primary:hover:not(:disabled){background:#1565c0}
.btn-danger{background:#c62828;border-color:#c62828;color:#fff}
.btn-danger:hover:not(:disabled){background:#b71c1c}
.btn-warn{background:#e65100;border-color:#e65100;color:#fff}
.btn-warn:hover:not(:disabled){background:#bf360c}

/* Pipeline steps */
.pipeline-section{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px}
.step-header{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.step-num{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;border:2px solid #555;color:#888;flex-shrink:0}
.step-num.active{border-color:#4fc3f7;color:#4fc3f7;animation:pulse 1.5s infinite}
.step-num.done{border-color:#4caf50;color:#4caf50;background:rgba(76,175,80,.1)}
.step-num.fail{border-color:#f44336;color:#f44336}
.step-num.skip{border-color:#ff9800;color:#ff9800}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.step-title{font-size:14px;font-weight:600}
.step-status{font-size:12px;padding:2px 8px;border-radius:10px;margin-left:auto}
.step-status.pending{background:#333;color:#888}
.step-status.running{background:rgba(79,195,247,.15);color:#4fc3f7}
.step-status.completed{background:rgba(76,175,80,.15);color:#4caf50}
.step-status.failed{background:rgba(244,67,54,.15);color:#f44336}
.step-status.skipped{background:rgba(255,152,0,.15);color:#ff9800}
.step-status.cancelled{background:rgba(158,158,158,.15);color:#9e9e9e}

.step-log{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px;margin:8px 0 16px 38px;font-family:'Fira Code','Consolas',monospace;font-size:12px;line-height:1.6;max-height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#bbb}
.step-log.expanded{max-height:600px}

.confirm-bar{background:rgba(255,152,0,.1);border:1px solid #e65100;border-radius:4px;padding:10px 14px;margin:8px 0 16px 38px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.confirm-bar .msg{font-size:13px;color:#ff9800;flex:1}

.wandb-link{margin-top:8px;margin-left:38px}
.wandb-link a{color:#4fc3f7;text-decoration:none;font-size:13px}
.wandb-link a:hover{text-decoration:underline}

.kill-cmd-bar{background:rgba(244,67,54,.08);border:1px solid #c62828;border-radius:6px;padding:10px 14px;margin:12px 0 0 0;display:flex;align-items:center;gap:10px}
.kill-label{font-size:13px;color:#f44336;font-weight:600;white-space:nowrap}
.kill-code{background:#1a1a2e;color:#ff8a80;padding:6px 12px;border-radius:4px;font-family:'Fira Code','Consolas',monospace;font-size:13px;flex:1;user-select:all;word-break:break-all}
.btn-copy{background:transparent;border:1px solid #555;padding:4px 8px;font-size:14px;border-radius:4px;cursor:pointer;color:#ccc;flex-shrink:0}
.btn-copy:hover{background:#333;border-color:#888}

.dataset-dropdown{position:relative;display:inline-block;flex:1;max-width:500px}
.dataset-dropdown .dd-list{position:absolute;top:100%;left:0;right:0;background:#222;border:1px solid #555;border-radius:0 0 4px 4px;max-height:250px;overflow-y:auto;z-index:50;display:none}
.dataset-dropdown.open .dd-list{display:block}
.dd-item{padding:8px 10px;cursor:pointer;font-size:12px;border-bottom:1px solid #333}
.dd-item:hover{background:#333}
.dd-item .dd-name{color:#eee;font-weight:500}
.dd-item .dd-meta{color:#888;font-size:11px;margin-top:2px}
</style></head><body>
<div class="header">
  <div>
    <h1>Training Pipeline</h1>
    <div class="subtitle">compute_norm_stats → postprocess → train</div>
  </div>
  <div id="clock" style="font-size:12px;color:#666"></div>
</div>

<div class="container">

  <!-- GPU Panel -->
  <div class="section" id="gpu-section">
    <div class="section-title">GPU Status <span class="refresh-hint" id="gpu-timer">10s auto-refresh</span></div>
    <div class="gpu-grid" id="gpu-grid"></div>
    <div class="selected-gpus">Selected: <span id="selected-gpu-text">none</span> | CUDA_VISIBLE_DEVICES=<span id="cuda-vis-text"></span> | fsdp_devices=<span id="fsdp-text"></span></div>
  </div>

  <!-- Config -->
  <div class="config-section" id="config-section">
    <div class="section-title" style="margin-bottom:14px">Training Configuration</div>

    <div class="form-row">
      <label>Dataset</label>
      <div class="dataset-dropdown" id="ds-dropdown">
        <input type="text" id="dataset-path" placeholder="/path/to/dataset" autocomplete="off"/>
        <div class="dd-list" id="ds-list"></div>
      </div>
    </div>

    <div class="form-row">
      <label>Exp Name</label>
      <input type="text" id="exp-name" placeholder="evorl_pi05_lora_xxx_260323"/>
      <button onclick="autoExpName()">Auto</button>
    </div>

    <div class="advanced-toggle" onclick="toggleAdvanced()">
      <span id="adv-arrow">▶</span> Advanced Parameters
    </div>
    <div class="advanced-panel" id="advanced-panel">
      <div class="form-row">
        <label>config_name</label>
        <select id="config-name">
          <option value="pi05_aloha_wbcd_lora" selected>pi05_aloha_wbcd_lora</option>
          <option value="pi0_aloha_pen_uncap">pi0_aloha_pen_uncap</option>
          <option value="pi05_aloha_pen_uncap">pi05_aloha_pen_uncap</option>
        </select>
      </div>
      <div class="form-row">
        <label>batch_size</label>
        <input type="number" id="batch-size" value="64" min="1"/>
      </div>
      <div class="form-row">
        <label>fsdp_devices</label>
        <input type="number" id="fsdp-devices" value="4" min="1"/>
      </div>
      <div class="form-row">
        <label>num_train_steps</label>
        <input type="number" id="num-train-steps" value="20000" min="1" step="1000"/>
      </div>
      <div class="form-row">
        <label>save_interval</label>
        <input type="number" id="save-interval" value="1000" min="100" step="100"/>
      </div>
      <div class="form-row">
        <label>min_range</label>
        <input type="number" id="min-range" value="0.1" min="0.01" step="0.01"/>
        <span class="hint">norm_stats clamp threshold</span>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">wandb</span>
        <label class="toggle"><input type="checkbox" id="wandb-enabled" checked/><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">overwrite</span>
        <label class="toggle"><input type="checkbox" id="overwrite-flag" checked/><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">resume</span>
        <label class="toggle"><input type="checkbox" id="resume-flag"/><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">skip norm_stats</span>
        <label class="toggle"><input type="checkbox" id="skip-norm"/><span class="slider"></span></label>
        <span class="hint">if already computed</span>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">skip postprocess</span>
        <label class="toggle"><input type="checkbox" id="skip-postprocess"/><span class="slider"></span></label>
      </div>
    </div>

    <div class="btn-group">
      <button class="btn-primary" id="btn-start" onclick="startPipeline()">Start Training Pipeline</button>
      <button class="btn-danger" id="btn-cancel" onclick="cancelPipeline()" disabled>Cancel</button>
    </div>
  </div>

  <!-- Pipeline steps -->
  <div class="pipeline-section" id="pipeline-section">
    <div class="section-title" style="margin-bottom:14px">Pipeline Execution</div>

    <div id="step-compute_norm_stats">
      <div class="step-header">
        <div class="step-num" id="sn-0">1</div>
        <div class="step-title">Compute norm_stats</div>
        <div class="step-status pending" id="ss-compute_norm_stats">pending</div>
      </div>
      <div class="step-log" id="log-compute_norm_stats"></div>
    </div>

    <div id="step-postprocess_norm_stats">
      <div class="step-header">
        <div class="step-num" id="sn-1">2</div>
        <div class="step-title">Postprocess norm_stats</div>
        <div class="step-status pending" id="ss-postprocess_norm_stats">pending</div>
      </div>
      <div class="step-log" id="log-postprocess_norm_stats"></div>
      <div class="confirm-bar" id="confirm-bar" style="display:none">
        <div class="msg">Norm stats need clamping. Review the dry-run output above, then confirm or skip.</div>
        <button class="btn-primary" onclick="confirmPostprocess()" style="padding:6px 18px;font-size:13px">Confirm &amp; Apply</button>
        <button class="btn-warn" onclick="skipStep()" style="padding:6px 18px;font-size:13px">Skip</button>
      </div>
    </div>

    <div id="step-train">
      <div class="step-header">
        <div class="step-num" id="sn-2">3</div>
        <div class="step-title">Train</div>
        <div class="step-status pending" id="ss-train">pending</div>
      </div>
      <div class="step-log" id="log-train"></div>
      <div class="wandb-link" id="wandb-link" style="display:none">
        WandB: <a id="wandb-url" href="#" target="_blank"></a>
      </div>
    </div>

    <div class="kill-cmd-bar" id="kill-cmd-bar" style="display:none">
      <span class="kill-label">终止命令:</span>
      <code class="kill-code" id="kill-cmd-text"></code>
      <button class="btn-copy" onclick="copyKillCmd()" id="btn-copy-kill" title="Copy">📋</button>
    </div>
  </div>
</div>

<script>
const STEPS = ['compute_norm_stats','postprocess_norm_stats','train'];
let selectedGPUs = new Set();
let pipelineRunning = false;
let gpuData = [];
let datasets = [];

// --- GPU ---
async function fetchGPU(){
  try{
    const r = await fetch('/api/gpu');
    const d = await r.json();
    gpuData = d.gpus || [];
    renderGPU();
  }catch(e){}
}

function renderGPU(){
  const grid = document.getElementById('gpu-grid');
  grid.innerHTML = '';
  gpuData.forEach(g => {
    const pct = g.memory_total_mb > 0 ? (g.memory_used_mb / g.memory_total_mb * 100) : 0;
    const freeMB = g.memory_free_mb;
    const freeGB = (freeMB/1024).toFixed(1);
    let color = 'green';
    if(freeMB < 10240) color = 'red';
    else if(freeMB < 30720) color = 'yellow';

    const card = document.createElement('div');
    card.className = 'gpu-card' + (selectedGPUs.has(g.index) ? ' selected' : '');
    card.onclick = () => toggleGPU(g.index);
    card.innerHTML = `
      <div class="gpu-idx">GPU ${g.index}</div>
      <div class="gpu-name">${g.name}</div>
      <div class="gpu-mem">${(g.memory_used_mb/1024).toFixed(1)} / ${(g.memory_total_mb/1024).toFixed(1)} GB</div>
      <div class="mem-bar"><div class="mem-bar-fill ${color}" style="width:${pct}%"></div></div>
      <div class="gpu-free ${color}">Free: ${freeGB} GB</div>
      <div class="gpu-util">Utilization: ${g.utilization_pct}%</div>
    `;
    grid.appendChild(card);
  });
  updateSelectedText();
}

function toggleGPU(idx){
  if(pipelineRunning) return;
  if(selectedGPUs.has(idx)) selectedGPUs.delete(idx);
  else selectedGPUs.add(idx);
  renderGPU();
}

function updateSelectedText(){
  const arr = [...selectedGPUs].sort((a,b)=>a-b);
  document.getElementById('selected-gpu-text').textContent = arr.length ? arr.map(i=>`GPU ${i}`).join(', ') : 'none';
  document.getElementById('cuda-vis-text').textContent = arr.join(',');
  document.getElementById('fsdp-text').textContent = arr.length || 1;
  if(!pipelineRunning) document.getElementById('fsdp-devices').value = arr.length || 1;
}

function autoSelectFreeGPUs(){
  selectedGPUs.clear();
  gpuData.forEach(g => {
    if(g.memory_free_mb > 30000) selectedGPUs.add(g.index);
  });
  renderGPU();
}

// --- Datasets ---
async function fetchDatasets(){
  try{
    const r = await fetch('/api/datasets');
    const d = await r.json();
    datasets = d.datasets || [];
    renderDatasetDropdown();
  }catch(e){}
}

function renderDatasetDropdown(){
  const list = document.getElementById('ds-list');
  list.innerHTML = '';
  datasets.forEach(ds => {
    const item = document.createElement('div');
    item.className = 'dd-item';
    item.innerHTML = `<div class="dd-name">${ds.name}</div><div class="dd-meta">${ds.total_episodes} eps, ${ds.total_frames} frames | ${ds.path}</div>`;
    item.onclick = (e) => {
      e.stopPropagation();
      document.getElementById('dataset-path').value = ds.path;
      document.getElementById('ds-dropdown').classList.remove('open');
      autoExpName();
    };
    list.appendChild(item);
  });
}

document.addEventListener('click', e => {
  if(!e.target.closest('#ds-dropdown')) document.getElementById('ds-dropdown').classList.remove('open');
});
document.getElementById('dataset-path').addEventListener('focus', () => {
  if(datasets.length) document.getElementById('ds-dropdown').classList.add('open');
});

// --- Config helpers ---
function toggleAdvanced(){
  const panel = document.getElementById('advanced-panel');
  const arrow = document.getElementById('adv-arrow');
  panel.classList.toggle('open');
  arrow.textContent = panel.classList.contains('open') ? '▼' : '▶';
}

function autoExpName(){
  const dsPath = document.getElementById('dataset-path').value.trim();
  if(!dsPath) return;
  const parts = dsPath.replace(/\/+$/,'').split('/');
  const shortName = parts[parts.length-1].replace(/[^a-zA-Z0-9_]/g,'_').substring(0,30);
  const d = new Date();
  const dateStr = String(d.getFullYear()).slice(2) + String(d.getMonth()+1).padStart(2,'0') + String(d.getDate()).padStart(2,'0');
  document.getElementById('exp-name').value = `evorl_pi05_lora_${shortName}_${dateStr}`;
}

// --- Pipeline control ---
async function startPipeline(){
  const dsPath = document.getElementById('dataset-path').value.trim();
  const expName = document.getElementById('exp-name').value.trim();
  if(!dsPath){alert('Please specify dataset path'); return;}
  if(!expName){alert('Please specify experiment name'); return;}
  if(selectedGPUs.size === 0){alert('Please select at least one GPU'); return;}

  const body = {
    dataset_path: dsPath,
    exp_name: expName,
    config_name: document.getElementById('config-name').value,
    gpu_indices: [...selectedGPUs].sort((a,b)=>a-b),
    batch_size: parseInt(document.getElementById('batch-size').value) || 64,
    fsdp_devices: parseInt(document.getElementById('fsdp-devices').value) || selectedGPUs.size,
    num_train_steps: parseInt(document.getElementById('num-train-steps').value) || 20000,
    save_interval: parseInt(document.getElementById('save-interval').value) || 1000,
    min_range: parseFloat(document.getElementById('min-range').value) || 0.1,
    resume: document.getElementById('resume-flag').checked,
    overwrite: document.getElementById('overwrite-flag').checked,
    wandb_enabled: document.getElementById('wandb-enabled').checked,
    skip_norm_stats: document.getElementById('skip-norm').checked,
    skip_postprocess: document.getElementById('skip-postprocess').checked,
  };

  try{
    const r = await fetch('/api/pipeline/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if(!d.ok){alert(d.error || 'Failed to start'); return;}
    pipelineRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-cancel').disabled = false;
  }catch(e){alert('Network error');}
}

async function cancelPipeline(){
  if(!confirm('Cancel the running pipeline?')) return;
  try{await fetch('/api/pipeline/cancel', {method:'POST'});}catch(e){}
}

async function confirmPostprocess(){
  try{await fetch('/api/pipeline/confirm-postprocess', {method:'POST'});}catch(e){}
}

async function skipStep(){
  try{await fetch('/api/pipeline/skip', {method:'POST'});}catch(e){}
}

// --- Status polling ---
async function pollStatus(){
  try{
    const r = await fetch('/api/pipeline/status');
    const d = await r.json();
    pipelineRunning = d.running;
    document.getElementById('btn-start').disabled = d.running;
    document.getElementById('btn-cancel').disabled = !d.running;

    STEPS.forEach((name, i) => {
      const st = d.steps[name];
      if(!st) return;
      const badge = document.getElementById('ss-'+name);
      badge.className = 'step-status ' + st.status;
      badge.textContent = st.status;

      const numEl = document.getElementById('sn-'+i);
      numEl.className = 'step-num';
      if(st.status === 'running') numEl.classList.add('active');
      else if(st.status === 'completed') numEl.classList.add('done');
      else if(st.status === 'failed') numEl.classList.add('fail');
      else if(st.status === 'skipped' || st.status === 'cancelled') numEl.classList.add('skip');

      const logEl = document.getElementById('log-'+name);
      if(st.logs !== logEl._lastLogs){
        logEl.textContent = st.logs;
        logEl._lastLogs = st.logs;
        if(st.status === 'running') logEl.scrollTop = logEl.scrollHeight;
      }
    });

    const confirmBar = document.getElementById('confirm-bar');
    confirmBar.style.display = d.waiting_confirm ? 'flex' : 'none';

    const wandbDiv = document.getElementById('wandb-link');
    if(d.wandb_url){
      wandbDiv.style.display = 'block';
      const a = document.getElementById('wandb-url');
      a.href = d.wandb_url;
      a.textContent = d.wandb_url;
    }else{
      wandbDiv.style.display = 'none';
    }

    const killBar = document.getElementById('kill-cmd-bar');
    if(d.running && d.kill_cmd){
      killBar.style.display = 'flex';
      document.getElementById('kill-cmd-text').textContent = d.kill_cmd;
    }else{
      killBar.style.display = 'none';
    }
  }catch(e){}
}

function copyKillCmd(){
  const text = document.getElementById('kill-cmd-text').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('btn-copy-kill');
    btn.textContent = '✓';
    setTimeout(() => { btn.textContent = '📋'; }, 1500);
  });
}

// --- Clock ---
function updateClock(){
  document.getElementById('clock').textContent = new Date().toLocaleString('zh-CN');
}

// --- Init ---
async function init(){
  const r = await fetch('/api/defaults');
  const d = await r.json();
  if(d.dataset_path) document.getElementById('dataset-path').value = d.dataset_path;

  await fetchGPU();
  autoSelectFreeGPUs();
  await fetchDatasets();
  if(document.getElementById('dataset-path').value) autoExpName();
}

init();
updateClock();
setInterval(updateClock, 1000);
setInterval(fetchGPU, 10000);
setInterval(pollStatus, 2000);
</script></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training pipeline GUI")
    p.add_argument("--dataset", type=str, default=None,
                   help="Pre-fill dataset path")
    p.add_argument("--datasets-scan-root", type=str,
                   default="/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD",
                   help="Root to scan for available datasets")
    p.add_argument("--openpi-root", type=str, default=None,
                   help="OpenPI code root (default: <project>/third_party/openpi)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18090)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent.parent.resolve()
    openpi_root = Path(args.openpi_root) if args.openpi_root else project_root / "third_party" / "openpi"

    if not openpi_root.exists():
        print(f"ERROR: openpi root not found: {openpi_root}", file=sys.stderr)
        sys.exit(1)

    scan_root = Path(args.datasets_scan_root) if args.datasets_scan_root else None

    app = app_factory(
        openpi_root=openpi_root,
        project_root=project_root,
        datasets_scan_root=scan_root,
        default_dataset=args.dataset,
    )
    print(f"Starting training pipeline GUI at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
