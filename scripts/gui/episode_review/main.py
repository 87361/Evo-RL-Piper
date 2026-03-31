#!/usr/bin/env python
"""Headless-friendly web GUI for episode task review (multi-category, grid view)."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from app import app_factory
from data_ops import scan_lerobot_datasets


LAUNCHER_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Episode Review 数据集选择</title>
  <style>
    body { background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:20px; }
    .box { max-width:1100px; margin:0 auto; }
    h1 { margin:0 0 12px 0; font-size:24px; }
    .small { color:#bbb; margin-bottom:14px; display: flex; align-items: center; gap: 10px; }
    table { width:100%; border-collapse:collapse; background:#1b1b1b; }
    th, td { border:1px solid #333; padding:8px 10px; text-align:left; font-size:13px; }
    th { background:#222; }
    button { background:#1565c0; color:#fff; border:none; border-radius:4px; padding:6px 10px; cursor:pointer; }
    button:hover { background:#1976d2; }
    button.refresh-btn { background:#2e7d32; }
    button.refresh-btn:hover { background:#388e3c; }
    .warn { color:#ffb74d; margin-top:10px; }
  </style>
</head>
<body>
  <div class="box">
    <h1>Episode Review 入口</h1>
    <div class="small">
      <span id="meta"></span>
      <button class="refresh-btn" onclick="refresh()" id="refreshBtn">刷新扫描</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>名称</th><th>路径</th><th>Episodes(meta)</th><th>可预览视频数</th><th>FPS</th><th>操作</th>
        </tr>
      </thead>
      <tbody id="body"></tbody>
    </table>
    <div class="warn" id="warn"></div>
  </div>
  <script>
    async function refresh(){
      const btn = document.getElementById('refreshBtn');
      btn.innerText = '刷新中...';
      btn.disabled = true;
      try {
        const res = await fetch('api/datasets/refresh', {method: 'POST'});
        const d = await res.json();
        render(d);
      } catch (err) {
        alert('刷新失败: ' + err);
      } finally {
        btn.innerText = '刷新扫描';
        btn.disabled = false;
      }
    }
    async function load(){
      const res = await fetch('api/datasets');
      const d = await res.json();
      render(d);
    }
    function render(d){
      document.getElementById('meta').innerText = '扫描根目录: ' + d.scan_root;
      const body = document.getElementById('body');
      body.innerHTML = '';
      if(!d.datasets.length){
        document.getElementById('warn').innerText = '未发现可用数据集，请通过 --datasets-scan-root 指定正确目录。';
        return;
      }
      document.getElementById('warn').innerText = '';
      for(const ds of d.datasets){
        const tr = document.createElement('tr');
        const openBtn = ds.preview_episode_count > 0
          ? `<button onclick="location.href='${ds.entry_url}'">打开</button>`
          : `<button disabled title="没有 episode_*.mp4 可预览">无可预览视频</button>`;
        tr.innerHTML = `
          <td>${ds.name}</td>
          <td>${ds.dataset_root}</td>
          <td>${ds.total_episodes}</td>
          <td>${ds.preview_episode_count}</td>
          <td>${ds.fps}</td>
          <td>${openBtn}</td>
        `;
        body.appendChild(tr);
      }
    }
    load();
  </script>
</body>
</html>
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode task review GUI")
    p.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root dir containing episode_*.mp4. Omit to use launcher mode.",
    )
    p.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help="CSV output path for labels. Omit to use launcher mode.",
    )
    p.add_argument(
        "--categories-json", type=Path, default=None,
        help="Shared categories JSON path (defaults to <label-csv-stem>_categories.json next to label-csv). "
             "Use the same path across multiple datasets to share categories.",
    )
    p.add_argument(
        "--merge-sources", nargs="*", default=[],
        help="Additional datasets for cross-dataset merge, format: name::dataset_root::label_csv. "
             "Can specify multiple.",
    )
    p.add_argument(
        "--datasets-scan-root", type=Path, default=None,
        help="Root directory to scan for discoverable LeRobot datasets. "
             "Defaults to two levels up from video-root (e.g. the WBCD parent dir).",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18080)
    return p.parse_args()


def _parse_merge_source(raw: str) -> dict:
    parts = raw.split("::")
    if len(parts) < 2:
        raise ValueError(f"--merge-sources format: name::dataset_root[::label_csv], got: {raw}")
    name = parts[0].strip()
    ds_root = parts[1].strip()
    label_csv = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else str(Path(ds_root) / "task_labels.csv")
    return {"name": name, "dataset_root": ds_root, "label_csv": label_csv}


def _resolve_scan_root(args: argparse.Namespace) -> Path:
    if args.datasets_scan_root is not None:
        return args.datasets_scan_root.resolve()
    for candidate in [
        Path("/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD"),
        Path.home() / "datasets",
        Path.cwd(),
    ]:
        if candidate.exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def _build_launcher_app(
    scan_root: Path,
    categories_json: Path | None,
    merge_sources: list[dict] | None,
) -> FastAPI:
    launcher = FastAPI(title="Episode Review Launcher")
    datasets = scan_lerobot_datasets(scan_root)
    mounted: list[dict] = []
    
    app_state = {"next_idx": 0}

    def _mount_new_dataset(ds: dict) -> bool:
        ds_root = Path(ds["dataset_root"]).resolve()
        
        # Check if already mounted
        if any(m["dataset_root"] == str(ds_root) for m in mounted):
            return False

        video_root = Path(ds["video_root"]).resolve() if ds.get("video_root") else (ds_root / "videos").resolve()
        if not video_root.exists():
            return False
            
        label_csv = Path(ds["label_csv"]).resolve() if ds.get("label_csv") else (ds_root / "task_labels.csv").resolve()
        preview_episode_count = sum(1 for _ in video_root.rglob("episode_*.mp4"))
        
        idx = app_state["next_idx"]
        app_state["next_idx"] += 1
        mount_path = f"/review/{idx}"
        
        review_app = app_factory(
            video_root=video_root,
            label_csv=label_csv,
            categories_json_override=categories_json,
            merge_sources=merge_sources,
            datasets_scan_root=scan_root,
        )
        launcher.mount(mount_path, review_app, name=f"review_{idx}")
        mounted.append(
            {
                "name": ds.get("name", ds_root.name),
                "dataset_root": str(ds_root),
                "total_episodes": int(ds.get("total_episodes", 0)),
                "preview_episode_count": int(preview_episode_count),
                "fps": int(ds.get("fps", 0)),
                "entry_url": f"review/{idx}/",
            }
        )
        return True

    for ds in datasets:
        _mount_new_dataset(ds)

    @launcher.get("/", response_class=HTMLResponse)
    def home() -> str:
        return LAUNCHER_PAGE

    @launcher.get("/api/datasets")
    def list_datasets() -> dict:
        return {"scan_root": str(scan_root), "datasets": mounted}

    @launcher.post("/api/datasets/refresh")
    def refresh_datasets() -> dict:
        latest_datasets = scan_lerobot_datasets(scan_root)
        for ds in latest_datasets:
            _mount_new_dataset(ds)
        return {"scan_root": str(scan_root), "datasets": mounted}

    return launcher


def main() -> None:
    args = parse_args()
    cat_json = args.categories_json.resolve() if args.categories_json else None
    merge_sources = [_parse_merge_source(s) for s in args.merge_sources] if args.merge_sources else None
    scan_root = _resolve_scan_root(args)

    if (args.video_root is None) != (args.label_csv is None):
        raise ValueError("--video-root 和 --label-csv 要么同时提供，要么都不提供")

    if args.video_root is not None and args.label_csv is not None:
        if not args.video_root.exists():
            raise FileNotFoundError(f"video root not found: {args.video_root}")
        app = app_factory(
            args.video_root.resolve(),
            args.label_csv.resolve(),
            cat_json,
            merge_sources=merge_sources,
            datasets_scan_root=scan_root,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return

    launcher_app = _build_launcher_app(
        scan_root=scan_root,
        categories_json=cat_json,
        merge_sources=merge_sources,
    )
    uvicorn.run(launcher_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
