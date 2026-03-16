#!/usr/bin/env python
"""Headless-friendly web GUI for episode task review (A/B)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pyarrow.parquet as pq
import uvicorn


EP_RE = re.compile(r"(episode_\d+)\.mp4$")


class LabelPayload(BaseModel):
    episode_id: str
    label: str
    note: str = ""


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


def app_factory(video_root: Path, label_csv: Path) -> FastAPI:
    app = FastAPI(title="Episode Review")
    episodes = discover(video_root)
    labels = load_csv(label_csv)
    dataset_root = video_root.parent
    info_path = dataset_root / "meta" / "info.json"
    info: dict = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    joint_names = info.get("features", {}).get("agent_pos", {}).get("names", [])
    app.mount("/media", StaticFiles(directory=str(video_root)), name="media")

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
            "open_ok": False,
            "read_first_frame_ok": False,
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "warnings": [],
        }
        if not full_path.exists():
            result["warnings"].append("file_missing")
            return result
        if full_path.suffix.lower() != ".mp4":
            result["warnings"].append("not_mp4")
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
        if result["fps"] <= 0:
            result["warnings"].append("fps_invalid")
        if result["frame_count"] <= 0:
            result["warnings"].append("frame_count_invalid")
        if result["width"] <= 0 or result["height"] <= 0:
            result["warnings"].append("resolution_invalid")
        if not result["read_first_frame_ok"]:
            result["warnings"].append("first_frame_decode_failed")
        if result["size_bytes"] <= 0:
            result["warnings"].append("size_zero")
        return result

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return PAGE

    @app.get("/api/list")
    def list_episodes(q: str = "", lf: str = "all") -> dict:
        items = []
        for ep in sorted(episodes.keys()):
            if q and q.lower() not in ep.lower():
                continue
            label = labels.get(ep, {}).get("label", "")
            if lf == "unlabeled" and label:
                continue
            if lf in {"A", "B", "uncertain"} and label != lf:
                continue
            items.append(
                {
                    "episode_id": ep,
                    "camera_count": len(episodes[ep]),
                    "label": label,
                }
            )
        return {"items": items}

    @app.get("/api/episode/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        if episode_id not in episodes:
            return {"ok": False, "error": "episode not found"}
        videos = [
            {
                "camera": cam,
                "url": f"/media/{rel}",
                "rel_path": rel,
            }
            for cam, rel in sorted(episodes[episode_id].items())
        ]
        row = labels.get(episode_id, {})
        return {
            "ok": True,
            "episode_id": episode_id,
            "videos": videos,
            "label": row.get("label", ""),
            "note": row.get("note", ""),
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
        return {
            "ok": True,
            "episode_id": episode_id,
            "joint_names": joint_names,
            "frame_index": frame_index,
            "agent_pos": agent_pos,
        }

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
        if payload.label not in {"", "A", "B", "uncertain"}:
            return {"ok": False, "error": "label invalid"}
        labels[payload.episode_id] = {
            "episode_id": payload.episode_id,
            "label": payload.label,
            "note": payload.note,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_csv(label_csv, labels)
        return {"ok": True}

    @app.get("/api/meta")
    def meta() -> dict:
        return {"episode_count": len(episodes), "label_csv": str(label_csv)}

    return app


PAGE = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/>
<title>Episode Review</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:Arial}
.wrap{display:grid;grid-template-columns:320px 1fr;height:100vh}
.left{padding:10px;border-right:1px solid #333;overflow:auto}
.right{padding:10px;overflow:auto}
.item{padding:6px;border:1px solid #333;margin:6px 0;cursor:pointer}
.item.active{border-color:#66aaff;background:#1c2435}
input,select,button,textarea{background:#222;color:#eee;border:1px solid #555;padding:6px}
.videos{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}
.card{border:1px solid #333;padding:6px}.cam{font-size:12px;color:#bbb}
video{width:100%;max-height:320px;background:#000}
.panel{border:1px solid #333;padding:8px;margin-top:10px}
.small{font-size:12px;color:#bbb}
.warn{color:#ffb74d}
</style></head><body><div class='wrap'>
<div class='left'>
  <div><input id='q' placeholder='搜索 episode_000123' style='width:180px'/>
  <select id='lf'><option value='all'>全部</option><option value='unlabeled'>未标注</option><option value='A'>A</option><option value='B'>B</option><option value='uncertain'>不确定</option></select>
  <button id='refresh'>刷新</button></div>
  <div id='meta' style='margin:8px 0;color:#bbb'></div><div id='list'></div>
</div>
<div class='right'>
  <div id='head' style='font-size:18px;margin-bottom:8px'>未选择</div>
  <div id='save_status' style='font-size:12px;color:#8bc34a;margin-bottom:6px'></div>
  <div style='margin-bottom:8px'>
    <button onclick='setL("A")'>标A(键盘1)</button>
    <button onclick='setL("B")'>标B(键盘2)</button>
    <button onclick='setL("uncertain")'>不确定(键盘3)</button>
    <button onclick='setL("")'>清空(键盘0)</button>
    <button onclick='togglePlay()'>播放/暂停(空格)</button>
    <button onclick='save()'>保存(Enter)</button>
  </div>
  <textarea id='note' rows='2' style='width:100%;margin-bottom:8px' placeholder='备注'></textarea>
  <div id='videos' class='videos'></div>
  <div class='panel'>
    <div style='font-weight:bold;margin-bottom:6px'>视频质量检查</div>
    <button id='run_quality' style='margin-bottom:6px'>运行当前 episode 质量检查</button>
    <div id='quality' class='small'>未检查</div>
  </div>
  <div class='panel'>
    <div style='font-weight:bold;margin-bottom:6px'>机械臂关节角运动折线图（agent_pos）</div>
    <canvas id='joint_chart' width='1200' height='260' style='width:100%;background:#0f0f0f;border:1px solid #333'></canvas>
    <div id='joint_legend' class='small' style='margin-top:6px'></div>
  </div>
</div></div>
<script>
let cur=null, curLabel="", eps=[], playing=false;
const jointsCache = {};
const qualityCache = {};
async function j(u,o){const r=await fetch(u,o);return await r.json();}
async function loadMeta(){const d=await j('/api/meta');document.getElementById('meta').innerText=`episodes: ${d.episode_count}`;}
function renderQuality(checks){
  const el=document.getElementById('quality');
  if(!checks||!checks.length){ el.innerText='无质量数据'; return; }
  const lines=[];
  for(const c of checks){
    const warn=(c.warnings&&c.warnings.length)?` warnings=[${c.warnings.join(',')}]`:'';
    lines.push(`${c.camera}: ${c.width}x${c.height}, fps=${Number(c.fps).toFixed(2)}, frames=${c.frame_count}, open=${c.open_ok}, first_frame=${c.read_first_frame_ok}${warn}`);
  }
  el.innerHTML=lines.map(x=>`<div class='${x.includes('warnings=[')?'warn':''}'>${x}</div>`).join('');
}
function renderJointChart(payload){
  const canvas=document.getElementById('joint_chart');
  const legend=document.getElementById('joint_legend');
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!payload||!payload.ok){ legend.innerText='关节数据读取失败'; return; }
  const frames=payload.frame_index||[];
  const pos=payload.agent_pos||[];
  const names=payload.joint_names||[];
  if(!frames.length||!pos.length){ legend.innerText='关节数据为空'; return; }
  const dims=pos[0].length;
  let ymin=Infinity,ymax=-Infinity;
  for(const row of pos){ for(let i=0;i<dims;i++){ const v=row[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; } }
  if(ymax===ymin){ ymax=ymin+1e-6; }
  const pad=24, w=canvas.width-pad*2, h=canvas.height-pad*2;
  ctx.strokeStyle='#555'; ctx.strokeRect(pad,pad,w,h);
  const colors=['#ff5252','#ff9800','#ffeb3b','#4caf50','#00bcd4','#2196f3','#9c27b0','#e91e63','#8bc34a','#03a9f4','#ffc107','#795548','#f44336','#cddc39'];
  for(let d=0; d<dims; d++){
    ctx.beginPath(); ctx.strokeStyle=colors[d%colors.length]; ctx.lineWidth=1.1;
    for(let i=0; i<pos.length; i++){
      const x=pad + (i/(pos.length-1||1))*w;
      const y=pad + (1-(pos[i][d]-ymin)/(ymax-ymin))*h;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  const showNames=[];
  for(let i=0;i<Math.min(dims,14);i++){ showNames.push(`${i}:${names[i]||('joint_'+i)}`); }
  legend.innerText=`范围[${ymin.toFixed(3)}, ${ymax.toFixed(3)}] | ${showNames.join(' | ')}`;
}
async function loadAndRenderJoints(ep){
  if(jointsCache[ep]){ renderJointChart(jointsCache[ep]); return; }
  const jd=await j(`/api/episode/${ep}/joints`);
  jointsCache[ep]=jd;
  renderJointChart(jd);
}
async function runQualityCheck(ep){
  if(!ep) return;
  if(qualityCache[ep]){ renderQuality(qualityCache[ep]); return; }
  document.getElementById('quality').innerText='检查中...';
  const qd=await j(`/api/episode/${ep}/quality`);
  if(qd.ok){ qualityCache[ep]=qd.checks; renderQuality(qd.checks); }
  else { document.getElementById('quality').innerText=(qd.error||'质量检查失败'); }
}
async function loadList(){
  const q=encodeURIComponent(document.getElementById('q').value.trim());
  const lf=encodeURIComponent(document.getElementById('lf').value);
  const d=await j(`/api/list?q=${q}&lf=${lf}`); eps=d.items; const box=document.getElementById('list'); box.innerHTML='';
  for(const e of eps){ const div=document.createElement('div'); div.className='item'+(e.episode_id===cur?' active':'');
    div.innerText=`${e.episode_id} [cams:${e.camera_count}] ${e.label?('label:'+e.label):''}`;
    div.onclick=()=>selectEp(e.episode_id); box.appendChild(div); }
}
async function selectEp(ep){
  const d=await j(`/api/episode/${ep}`); if(!d.ok){alert(d.error); return;}
  cur=ep; curLabel=d.label||""; document.getElementById('head').innerText=`${ep} 当前标注: ${curLabel||'(空)'}`;
  document.getElementById('save_status').innerText='';
  document.getElementById('quality').innerText='点击“运行当前 episode 质量检查”';
  document.getElementById('note').value=d.note||""; const v=document.getElementById('videos'); v.innerHTML='';
  for(const x of d.videos){ const c=document.createElement('div'); c.className='card';
    const auto = x.camera.includes('left_wrist_cam');
    c.innerHTML=`<div class='cam'>${x.camera}${auto?' (自动播放)':''}</div><video controls preload='${auto?'metadata':'none'}' ${auto?'autoplay':''} muted playsinline src='${x.url}'></video>`; v.appendChild(c);}
  playing=true;
  const vs=document.querySelectorAll('video');
  for(const vv of vs){
    const isLeft = vv.src.includes('left_wrist_cam');
    if(isLeft){ const p=vv.play(); if(p&&p.catch){ p.catch(()=>{}); } }
    else { vv.pause(); }
  }
  loadAndRenderJoints(ep);
}
async function setL(l){
  curLabel=l;
  if(cur) document.getElementById('head').innerText=`${cur} 当前标注: ${curLabel||'(空)'}`;
  await save();
}
function togglePlay(){ const vs=document.querySelectorAll('video'); playing=!playing; vs.forEach(v=>playing?v.play():v.pause()); }
async function save(){ if(!cur){alert('先选择episode');return;}
  const note=document.getElementById('note').value; const d=await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode_id:cur,label:curLabel,note})});
  if(!d.ok){document.getElementById('save_status').style.color='#ff6b6b';document.getElementById('save_status').innerText='保存失败';alert(d.error||'保存失败');return;}
  document.getElementById('save_status').style.color='#8bc34a';
  document.getElementById('save_status').innerText=`已保存: ${cur} -> ${curLabel||'(空)'}`;
}
document.getElementById('refresh').onclick=loadList;
document.getElementById('run_quality').onclick=()=>runQualityCheck(cur);
document.addEventListener('keydown',e=>{ if(e.target&&['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if(e.key==='1')setL('A'); if(e.key==='2')setL('B'); if(e.key==='3')setL('uncertain'); if(e.key==='0')setL('');
  if(e.key===' ') {e.preventDefault(); togglePlay();} if(e.key==='Enter') save(); });
loadMeta(); loadList();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode task review GUI")
    p.add_argument("--video-root", type=Path, required=True, help="Root dir containing episode_*.mp4")
    p.add_argument("--label-csv", type=Path, required=True, help="CSV output path for labels")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18080)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video_root.exists():
        raise FileNotFoundError(f"video root not found: {args.video_root}")
    app = app_factory(args.video_root.resolve(), args.label_csv.resolve())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
