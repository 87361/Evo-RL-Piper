#!/usr/bin/env python3
"""
LeRobot v2.1 数据集帧级别裁剪工具（独立版 v3）
================================================
不依赖 lerobot 库，仅需: pip install pandas pyarrow numpy
系统依赖: ffmpeg（用于视频裁剪）

使用方法:
  1. 修改下方【用户配置区】的参数
  2. 直接运行: python lerobot_editor.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         【用户配置区】                                    ║
# ║                    所有参数都在这里修改                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────
# 1. 数据集路径
# ──────────────────────────────────────────────
DATASET_DIR = "/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_batch1/alicia_dual_piper_0330_batch1_copy"

# ──────────────────────────────────────────────
# 2. 操作模式（选一个，取消注释即可）
# ──────────────────────────────────────────────
#   "inspect"  - 查看数据集信息
#   "trim"     - 裁剪指定 episode 的帧
#   "remove"   - 删除整个 episode
#   "batch"    - 批量裁剪多个 episode
#   "verify"   - 验证数据集完整性
MODE = "trim"

# ──────────────────────────────────────────────
# 3. 输出方式
# ──────────────────────────────────────────────
#   True  = 拷贝到新目录再修改（安全，不动原数据）
#   False = 直接在原目录上修改（快速，但不可逆）
COPY_TO_NEW_DIR = False

# 输出目录（仅当 COPY_TO_NEW_DIR = True 时生效）
# 留空 "" 则自动生成: 原目录名 + OUTPUT_SUFFIX
OUTPUT_DIR = ""
OUTPUT_SUFFIX = "_trimmed"

# ──────────────────────────────────────────────
# 4. inspect 模式参数
# ──────────────────────────────────────────────
# 要查看的 episode 编号，设为 None 则查看整个数据集概览
INSPECT_EPISODE = 5

# ──────────────────────────────────────────────
# 5. trim 模式参数
# ──────────────────────────────────────────────
TRIM_EPISODE = 5            # 要裁剪的 episode 编号
TRIM_FRAMES = "0-60"        # 要删除的帧范围
                            #   "0-50"    → 删除开头 50 帧
                            #   "200-end" → 删除第 200 帧到末尾
                            #   "100-200" → 删除中间第 100~199 帧

# ──────────────────────────────────────────────
# 6. remove 模式参数
# ──────────────────────────────────────────────
REMOVE_EPISODE = 3          # 要删除的 episode 编号

# ──────────────────────────────────────────────
# 7. batch 模式参数
# ──────────────────────────────────────────────
# 每一项: {"episode": 编号, "frames": "要删除的范围"}
BATCH_TASKS = [
    {"episode": 0, "frames": "0-30"},
    {"episode": 3, "frames": "200-end"},
    {"episode": 7, "frames": "100-150"},
]

# ──────────────────────────────────────────────
# 8. ffmpeg 视频编码参数
# ──────────────────────────────────────────────
FFMPEG_VCODEC = "libx264"   # 视频编码器
FFMPEG_PRESET = "fast"      # 编码速度 (ultrafast/fast/medium/slow)
FFMPEG_CRF = "18"           # 质量 (0=无损, 18=高质量, 23=默认, 28=低质量)

# ──────────────────────────────────────────────
# 9. 图片文件格式
# ──────────────────────────────────────────────
IMAGE_EXTENSION = ".png"    # 逐帧图片的扩展名


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      以下为工具代码，无需修改                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# =============================================================================
# 数据集路径工具
# =============================================================================

def get_parquet_path(dataset_dir: Path, ep_idx: int, info: dict) -> Path:
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = info.get("chunks_size", 1000)
    return dataset_dir / template.format(episode_chunk=ep_idx // chunks_size, episode_index=ep_idx)


def get_video_keys(info: dict) -> List[str]:
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]


def get_image_keys(info: dict) -> List[str]:
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "image"]


def get_video_path(dataset_dir: Path, ep_idx: int, video_key: str, info: dict) -> Path:
    template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    chunks_size = info.get("chunks_size", 1000)
    return dataset_dir / template.format(
        episode_chunk=ep_idx // chunks_size, video_key=video_key, episode_index=ep_idx
    )


def get_image_dir(dataset_dir: Path, ep_idx: int, image_key: str) -> Path:
    return dataset_dir / "images" / image_key / f"episode_{ep_idx:06d}"


def get_image_frame_path(image_dir: Path, frame_idx: int) -> Path:
    return image_dir / f"frame_{frame_idx:06d}{IMAGE_EXTENSION}"


# =============================================================================
# 元数据读写
# =============================================================================

def load_info(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def save_info(output_dir: Path, info: dict) -> None:
    path = output_dir / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def load_episodes(dataset_dir: Path) -> List[dict]:
    episodes = []
    path = dataset_dir / "meta" / "episodes.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def save_episodes(output_dir: Path, episodes: List[dict]) -> None:
    path = output_dir / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def save_episodes_stats(output_dir: Path, stats: List[dict]) -> None:
    path = output_dir / "meta" / "episodes_stats.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# =============================================================================
# 输出目录
# =============================================================================

def resolve_output_dir(src_dir: Path) -> Path:
    if not COPY_TO_NEW_DIR:
        return src_dir
    if OUTPUT_DIR:
        return Path(OUTPUT_DIR)
    return src_dir.parent / (src_dir.name + OUTPUT_SUFFIX)


def init_output_dir(src_dir: Path, out_dir: Path) -> None:
    if src_dir == out_dir:
        print(f"  直接在原目录上操作: {out_dir}")
        return
    if out_dir.exists():
        print(f"  输出目录已存在，在此基础上继续操作: {out_dir}")
        return
    print(f"  正在拷贝数据集到输出目录...")
    print(f"    源: {src_dir}")
    print(f"    目标: {out_dir}")
    shutil.copytree(src_dir, out_dir)
    print(f"  拷贝完成")


# =============================================================================
# 统计量计算
# =============================================================================

def compute_single_episode_stats(df: pd.DataFrame, features: dict) -> dict:
    stats = {}
    skip_keys = {"frame_index", "episode_index", "index", "task_index", "timestamp"}
    for key, feat in features.items():
        if feat.get("dtype") not in ("float32", "float64", "int64"):
            continue
        if key in skip_keys or key not in df.columns:
            continue
        col = df[key]
        first_val = col.iloc[0]
        if isinstance(first_val, (list, np.ndarray)):
            arr = np.array(col.tolist(), dtype=np.float32)
        else:
            arr = col.values.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }
    return stats


def recompute_all_stats(output_dir: Path, info: dict) -> None:
    episodes = load_episodes(output_dir)
    features = info.get("features", {})
    all_stats = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        pq_path = get_parquet_path(output_dir, ep_idx, info)
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        ep_stats = compute_single_episode_stats(df, features)
        ep_stats["episode_index"] = ep_idx
        all_stats.append(ep_stats)
    save_episodes_stats(output_dir, all_stats)
    print(f"  已重新计算 {len(all_stats)} 个 episode 的统计信息")


# =============================================================================
# 视频裁剪
# =============================================================================

def trim_video_keep_range(src_video: Path, dst_video: Path, fps: int, keep_start: int, keep_end: int) -> bool:
    if not src_video.exists():
        return False
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    start_time = keep_start / fps
    duration = (keep_end - keep_start) / fps
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.6f}", "-i", str(src_video),
        "-t", f"{duration:.6f}",
        "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an",
        str(dst_video),
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def trim_video_remove_middle(src_video: Path, dst_video: Path, fps: int,
                              del_start: int, del_end: int, total: int) -> bool:
    if not src_video.exists():
        return False
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dst_video.parent / f"_tmp_{dst_video.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part1, part2 = tmp_dir / "part1.mp4", tmp_dir / "part2.mp4"
    concat_list = tmp_dir / "concat.txt"
    try:
        cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_video),
                "-t", f"{del_start / fps:.6f}",
                "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(part1)]
        if subprocess.run(cmd1, capture_output=True).returncode != 0:
            return False
        cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{del_end / fps:.6f}",
                "-i", str(src_video),
                "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(part2)]
        if subprocess.run(cmd2, capture_output=True).returncode != 0:
            return False
        with open(concat_list, "w") as f:
            f.write(f"file '{part1}'\nfile '{part2}'\n")
        cmd3 = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(dst_video)]
        return subprocess.run(cmd3, capture_output=True).returncode == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# 图片裁剪
# =============================================================================

def trim_images(src_img_dir: Path, dst_img_dir: Path, keep_indices: List[int]) -> int:
    if not src_img_dir.exists():
        return 0
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for new_idx, old_idx in enumerate(keep_indices):
        src_file = get_image_frame_path(src_img_dir, old_idx)
        dst_file = get_image_frame_path(dst_img_dir, new_idx)
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            count += 1
    return count


# =============================================================================
# 帧范围解析
# =============================================================================

def parse_frame_range(expr: str, total_frames: int) -> Tuple[int, int]:
    expr = expr.replace(":", "-")
    if "-" not in expr:
        raise ValueError(f"帧范围必须包含 '-'，例如 '100-200' 或 '250-end'，收到: '{expr}'")
    parts = expr.split("-", 1)
    start_s, end_s = parts[0].strip(), parts[1].strip()
    start = int(start_s) if start_s else 0
    end = total_frames if end_s in ("end", "") else int(end_s)
    if not (0 <= start < end <= total_frames):
        raise ValueError(f"范围 [{start}, {end}) 无效（总帧数: {total_frames}）")
    return start, end


# =============================================================================
# 核心操作
# =============================================================================

def find_episode(episodes: List[dict], ep_idx: int) -> dict | None:
    for ep in episodes:
        if ep["episode_index"] == ep_idx:
            return ep
    return None


def do_inspect(dataset_dir: Path, episode: int | None) -> None:
    info = load_info(dataset_dir)
    episodes = load_episodes(dataset_dir)

    if episode is not None:
        ep_entry = find_episode(episodes, episode)
        if ep_entry is None:
            print(f"Episode {episode} 不存在")
            return

        pq_path = get_parquet_path(dataset_dir, episode, info)
        print(f"Episode {episode}:")
        print(f"  元数据帧数: {ep_entry.get('length', 'N/A')}")
        print(f"  任务: {ep_entry.get('tasks', [])}")
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            print(f"  Parquet 实际行数: {len(df)}")
            print(f"  列名: {list(df.columns)}")

        print(f"  视频 (dtype=video):")
        for vkey in get_video_keys(info):
            vpath = get_video_path(dataset_dir, episode, vkey, info)
            print(f"    {vkey}: {'✓' if vpath.exists() else '✗ 缺失'}")

        print(f"  图片 (dtype=image):")
        for ikey in get_image_keys(info):
            idir = get_image_dir(dataset_dir, episode, ikey)
            if idir.exists():
                count = len(list(idir.glob(f"*{IMAGE_EXTENSION}")))
                print(f"    {ikey}: ✓ {count} 帧")
            else:
                print(f"    {ikey}: ✗ 缺失")
    else:
        print(f"数据集: {dataset_dir}")
        print(f"  格式版本: {info.get('codebase_version', 'unknown')}")
        print(f"  机器人类型: {info.get('robot_type', 'unknown')}")
        print(f"  FPS: {info.get('fps', 'N/A')}")
        print(f"  总 episodes: {info.get('total_episodes', len(episodes))}")
        print(f"  总帧数: {info.get('total_frames', 'N/A')}")
        print(f"  数据特征: {[k for k, f in info.get('features', {}).items() if f.get('dtype') not in ('video', 'image')]}")
        print(f"  视频特征 (MP4): {get_video_keys(info)}")
        print(f"  图片特征 (PNG): {get_image_keys(info)}")
        print(f"\n  Episode 帧数列表:")
        for ep in episodes:
            print(f"    Episode {ep['episode_index']:3d}: {ep.get('length', 'N/A'):>5} 帧")


def do_trim(src_dir: Path, out_dir: Path, ep_idx: int, del_start: int, del_end: int) -> None:
    info = load_info(out_dir)
    episodes = load_episodes(out_dir)
    fps = info.get("fps", 10)

    ep_entry = find_episode(episodes, ep_idx)
    if ep_entry is None:
        print(f"错误: Episode {ep_idx} 不存在")
        sys.exit(1)

    pq_path = get_parquet_path(out_dir, ep_idx, info)
    if not pq_path.exists():
        print(f"错误: 找不到 {pq_path}")
        sys.exit(1)

    df = pd.read_parquet(pq_path)
    original_len = len(df)
    keep_indices = [i for i in range(original_len) if not (del_start <= i < del_end)]
    new_len = len(keep_indices)

    if new_len == 0:
        print("错误: 所有帧都会被删除，请改用 remove 模式")
        sys.exit(1)

    print(f"  原始帧数: {original_len}")
    print(f"  删除帧: [{del_start}, {del_end})  共 {del_end - del_start} 帧")
    print(f"  裁剪后帧数: {new_len}")

    # 1. 裁剪 Parquet
    df_trimmed = df.iloc[keep_indices].copy().reset_index(drop=True)
    df_trimmed["frame_index"] = range(new_len)
    df_trimmed["timestamp"] = [i / fps for i in range(new_len)]
    df_trimmed.to_parquet(pq_path, index=False)
    print(f"  ✓ Parquet 已更新")

    # 2. 裁剪视频
    inplace = (src_dir == out_dir)
    is_contiguous = all(keep_indices[i] + 1 == keep_indices[i + 1] for i in range(len(keep_indices) - 1))
    for vkey in get_video_keys(info):
        src_video = get_video_path(src_dir, ep_idx, vkey, info)
        dst_video = get_video_path(out_dir, ep_idx, vkey, info)
        if not src_video.exists():
            print(f"  ⚠ 视频缺失: {vkey}")
            continue
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        # 当原地操作时，先输出到临时文件，再替换原文件
        if inplace:
            tmp_video = dst_video.with_suffix(".trimtmp.mp4")
            if is_contiguous:
                ok = trim_video_keep_range(src_video, tmp_video, fps, keep_indices[0], keep_indices[-1] + 1)
            else:
                ok = trim_video_remove_middle(src_video, tmp_video, fps, del_start, del_end, original_len)
            if ok:
                tmp_video.replace(dst_video)
            elif tmp_video.exists():
                tmp_video.unlink()
        else:
            if is_contiguous:
                ok = trim_video_keep_range(src_video, dst_video, fps, keep_indices[0], keep_indices[-1] + 1)
            else:
                ok = trim_video_remove_middle(src_video, dst_video, fps, del_start, del_end, original_len)
        print(f"  {'✓' if ok else '✗ 失败'} 视频裁剪: {vkey}")

    # 3. 裁剪图片
    for ikey in get_image_keys(info):
        src_img_dir = get_image_dir(src_dir, ep_idx, ikey)
        dst_img_dir = get_image_dir(out_dir, ep_idx, ikey)
        if not src_img_dir.exists():
            print(f"  ⚠ 图片目录缺失: {ikey}")
            continue
        if inplace:
            # 原地操作：先重命名到临时目录，从临时目录拷贝保留帧，再删除临时目录
            tmp_img_dir = src_img_dir.parent / f"{src_img_dir.name}_trimtmp"
            src_img_dir.rename(tmp_img_dir)
            count = trim_images(tmp_img_dir, dst_img_dir, keep_indices)
            shutil.rmtree(tmp_img_dir, ignore_errors=True)
        else:
            if dst_img_dir.exists():
                shutil.rmtree(dst_img_dir)
            count = trim_images(src_img_dir, dst_img_dir, keep_indices)
        print(f"  ✓ 图片裁剪: {ikey} ({count} 帧)")

    # 4. 更新元数据
    ep_entry["length"] = new_len
    save_episodes(out_dir, episodes)
    info["total_frames"] = info.get("total_frames", 0) - (original_len - new_len)
    save_info(out_dir, info)
    print(f"  ✓ 元数据已更新")

    # 5. 重新计算统计量
    print(f"  正在重新计算统计信息...")
    recompute_all_stats(out_dir, info)
    print(f"✔ Episode {ep_idx} 裁剪完成: {original_len} → {new_len} 帧")


def do_remove(src_dir: Path, out_dir: Path, ep_idx: int) -> None:
    info = load_info(out_dir)
    episodes = load_episodes(out_dir)

    ep_entry = find_episode(episodes, ep_idx)
    if ep_entry is None:
        print(f"错误: Episode {ep_idx} 不存在")
        sys.exit(1)

    new_episodes = [ep for ep in episodes if ep["episode_index"] != ep_idx]

    pq_path = get_parquet_path(out_dir, ep_idx, info)
    if pq_path.exists():
        pq_path.unlink()
        print(f"  ✓ 已删除 parquet")

    for vkey in get_video_keys(info):
        vpath = get_video_path(out_dir, ep_idx, vkey, info)
        if vpath.exists():
            vpath.unlink()
            print(f"  ✓ 已删除视频: {vkey}")

    for ikey in get_image_keys(info):
        img_dir = get_image_dir(out_dir, ep_idx, ikey)
        if img_dir.exists():
            shutil.rmtree(img_dir)
            print(f"  ✓ 已删除图片: {ikey}")

    frames_removed = ep_entry.get("length", 0)
    save_episodes(out_dir, new_episodes)
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = info.get("total_frames", 0) - frames_removed
    info["total_videos"] = len(get_video_keys(info)) * len(new_episodes)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    save_info(out_dir, info)

    print(f"  正在重新计算统计信息...")
    recompute_all_stats(out_dir, info)
    print(f"✔ Episode {ep_idx} 已删除 ({frames_removed} 帧)")


def do_batch(src_dir: Path, out_dir: Path, tasks: List[dict]) -> None:
    print(f"批量裁剪: 共 {len(tasks)} 个任务")
    for i, task in enumerate(tasks):
        ep_idx = task["episode"]
        frame_expr = task["frames"]
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(tasks)}] Episode {ep_idx}, 删除帧: {frame_expr}")
        print(f"{'='*50}")

        info = load_info(out_dir)
        episodes = load_episodes(out_dir)
        ep_entry = find_episode(episodes, ep_idx)
        if ep_entry is None:
            print(f"  跳过: Episode {ep_idx} 不存在")
            continue

        try:
            del_start, del_end = parse_frame_range(frame_expr, ep_entry.get("length", 0))
        except ValueError as e:
            print(f"  跳过: {e}")
            continue

        do_trim(src_dir, out_dir, ep_idx, del_start, del_end)

    print(f"\n✔ 批量裁剪完成")


def do_verify(dataset_dir: Path) -> None:
    info = load_info(dataset_dir)
    episodes = load_episodes(dataset_dir)
    errors = 0

    print(f"正在验证: {dataset_dir}")
    print(f"  共 {len(episodes)} 个 episode\n")

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep.get("length", 0)
        prefix = f"  Episode {ep_idx:3d}:"
        ep_ok = True

        pq_path = get_parquet_path(dataset_dir, ep_idx, info)
        if not pq_path.exists():
            print(f"{prefix} ✗ Parquet 缺失")
            errors += 1
            continue
        df = pd.read_parquet(pq_path)
        if len(df) != ep_len:
            print(f"{prefix} ✗ Parquet 帧数不匹配 (元数据: {ep_len}, 实际: {len(df)})")
            errors += 1
            ep_ok = False

        for vkey in get_video_keys(info):
            vpath = get_video_path(dataset_dir, ep_idx, vkey, info)
            if not vpath.exists():
                print(f"{prefix} ✗ 视频缺失: {vkey}")
                errors += 1
                ep_ok = False

        for ikey in get_image_keys(info):
            idir = get_image_dir(dataset_dir, ep_idx, ikey)
            if idir.exists():
                img_count = len(list(idir.glob(f"*{IMAGE_EXTENSION}")))
                if img_count != ep_len:
                    print(f"{prefix} ✗ 图片帧数不匹配: {ikey} (元数据: {ep_len}, 实际: {img_count})")
                    errors += 1
                    ep_ok = False
            else:
                print(f"{prefix} ✗ 图片目录缺失: {ikey}")
                errors += 1
                ep_ok = False

        if ep_ok:
            print(f"{prefix} ✓ 正常 ({ep_len} 帧)")

    actual_total = sum(ep.get("length", 0) for ep in episodes)
    if info.get("total_frames") != actual_total:
        print(f"\n  ✗ total_frames 不匹配 (info.json: {info.get('total_frames')}, 实际: {actual_total})")
        errors += 1

    print(f"\n{'✔ 验证通过，无错误' if errors == 0 else f'✗ 发现 {errors} 个错误'}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    src_dir = Path(DATASET_DIR)
    if not src_dir.exists():
        print(f"错误: 数据集目录不存在: {src_dir}")
        sys.exit(1)

    out_dir = resolve_output_dir(src_dir)

    print(f"LeRobot 数据集裁剪工具 v3")
    print(f"  数据集: {src_dir}")
    print(f"  操作模式: {MODE}")
    print(f"  输出方式: {'拷贝到新目录' if COPY_TO_NEW_DIR else '直接修改原数据'}")
    if COPY_TO_NEW_DIR:
        print(f"  输出目录: {out_dir}")
    print()

    if MODE == "inspect":
        do_inspect(src_dir, INSPECT_EPISODE)

    elif MODE == "trim":
        init_output_dir(src_dir, out_dir)
        info = load_info(out_dir)
        episodes = load_episodes(out_dir)
        ep_entry = find_episode(episodes, TRIM_EPISODE)
        if ep_entry is None:
            print(f"错误: Episode {TRIM_EPISODE} 不存在")
            sys.exit(1)
        del_start, del_end = parse_frame_range(TRIM_FRAMES, ep_entry.get("length", 0))
        print(f"正在裁剪 Episode {TRIM_EPISODE}...")
        do_trim(src_dir, out_dir, TRIM_EPISODE, del_start, del_end)

    elif MODE == "remove":
        init_output_dir(src_dir, out_dir)
        print(f"正在删除 Episode {REMOVE_EPISODE}...")
        do_remove(src_dir, out_dir, REMOVE_EPISODE)

    elif MODE == "batch":
        init_output_dir(src_dir, out_dir)
        do_batch(src_dir, out_dir, BATCH_TASKS)

    elif MODE == "verify":
        do_verify(src_dir)

    else:
        print(f"错误: 未知操作模式 '{MODE}'，可选: inspect / trim / remove / batch / verify")
        sys.exit(1)


if __name__ == "__main__":
    main()