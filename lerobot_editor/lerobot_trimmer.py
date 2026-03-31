#!/usr/bin/env python3
"""
LeRobot v2.1 数据集帧级别裁剪工具（独立版 v2）
================================================
不依赖 lerobot 库，仅需: pip install pandas pyarrow numpy click
系统依赖: ffmpeg（用于视频裁剪）

特性:
  - 所有修改都输出到新目录，原始数据绝不改动
  - 默认输出到 原目录名_trimmed，可通过 -o 自定义
  - 支持 MP4 视频 (dtype: video) 和逐帧图片 (dtype: image, 如深度图)
  - 自动更新 parquet、视频、图片、元数据、统计信息

用法:
  python lerobot_trimmer.py inspect /data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_batch1/alicia_dual_piper_0330_batch1_copy
  python lerobot_trimmer.py inspect DATASET_DIR -e 5
  python lerobot_trimmer.py trim /data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_batch1/alicia_dual_piper_0330_batch1_copy -e 5 -f 0-60
  python lerobot_trimmer.py trim DATASET_DIR -e 5 -f 0-50 -o /path/to/output
  python lerobot_trimmer.py remove DATASET_DIR -e 3
  python lerobot_trimmer.py batch DATASET_DIR -c trim_config.json
  python lerobot_trimmer.py verify DATASET_DIR
"""
#lerobot_editor.py的初始版本，没有将所有的可操作整理在top
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import pandas as pd


# =============================================================================
# 默认参数配置（可根据需要修改）
# =============================================================================

# 默认输出目录后缀，最终输出路径 = 原目录路径 + 此后缀
DEFAULT_OUTPUT_SUFFIX = "_trimmed"

# ffmpeg 视频编码参数
FFMPEG_VCODEC = "libx264"       # 视频编码器
FFMPEG_PRESET = "fast"          # 编码速度预设 (ultrafast/fast/medium/slow)
FFMPEG_CRF = "18"               # 质量 (0=无损, 18=高质量, 23=默认, 28=低质量)

# 图片文件格式
IMAGE_EXTENSION = ".png"        # 逐帧图片的扩展名


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


def load_episodes_stats(dataset_dir: Path) -> List[dict]:
    stats = []
    path = dataset_dir / "meta" / "episodes_stats.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    stats.append(json.loads(line))
    return stats


def save_episodes_stats(output_dir: Path, stats: List[dict]) -> None:
    path = output_dir / "meta" / "episodes_stats.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# =============================================================================
# 初始化输出目录
# =============================================================================

def init_output_dir(src_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        click.echo(f"  输出目录已存在，将在此基础上继续操作: {out_dir}")
        return
    click.echo(f"  正在拷贝数据集到输出目录...")
    click.echo(f"    源: {src_dir}")
    click.echo(f"    目标: {out_dir}")
    shutil.copytree(src_dir, out_dir)
    click.echo(f"  拷贝完成")


def get_default_output_dir(src_dir: Path) -> Path:
    return src_dir.parent / (src_dir.name + DEFAULT_OUTPUT_SUFFIX)


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
    click.echo(f"  已重新计算 {len(all_stats)} 个 episode 的统计信息")


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
    part1 = tmp_dir / "part1.mp4"
    part2 = tmp_dir / "part2.mp4"
    concat_list = tmp_dir / "concat.txt"
    try:
        cmd1 = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src_video),
            "-t", f"{del_start / fps:.6f}",
            "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(part1),
        ]
        if subprocess.run(cmd1, capture_output=True).returncode != 0:
            return False
        cmd2 = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{del_end / fps:.6f}", "-i", str(src_video),
            "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(part2),
        ]
        if subprocess.run(cmd2, capture_output=True).returncode != 0:
            return False
        with open(concat_list, "w") as f:
            f.write(f"file '{part1}'\n")
            f.write(f"file '{part2}'\n")
        cmd3 = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:v", FFMPEG_VCODEC, "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF, "-an", str(dst_video),
        ]
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
        raise click.BadParameter(f"帧范围必须包含 '-'，例如 '100-200' 或 '250-end'")
    parts = expr.split("-", 1)
    start_s, end_s = parts[0].strip(), parts[1].strip()
    start = int(start_s) if start_s else 0
    end = total_frames if end_s in ("end", "") else int(end_s)
    if not (0 <= start < end <= total_frames):
        raise click.BadParameter(f"范围 [{start}, {end}) 无效（总帧数: {total_frames}）")
    return start, end


# =============================================================================
# 核心操作
# =============================================================================

def do_trim(src_dir: Path, out_dir: Path, ep_idx: int, del_start: int, del_end: int) -> None:
    info = load_info(out_dir)
    episodes = load_episodes(out_dir)
    fps = info.get("fps", 10)

    ep_entry = None
    for ep in episodes:
        if ep["episode_index"] == ep_idx:
            ep_entry = ep
            break
    if ep_entry is None:
        click.echo(f"错误: Episode {ep_idx} 不存在", err=True)
        sys.exit(1)

    pq_path = get_parquet_path(out_dir, ep_idx, info)
    if not pq_path.exists():
        click.echo(f"错误: 找不到 {pq_path}", err=True)
        sys.exit(1)

    df = pd.read_parquet(pq_path)
    original_len = len(df)
    keep_indices = [i for i in range(original_len) if not (del_start <= i < del_end)]
    new_len = len(keep_indices)

    if new_len == 0:
        click.echo("错误: 所有帧都会被删除，请改用 remove 命令", err=True)
        sys.exit(1)

    click.echo(f"  原始帧数: {original_len}")
    click.echo(f"  删除帧: [{del_start}, {del_end})  共 {del_end - del_start} 帧")
    click.echo(f"  裁剪后帧数: {new_len}")

    # 1. 裁剪 Parquet
    df_trimmed = df.iloc[keep_indices].copy().reset_index(drop=True)
    df_trimmed["frame_index"] = range(new_len)
    df_trimmed["timestamp"] = [i / fps for i in range(new_len)]
    df_trimmed.to_parquet(pq_path, index=False)
    click.echo(f"  ✓ Parquet 已更新")

    # 2. 裁剪视频 (dtype: video)
    is_contiguous = all(keep_indices[i] + 1 == keep_indices[i + 1] for i in range(len(keep_indices) - 1))
    for vkey in get_video_keys(info):
        src_video = get_video_path(src_dir, ep_idx, vkey, info)
        dst_video = get_video_path(out_dir, ep_idx, vkey, info)
        if not src_video.exists():
            click.echo(f"  ⚠ 视频缺失: {vkey}")
            continue
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if is_contiguous:
            ok = trim_video_keep_range(src_video, dst_video, fps, keep_indices[0], keep_indices[-1] + 1)
        else:
            ok = trim_video_remove_middle(src_video, dst_video, fps, del_start, del_end, original_len)
        click.echo(f"  {'✓' if ok else '✗ 失败'} 视频裁剪: {vkey}")

    # 3. 裁剪图片 (dtype: image)
    for ikey in get_image_keys(info):
        src_img_dir = get_image_dir(src_dir, ep_idx, ikey)
        dst_img_dir = get_image_dir(out_dir, ep_idx, ikey)
        if not src_img_dir.exists():
            click.echo(f"  ⚠ 图片目录缺失: {ikey}")
            continue
        if dst_img_dir.exists():
            shutil.rmtree(dst_img_dir)
        count = trim_images(src_img_dir, dst_img_dir, keep_indices)
        click.echo(f"  ✓ 图片裁剪: {ikey} ({count} 帧)")

    # 4. 更新元数据
    frames_removed = original_len - new_len
    ep_entry["length"] = new_len
    save_episodes(out_dir, episodes)
    info["total_frames"] = info.get("total_frames", 0) - frames_removed
    save_info(out_dir, info)
    click.echo(f"  ✓ 元数据已更新")

    # 5. 重新计算统计量
    click.echo(f"  正在重新计算统计信息...")
    recompute_all_stats(out_dir, info)
    click.echo(f"✔ Episode {ep_idx} 裁剪完成: {original_len} → {new_len} 帧")


def do_remove(src_dir: Path, out_dir: Path, ep_idx: int) -> None:
    info = load_info(out_dir)
    episodes = load_episodes(out_dir)

    ep_entry = None
    new_episodes = []
    for ep in episodes:
        if ep["episode_index"] == ep_idx:
            ep_entry = ep
        else:
            new_episodes.append(ep)
    if ep_entry is None:
        click.echo(f"错误: Episode {ep_idx} 不存在", err=True)
        sys.exit(1)

    pq_path = get_parquet_path(out_dir, ep_idx, info)
    if pq_path.exists():
        pq_path.unlink()
        click.echo(f"  ✓ 已删除 parquet")

    for vkey in get_video_keys(info):
        vpath = get_video_path(out_dir, ep_idx, vkey, info)
        if vpath.exists():
            vpath.unlink()
            click.echo(f"  ✓ 已删除视频: {vkey}")

    for ikey in get_image_keys(info):
        img_dir = get_image_dir(out_dir, ep_idx, ikey)
        if img_dir.exists():
            shutil.rmtree(img_dir)
            click.echo(f"  ✓ 已删除图片: {ikey}")

    frames_removed = ep_entry.get("length", 0)
    save_episodes(out_dir, new_episodes)
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = info.get("total_frames", 0) - frames_removed
    info["total_videos"] = len(get_video_keys(info)) * len(new_episodes)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    save_info(out_dir, info)

    click.echo(f"  正在重新计算统计信息...")
    recompute_all_stats(out_dir, info)
    click.echo(f"✔ Episode {ep_idx} 已删除 ({frames_removed} 帧)")


# =============================================================================
# CLI
# =============================================================================

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """LeRobot v2.1 数据集帧级别裁剪工具（独立版 v2）

    所有修改都输出到新目录，原始数据绝不改动。
    默认输出到 原目录名_trimmed，可通过 -o 自定义。
    """
    pass


@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--episode", "-e", type=int, default=None, help="查看指定 episode 的详细信息")
def inspect(dataset_dir: Path, episode: Optional[int]):
    """查看数据集 / episode 信息"""
    info = load_info(dataset_dir)
    episodes = load_episodes(dataset_dir)

    if episode is not None:
        ep_entry = None
        for ep in episodes:
            if ep["episode_index"] == episode:
                ep_entry = ep
                break
        if ep_entry is None:
            click.echo(f"Episode {episode} 不存在")
            return

        pq_path = get_parquet_path(dataset_dir, episode, info)
        click.echo(f"Episode {episode}:")
        click.echo(f"  元数据帧数: {ep_entry.get('length', 'N/A')}")
        click.echo(f"  任务: {ep_entry.get('tasks', [])}")
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            click.echo(f"  Parquet 实际行数: {len(df)}")
            click.echo(f"  列名: {list(df.columns)}")

        click.echo(f"  视频 (dtype=video):")
        for vkey in get_video_keys(info):
            vpath = get_video_path(dataset_dir, episode, vkey, info)
            click.echo(f"    {vkey}: {'✓' if vpath.exists() else '✗ 缺失'}")

        click.echo(f"  图片 (dtype=image):")
        for ikey in get_image_keys(info):
            idir = get_image_dir(dataset_dir, episode, ikey)
            if idir.exists():
                count = len(list(idir.glob(f"*{IMAGE_EXTENSION}")))
                click.echo(f"    {ikey}: ✓ {count} 帧")
            else:
                click.echo(f"    {ikey}: ✗ 缺失")
    else:
        click.echo(f"数据集: {dataset_dir}")
        click.echo(f"  格式版本: {info.get('codebase_version', 'unknown')}")
        click.echo(f"  机器人类型: {info.get('robot_type', 'unknown')}")
        click.echo(f"  FPS: {info.get('fps', 'N/A')}")
        click.echo(f"  总 episodes: {info.get('total_episodes', len(episodes))}")
        click.echo(f"  总帧数: {info.get('total_frames', 'N/A')}")
        click.echo(f"  数据特征: {[k for k, f in info.get('features', {}).items() if f.get('dtype') not in ('video', 'image')]}")
        click.echo(f"  视频特征 (MP4): {get_video_keys(info)}")
        click.echo(f"  图片特征 (PNG): {get_image_keys(info)}")
        click.echo(f"\n  Episode 帧数列表:")
        for ep in episodes:
            click.echo(f"    Episode {ep['episode_index']:3d}: {ep.get('length', 'N/A'):>5} 帧")


@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--episode", "-e", required=True, type=int, help="要裁剪的 episode 编号")
@click.option("--frames", "-f", required=True, type=str,
              help="要删除的帧范围 (例: '0-50', '200-end', '100-200')")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help=f"输出目录 (默认: 原目录名{DEFAULT_OUTPUT_SUFFIX})")
def trim(dataset_dir: Path, episode: int, frames: str, output: Optional[Path]):
    """裁剪 episode 中指定范围的帧"""
    out_dir = output or get_default_output_dir(dataset_dir)
    init_output_dir(dataset_dir, out_dir)

    info = load_info(out_dir)
    episodes = load_episodes(out_dir)
    ep_entry = None
    for ep in episodes:
        if ep["episode_index"] == episode:
            ep_entry = ep
            break
    if ep_entry is None:
        click.echo(f"错误: Episode {episode} 不存在", err=True)
        sys.exit(1)

    del_start, del_end = parse_frame_range(frames, ep_entry.get("length", 0))
    click.echo(f"\n正在裁剪 Episode {episode}...")
    click.echo(f"  输出目录: {out_dir}")
    do_trim(dataset_dir, out_dir, episode, del_start, del_end)


@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--episode", "-e", required=True, type=int, help="要删除的 episode 编号")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help=f"输出目录 (默认: 原目录名{DEFAULT_OUTPUT_SUFFIX})")
def remove(dataset_dir: Path, episode: int, output: Optional[Path]):
    """删除整个 episode"""
    out_dir = output or get_default_output_dir(dataset_dir)
    init_output_dir(dataset_dir, out_dir)
    click.echo(f"\n正在删除 Episode {episode}...")
    click.echo(f"  输出目录: {out_dir}")
    do_remove(dataset_dir, out_dir, episode)


@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "-c", required=True, type=click.Path(exists=True, path_type=Path),
              help="批量裁剪配置文件 (JSON)")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help=f"输出目录 (默认: 原目录名{DEFAULT_OUTPUT_SUFFIX})")
def batch(dataset_dir: Path, config: Path, output: Optional[Path]):
    """批量裁剪多个 episode

    配置文件格式:
    \b
    [
        {"episode": 0, "frames": "0-30"},
        {"episode": 3, "frames": "200-end"},
        {"episode": 7, "frames": "100-150"}
    ]
    """
    out_dir = output or get_default_output_dir(dataset_dir)
    init_output_dir(dataset_dir, out_dir)

    with open(config) as f:
        tasks = json.load(f)

    click.echo(f"\n批量裁剪: 共 {len(tasks)} 个任务")
    click.echo(f"输出目录: {out_dir}")

    for i, task in enumerate(tasks):
        ep_idx = task["episode"]
        frame_expr = task["frames"]
        click.echo(f"\n{'='*50}")
        click.echo(f"[{i+1}/{len(tasks)}] Episode {ep_idx}, 删除帧: {frame_expr}")
        click.echo(f"{'='*50}")

        info = load_info(out_dir)
        episodes = load_episodes(out_dir)
        ep_entry = None
        for ep in episodes:
            if ep["episode_index"] == ep_idx:
                ep_entry = ep
                break
        if ep_entry is None:
            click.echo(f"  跳过: Episode {ep_idx} 不存在")
            continue

        try:
            del_start, del_end = parse_frame_range(frame_expr, ep_entry.get("length", 0))
        except click.BadParameter as e:
            click.echo(f"  跳过: {e}")
            continue

        do_trim(dataset_dir, out_dir, ep_idx, del_start, del_end)

    click.echo(f"\n✔ 批量裁剪完成")


@cli.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify(dataset_dir: Path):
    """验证数据集完整性"""
    info = load_info(dataset_dir)
    episodes = load_episodes(dataset_dir)
    errors = 0

    click.echo(f"正在验证: {dataset_dir}")
    click.echo(f"  共 {len(episodes)} 个 episode\n")

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep.get("length", 0)
        prefix = f"  Episode {ep_idx:3d}:"
        ep_ok = True

        pq_path = get_parquet_path(dataset_dir, ep_idx, info)
        if not pq_path.exists():
            click.echo(f"{prefix} ✗ Parquet 缺失")
            errors += 1
            continue
        df = pd.read_parquet(pq_path)
        if len(df) != ep_len:
            click.echo(f"{prefix} ✗ Parquet 帧数不匹配 (元数据: {ep_len}, 实际: {len(df)})")
            errors += 1
            ep_ok = False

        for vkey in get_video_keys(info):
            vpath = get_video_path(dataset_dir, ep_idx, vkey, info)
            if not vpath.exists():
                click.echo(f"{prefix} ✗ 视频缺失: {vkey}")
                errors += 1
                ep_ok = False

        for ikey in get_image_keys(info):
            idir = get_image_dir(dataset_dir, ep_idx, ikey)
            if idir.exists():
                img_count = len(list(idir.glob(f"*{IMAGE_EXTENSION}")))
                if img_count != ep_len:
                    click.echo(f"{prefix} ✗ 图片帧数不匹配: {ikey} (元数据: {ep_len}, 实际: {img_count})")
                    errors += 1
                    ep_ok = False
            else:
                click.echo(f"{prefix} ✗ 图片目录缺失: {ikey}")
                errors += 1
                ep_ok = False

        if ep_ok:
            click.echo(f"{prefix} ✓ 正常 ({ep_len} 帧)")

    actual_total = sum(ep.get("length", 0) for ep in episodes)
    if info.get("total_frames") != actual_total:
        click.echo(f"\n  ✗ total_frames 不匹配 (info.json: {info.get('total_frames')}, 实际: {actual_total})")
        errors += 1

    click.echo(f"\n{'✔ 验证通过，无错误' if errors == 0 else f'✗ 发现 {errors} 个错误'}")


if __name__ == "__main__":
    cli()